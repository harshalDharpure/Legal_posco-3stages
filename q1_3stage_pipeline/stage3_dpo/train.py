#!/usr/bin/env python3
"""
Stage 3: DPO alignment (TRL) on top of M2.

Preferences JSONL lines:
  {"prompt": "User: ...\\nAssistant:", "chosen": " ...", "rejected": " ..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def load_prefs(path: str) -> dict:
    prompts, chosen, rejected = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            chosen.append(row["chosen"])
            rejected.append(row["rejected"])
    return {"prompt": prompts, "chosen": chosen, "rejected": rejected}


def main() -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise SystemExit(f"Install trl peft datasets: {e}") from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--model-path",
        required=True,
        help="Merged HF checkpoint dir, or PEFT adapter folder (with base loaded from base-model)",
    )
    ap.add_argument("--preferences", required=True, help="JSONL prompt, chosen, rejected")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=1536)
    args = ap.parse_args()

    raw = load_prefs(args.preferences if os.path.isabs(args.preferences) else str(_REPO / args.preferences))
    dataset = Dataset.from_dict(raw)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
    )

    from peft import PeftModel

    if os.path.isfile(os.path.join(args.model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb,
            device_map="auto",
        )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_steps=200,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tok,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=dataset,
            tokenizer=tok,
        )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tok.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Saved", os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()

