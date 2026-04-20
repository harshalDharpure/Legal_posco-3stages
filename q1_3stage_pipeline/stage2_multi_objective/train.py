#!/usr/bin/env python3
"""
Stage 2: multi-objective training (L_gen + λ1 L_entail + λ2 L_triplet).

Example:
  python q1_3stage_pipeline/stage2_multi_objective/train.py \
    --config q1_3stage_pipeline/configs/pipeline_default.yaml \
    --init-from base \
    --ablation full \
    --train-jsonl q1_3stage_pipeline/data/splits/train.jsonl \
    --val-jsonl q1_3stage_pipeline/data/splits/val.jsonl \
    --output-dir q1_3stage_pipeline/logs/checkpoints/stage2/M2_base_full_seed42 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from q1_3stage_pipeline.stage1_sft.dataset import LegalSFTDataset, collate_sft_batch
from q1_3stage_pipeline.stage2_multi_objective.hard_negatives import get_negative_output
from q1_3stage_pipeline.stage2_multi_objective.losses import (
    FrozenSentenceEncoder,
    entailment_cosine_loss,
    pooled_assistant_hidden,
    triplet_margin_loss,
)
from q1_3stage_pipeline.utils import load_jsonl, set_global_seed


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_llama_qlora_model_only(base_name: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb,
        device_map="auto",
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora)


def load_init_from_exp3(exp3_rel: str):
    """Load merged Exp3 HF folder, then attach a new LoRA for Stage 2."""
    path = _REPO / exp3_rel
    if not path.is_dir():
        raise FileNotFoundError(f"exp3 checkpoint not found: {path}")
    print(f"Loading merged Exp3 weights from {path}")
    m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float16, device_map="auto")
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(m, lora)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", choices=["base", "exp3"], default="base")
    ap.add_argument("--ablation", choices=["gen_only", "gen_entail", "gen_triplet", "full"], default="full")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-epochs", type=int, default=1)
    args = ap.parse_args()

    set_global_seed(args.seed)
    cfg = load_yaml(os.path.join(_REPO, args.config) if not os.path.isabs(args.config) else args.config)
    base = cfg["project"]["base_model"]
    exp3_path = cfg["project"].get("exp3_checkpoint", "")
    s2 = cfg.get("stage2", {})
    lam_e = float(s2.get("lambda_entail", 0.5))
    lam_t = float(s2.get("lambda_triplet", 0.5))
    margin = float(s2.get("triplet_margin", 0.3))
    emb_name = s2.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
    train_cfg = cfg.get("training", {})

    train_rows = load_jsonl(os.path.join(_REPO, args.train_jsonl) if not os.path.isabs(args.train_jsonl) else args.train_jsonl)
    val_path = args.val_jsonl
    if val_path and not os.path.isabs(val_path):
        val_path = str(_REPO / val_path)
    _ = load_jsonl(val_path) if val_path and os.path.isfile(val_path) else []

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.init_from == "exp3" and exp3_path:
        model = load_init_from_exp3(exp3_path)
    else:
        model = load_llama_qlora_model_only(base)

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    st_model = FrozenSentenceEncoder(emb_name)
    st_dim = st_model.encoder.get_sentence_embedding_dimension()
    triplet_proj = torch.nn.Linear(hidden_size, st_dim).to(device)

    lr = float(train_cfg.get("learning_rate", 5e-5))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(triplet_proj.parameters()),
        lr=lr,
    )

    max_len = int(train_cfg.get("max_length", 512))
    # LegalSFTDataset supports dialogue-level rows (with `turns`) and will flatten them
    # into (input, output) examples in-memory. We then use those flattened examples
    # for entailment/triplet refs/negs.
    train_ds = LegalSFTDataset(train_rows, tokenizer, max_len, return_row_index=True)
    train_examples = train_ds.examples
    bs = max(1, int(train_cfg.get("per_device_batch_size", 1)))
    ga = max(1, int(train_cfg.get("gradient_accumulation_steps", 8)))
    loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, tokenizer),
        num_workers=0,
    )

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_f = open(os.path.join(args.output_dir, "train_log.jsonl"), "w", encoding="utf-8")

    def run_epoch(epoch: int) -> float:
        model.train()
        triplet_proj.train()
        total = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        step_i = 0

        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            row_indices = batch.pop("_row_indices")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
                output_hidden_states=True,
            )
            loss_gen = out.loss
            hidden = out.hidden_states[-1]
            mask = (labels != -100).long()
            pooled = pooled_assistant_hidden(hidden, mask).float()

            # row_indices refer to the flattened examples inside LegalSFTDataset.
            sub_rows = [train_examples[ri] for ri in row_indices]
            batch_refs: list[str] = [r.get("output", "").strip() for r in sub_rows]
            batch_negs: list[str] = [get_negative_output(sub_rows, j, rng) for j in range(len(sub_rows))]

            with torch.no_grad():
                ref_emb = st_model.encode_texts(batch_refs, torch.device("cpu")).to(device)
                neg_emb = st_model.encode_texts(batch_negs, torch.device("cpu")).to(device)

            loss_e = entailment_cosine_loss(pooled, ref_emb)
            anchor = triplet_proj(pooled)
            loss_tr = triplet_margin_loss(anchor, ref_emb, neg_emb, margin=margin)

            if args.ablation == "gen_only":
                loss = loss_gen
            elif args.ablation == "gen_entail":
                loss = loss_gen + lam_e * loss_e
            elif args.ablation == "gen_triplet":
                loss = loss_gen + lam_t * loss_tr
            else:
                loss = loss_gen + lam_e * loss_e + lam_t * loss_tr

            loss = loss / ga
            loss.backward()
            total += loss.item() * ga
            n += 1
            step_i += 1

            if step_i % ga == 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(triplet_proj.parameters()), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            log_f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(loss.item() * ga),
                        "loss_gen": float(loss_gen.item()),
                        "loss_entail": float(loss_e.item()),
                        "loss_triplet": float(loss_tr.item()),
                    }
                )
                + "\n"
            )
            log_f.flush()

        if step_i % ga != 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(triplet_proj.parameters()), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return total / max(n, 1)

    for ep in range(args.num_epochs):
        avg = run_epoch(ep)
        print(f"epoch {ep} avg_loss={avg:.4f}")

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    triplet_proj.cpu()
    torch.save(triplet_proj.state_dict(), os.path.join(args.output_dir, "triplet_proj.pt"))
    with open(os.path.join(args.output_dir, "stage2_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "init_from": args.init_from,
                "ablation": args.ablation,
                "lambda_entail": lam_e,
                "lambda_triplet": lam_t,
                "embedding_model": emb_name,
            },
            f,
            indent=2,
        )
    log_f.close()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()

