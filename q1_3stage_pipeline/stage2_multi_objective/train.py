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
from q1_3stage_pipeline.stage2_multi_objective.hard_negatives import (
    corrupt_legal_text,
    cross_sample_negative,
    model_negative_generate,
    select_hard_negative,
)
from q1_3stage_pipeline.stage2_multi_objective.losses import (
    FrozenSentenceEncoder,
    EntailmentStudentHead,
    FrozenNLITeacher,
    kl_teacher_student,
    pooled_assistant_hidden,
    triplet_margin_loss,
)
from q1_3stage_pipeline.utils import load_jsonl, set_global_seed


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_run_config(dst_dir: str, cfg_path: str, extra: dict) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        with open(cfg_path, encoding="utf-8") as src:
            f.write(src.read())
    with open(os.path.join(dst_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2, ensure_ascii=False)


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
    ap.add_argument("--init-from", choices=["base", "m1", "exp3"], default="m1")
    ap.add_argument("--m1-path", default="", help="Path to Stage1 'final' HF folder (required if --init-from m1).")
    ap.add_argument("--ablation", choices=["gen_only", "gen_entail", "gen_triplet", "full"], default="full")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=0, help="If >0, run validation every N optimizer steps.")
    args = ap.parse_args()

    set_global_seed(args.seed)
    cfg_path = os.path.join(_REPO, args.config) if not os.path.isabs(args.config) else args.config
    cfg = load_yaml(cfg_path)
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
    val_rows = load_jsonl(val_path) if val_path and os.path.isfile(val_path) else []

    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.init_from == "m1":
        if not args.m1_path:
            raise SystemExit("--m1-path is required when --init-from m1")
        m1 = args.m1_path if os.path.isabs(args.m1_path) else str(_REPO / args.m1_path)
        model = AutoModelForCausalLM.from_pretrained(m1, torch_dtype=torch.float16, device_map="auto")
    elif args.init_from == "exp3" and exp3_path:
        model = load_init_from_exp3(exp3_path)
    else:
        # Fallback: base model (QLoRA) if you want quick experiments.
        model = load_llama_qlora_model_only(base)

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    st_model = FrozenSentenceEncoder(emb_name)
    st_dim = st_model.encoder.get_sentence_embedding_dimension()
    triplet_proj = torch.nn.Linear(hidden_size, st_dim).to(device)
    entail_head = EntailmentStudentHead(hidden_size).to(device)
    nli_teacher = FrozenNLITeacher("microsoft/deberta-large-mnli")
    nli_device = device  # run teacher on same device if possible

    lr = float(train_cfg.get("learning_rate", 5e-5))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(triplet_proj.parameters()) + list(entail_head.parameters()),
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
    best_val = None
    best_dir = os.path.join(args.output_dir, "best")
    _save_run_config(
        args.output_dir,
        cfg_path,
        {
            "stage": "stage2_multi_objective",
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "init_from": args.init_from,
            "m1_path": args.m1_path,
            "ablation": args.ablation,
            "lambda_entail": lam_e,
            "lambda_triplet": lam_t,
            "triplet_margin": margin,
            "embedding_model": emb_name,
            "nli_teacher": "microsoft/deberta-large-mnli",
            "eval_every": args.eval_every,
        },
    )

    def evaluate(rows: list[dict[str, Any]]) -> dict[str, float]:
        """Validation pass: no weight updates, no negative caching to disk."""
        if not rows:
            return {}
        model.eval()
        triplet_proj.eval()
        entail_head.eval()
        ds = LegalSFTDataset(rows, tokenizer, max_len, return_row_index=True)
        examples = ds.examples
        dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=lambda b: collate_sft_batch(b, tokenizer), num_workers=0)
        totals = {"loss": 0.0, "loss_gen": 0.0, "loss_entail": 0.0, "loss_triplet": 0.0}
        n_batches = 0
        with torch.no_grad():
            for batch in dl:
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

                sub_rows = [examples[ri] for ri in row_indices]
                refs: list[str] = [r.get("output", "").strip() for r in sub_rows]
                prompts: list[str] = [str(r.get("input", "")).strip() for r in sub_rows]

                negs: list[str] = []
                for j, r in enumerate(sub_rows):
                    x = prompts[j]
                    y_pos = refs[j]
                    # Use deterministic corruption + cross-sample for val; avoid slow generation here.
                    cand2 = corrupt_legal_text(y_pos)
                    cand3 = cross_sample_negative(examples, rng, avoid_dialogue_id=str(r.get("dialogue_id", "")))
                    y_neg = select_hard_negative(
                        x=x,
                        y_pos=y_pos,
                        candidates=[cand2, cand3],
                        sentence_encoder=st_model.encoder,
                        sim_high_threshold=0.2,
                    )
                    negs.append(y_neg)

                ref_emb = st_model.encode_texts(refs, torch.device("cpu")).to(device)
                neg_emb = st_model.encode_texts(negs, torch.device("cpu")).to(device)

                # Teacher probs using greedy decode (no sampling)
                y_hats: list[str] = []
                for x in prompts:
                    inputs = tokenizer(x, return_tensors="pt", add_special_tokens=False).to(device)
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_ids = gen[0, inputs["input_ids"].shape[1] :]
                    y_hats.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                teacher_p = nli_teacher.probs(refs, y_hats, device=nli_device).to(device)
                student_logits = entail_head(pooled)
                loss_e = kl_teacher_student(teacher_p, student_logits)

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

                totals["loss"] += float(loss.item())
                totals["loss_gen"] += float(loss_gen.item())
                totals["loss_entail"] += float(loss_e.item())
                totals["loss_triplet"] += float(loss_tr.item())
                n_batches += 1
        if n_batches == 0:
            return {}
        return {k: v / n_batches for k, v in totals.items()}

    def run_epoch(epoch: int) -> float:
        nonlocal best_val
        model.train()
        triplet_proj.train()
        entail_head.train()
        total = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        step_i = 0

        opt_steps = 0
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
            batch_prompts: list[str] = [str(r.get("input", "")).strip() for r in sub_rows]

            # Dynamic hard negatives (DO NOT STORE in dataset):
            # 1) model negative
            # 2) contradictory legal corruption of y+
            # 3) cross-sample negative from another dialogue
            batch_negs: list[str] = []
            for j, r in enumerate(sub_rows):
                x = batch_prompts[j]
                y_pos = batch_refs[j]
                cand1 = model_negative_generate(model, tokenizer, x, device, rng)
                cand2 = corrupt_legal_text(y_pos)
                cand3 = cross_sample_negative(train_examples, rng, avoid_dialogue_id=str(r.get("dialogue_id", "")))

                # Filter + hard mine using frozen Sentence-BERT similarities.
                y_neg = select_hard_negative(
                    x=x,
                    y_pos=y_pos,
                    candidates=[cand1, cand2, cand3],
                    sentence_encoder=st_model.encoder,
                    sim_high_threshold=0.2,
                )
                batch_negs.append(y_neg)

            with torch.no_grad():
                ref_emb = st_model.encode_texts(batch_refs, torch.device("cpu")).to(device)
                neg_emb = st_model.encode_texts(batch_negs, torch.device("cpu")).to(device)

            # L_entail (spec): DeBERTa-large MNLI teacher KL.
            # premise = ground truth (y+), hypothesis = model output (y_hat).
            # Teacher forcing only; no sampling gradients (we decode y_hat under no_grad).
            with torch.no_grad():
                # Greedy decode using the prompt only.
                # Note: prompts already end with "[ASSISTANT]:"
                y_hats: list[str] = []
                for x in batch_prompts:
                    inputs = tokenizer(x, return_tensors="pt", add_special_tokens=False).to(device)
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_ids = gen[0, inputs["input_ids"].shape[1] :]
                    y_hats.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                teacher_p = nli_teacher.probs(batch_refs, y_hats, device=nli_device).to(device)
            student_logits = entail_head(pooled)
            loss_e = kl_teacher_student(teacher_p, student_logits)
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
                opt_steps += 1

                if args.eval_every and val_rows and (opt_steps % int(args.eval_every) == 0):
                    metrics = evaluate(val_rows)
                    if metrics:
                        log_f.write(json.dumps({"epoch": epoch, "opt_step": opt_steps, **{f"val_{k}": v for k, v in metrics.items()}}) + "\n")
                        log_f.flush()
                        cur = float(metrics.get("loss", 0.0))
                        if best_val is None or cur < best_val:
                            best_val = cur
                            os.makedirs(best_dir, exist_ok=True)
                            tokenizer.save_pretrained(best_dir)
                            model.save_pretrained(best_dir)
                            torch.save(triplet_proj.state_dict(), os.path.join(best_dir, "triplet_proj.pt"))
                            torch.save(entail_head.state_dict(), os.path.join(best_dir, "entail_head.pt"))

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

    # Save full final checkpoint (M2) for Stage 3 reference/policy init.
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    tokenizer.save_pretrained(final_dir)
    model.save_pretrained(final_dir)
    triplet_proj.cpu()
    torch.save(triplet_proj.state_dict(), os.path.join(final_dir, "triplet_proj.pt"))
    entail_head.cpu()
    torch.save(entail_head.state_dict(), os.path.join(final_dir, "entail_head.pt"))
    with open(os.path.join(final_dir, "stage2_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "init_from": args.init_from,
                "m1_path": args.m1_path,
                "ablation": args.ablation,
                "lambda_entail": lam_e,
                "lambda_triplet": lam_t,
                "embedding_model": emb_name,
                "nli_teacher": "microsoft/deberta-large-mnli",
            },
            f,
            indent=2,
        )
    log_f.close()
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()

