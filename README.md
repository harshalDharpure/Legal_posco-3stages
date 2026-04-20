#style 3-stage legal dialogue research pipeline (standalone)

This folder contains **only** the 3-stage pipeline we discussed:

- **Stage 1 (SFT)** → **Stage 2 (multi-objective)** → **Stage 3 (DPO)**
- Strict **train / validation / test** protocol (no test leakage)

## Directory layout

- `configs/`: default YAML (`pipeline_default.yaml`)
- `data/`: dataset sync + `prepare_splits.py`, `merge_train_val.py`
- `stage1_sft/`: masked causal LM SFT
- `stage2_multi_objective/`: \(L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}\)
- `stage3_dpo/`: TRL DPO
- `evaluation/`: `metrics.py`, `run_eval.py`, `stats.py`
- `ablation/`: Stage 2 ablation runner
- `logs/`: optional logs

## Quickstart

Run from the **repo root** (so relative paths work):

All commands below assume `python3` is available (on this machine `python` may not exist).

### Prerequisites (Hugging Face access + token)

You must have access to the base model repo:
`meta-llama/Meta-Llama-3.1-8B-Instruct`.

Set your Hugging Face token **in your shell** (do not write it into any file).

```bash
export HF_TOKEN="PASTE_YOUR_TOKEN_HERE"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Optional verification:

```bash
python3 -c "import os; print('HF_TOKEN' in os.environ, 'HUGGINGFACE_HUB_TOKEN' in os.environ)"
```

Alternative persistent login (writes to your user cache):

```bash
huggingface-cli login
```

### 0) Sync the dataset into this folder

```bash
python3 q1_3stage_pipeline/data/sync_dataset.py \
  --source-dir experiments/exp3_pretraining_finetuning/finetuning \
  --out-dir q1_3stage_pipeline/data/raw
```

### 1) Create splits (train/val/test)

```bash
python3 q1_3stage_pipeline/data/prepare_splits.py \
  --source q1_3stage_pipeline/data/raw/train.jsonl \
  --out-dir q1_3stage_pipeline/data/splits \
  --ratios 0.8 0.1 0.1 \
  --seed 42
```

### Using the NEW dialogue-level 70/10/20 split (recommended)

If you already created the dialogue-level split (NO pairs) with:
`data/create_70_10_20_split_dialogue_level.py`,
then use:

- `q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl`
- `q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl`
- `q1_3stage_pipeline/data/splits_dialogue_level/test_20_dialogues.jsonl`

The training code will flatten dialogues into (input, output) examples **in-memory**.

## STRICT experiment protocol (no leakage)

- **train / validation / test** splits are strict:
  - Train ONLY on train
  - Validation ONLY for tuning (hyperparams / early stopping / selecting β)
  - Test NEVER used until the very end
- After tuning is complete:
  - Create **final_train = train + validation**
  - Retrain **Stage 2 (M2)** and **Stage 3 (M3)** from scratch using `final_train`
  - Run evaluation **exactly once** on test

## Global formatting contract (must stay consistent)

All stages use the same strict prompt template:

```text
[USER]: {input}
[ASSISTANT]:
```

### 2) Stage 1 — SFT (M1)

```bash
python3 q1_3stage_pipeline/stage1_sft/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42 \
  --seed 42
```

### 3) Stage 2 — Multi-objective (M2)

Stage 2 is **initialized from M1** and trains:

\[
L = L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}
\]

- \(L_{entail}\): frozen DeBERTa-large MNLI teacher + KL distillation head (teacher forcing; no gradient through decoding)
- \(L_{triplet}\): dynamic hard negatives (model-gen + legal corruption + cross-sample) + SBERT filtering + hard mining

```bash
python3 q1_3stage_pipeline/stage2_multi_objective/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --init-from m1 \
  --m1-path q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42/final \
  --ablation full \
  --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_full_seed42 \
  --eval-every 50 \
  --seed 42
```

### 4) Stage 3 — DPO (M3)

Stage 3 runs DPO where:
- **chosen** = ground truth
- **rejected** = dynamic hard negatives (generated on-the-fly; not stored in the dataset)
- **reference model** = M2 (frozen)

```bash
python3 q1_3stage_pipeline/stage3_dpo/train.py \
  --m2-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_full_seed42/final \
  --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage3/M3_beta0.1_seed42 \
  --beta 0.1 \
  --seed 42
```

#### β sweep (required)

```bash
for beta in 0.1 0.5 1.0; do
  python3 q1_3stage_pipeline/stage3_dpo/train.py \
    --m2-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_fromM1_full_seed42/final \
    --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
    --output-dir "q1_3stage_pipeline/logs/checkpoints/stage3/M3_beta${beta}_seed42" \
    --beta "$beta" \
    --seed 42
done
```

### 5) Stage 2 ablations

```bash
python3 q1_3stage_pipeline/ablation/run_stage2_ablations.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl \
  --m1-path q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42/final \
  --out-root q1_3stage_pipeline/logs/checkpoints/stage2_ablations
```

### 6) Evaluation helper (reference/candidate pairs)

You must generate model outputs on the **test split** first, then run `run_eval.py`.
`pred-jsonl` must be in the same order as `test-jsonl` and contain either `candidate` or `output`.

```bash
python3 q1_3stage_pipeline/evaluation/run_eval.py \
  --test-jsonl q1_3stage_pipeline/data/splits_dialogue_level/test_20_dialogues.jsonl \
  --pred-jsonl q1_3stage_pipeline/logs/preds.jsonl
```

`run_eval.py` reports automatic metrics (ROUGE/BLEU/METEOR/NLI) plus:
- statute correctness proxies from `statutes_cited`
- safety/refusal proxy rates

## Multiple seeds (required)

Run every reported setting with **3 random seeds** (example: 42, 43, 44) and report mean/std.

Example for Stage 1:

```bash
for seed in 42 43 44; do
  python3 q1_3stage_pipeline/stage1_sft/train.py \
    --config q1_3stage_pipeline/configs/pipeline_default.yaml \
    --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
    --val-jsonl q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl \
    --output-dir "q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed${seed}" \
    --seed "$seed"
done
```

