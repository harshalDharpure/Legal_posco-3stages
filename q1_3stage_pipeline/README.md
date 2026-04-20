# Q1-style 3-stage legal dialogue research pipeline (standalone)

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

### 0) Sync the dataset into this folder

```bash
python q1_3stage_pipeline/data/sync_dataset.py \
  --source-dir experiments/exp3_pretraining_finetuning/finetuning \
  --out-dir q1_3stage_pipeline/data/raw
```

### 1) Create splits (train/val/test)

```bash
python q1_3stage_pipeline/data/prepare_splits.py \
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

### 2) Stage 1 — SFT (M1)

```bash
python q1_3stage_pipeline/stage1_sft/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage1/M1_seed42 \
  --seed 42
```

### 3) Stage 2 — Multi-objective (M2)

```bash
python q1_3stage_pipeline/stage2_multi_objective/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --init-from base \
  --ablation full \
  --train-jsonl q1_3stage_pipeline/data/splits_dialogue_level/train_70_dialogues.jsonl \
  --val-jsonl q1_3stage_pipeline/data/splits_dialogue_level/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage2/M2_base_full_seed42 \
  --seed 42
```

### 4) Stage 3 — DPO (M3)

Preferences JSONL format (one per line):

`{"prompt":"User: ...\nAssistant:","chosen":" ...","rejected":" ..."}`

```bash
python q1_3stage_pipeline/stage3_dpo/train.py \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-path q1_3stage_pipeline/logs/checkpoints/stage2/M2_base_full_seed42/adapter \
  --preferences q1_3stage_pipeline/data/preferences/preference_pairs.jsonl \
  --output-dir q1_3stage_pipeline/logs/checkpoints/stage3/M3_beta0.1_seed42 \
  --beta 0.1 \
  --seed 42
```

### 5) Stage 2 ablations

```bash
python q1_3stage_pipeline/ablation/run_stage2_ablations.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl q1_3stage_pipeline/data/splits/train.jsonl \
  --init-from base \
  --out-root q1_3stage_pipeline/logs/checkpoints/stage2_ablations
```

### 6) Evaluation helper (reference/candidate pairs)

```bash
python q1_3stage_pipeline/evaluation/run_eval.py \
  --test-jsonl q1_3stage_pipeline/data/splits/test.jsonl \
  --pred-jsonl q1_3stage_pipeline/logs/preds.jsonl
```

