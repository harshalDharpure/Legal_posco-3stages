# Three-Stage Legal Dialogue Training Pipeline Report (SFT → Multi-Objective → DPO)

Project: `q1_3stage_pipeline`  
Date: generated from `REPORT_3STAGE_PIPELINE.tex`

## Executive Summary
This report documents a strict **three-stage training pipeline** for a legal conversational model (Indian law: POCSO, IPC, CrPC) on **LLaMA-3.1-8B Instruct**. The stages are:

1. **Stage 1 (SFT)**: supervised fine-tuning on multi-turn dialogue pairs (teacher-forced causal LM).
2. **Stage 2 (Multi-objective)**: continue training from Stage 1 with a weighted combination of:
   - \(L_{\text{gen}}\) (LM loss)
   - \(L_{\text{entail}}\) (NLI teacher KL consistency)
   - \(L_{\text{triplet}}\) (embedding-space separation with dynamic hard negatives)
3. **Stage 3 (DPO)**: preference optimization using a reference model and dynamic rejected responses.

We additionally implemented **reproducibility, checkpointed resumption, dynamic negative mining, speed optimizations for entailment**, and **optimizer stability diagnostics** (AMP unscale + clipping, gradient sanity stats, update ratio proxy, contribution ratios).

## Dataset and Splits

### Data characteristics
The dataset consists of multi-turn legal dialogues with Hindi/English code-mixing and Indian law domain constraints (statutes, procedures, and safety).

### Split protocol
Training is dialogue-level split to avoid leakage across turns:

- **Train**: `q1_3stage_pipeline/data/train_70_dialogues.jsonl`
- **Val**: `q1_3stage_pipeline/data/val_10_dialogues.jsonl`
- **Test**: `q1_3stage_pipeline/data/test_20_dialogues.jsonl`

Each dialogue JSONL record contains `turns`. Training code flattens dialogues into (prompt, response) examples in-memory.

## Model and Tokenization

### Base model
Default base model (from config):

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Prompt format
Prompts use a strict user/assistant delimiter format:

```text
[USER]: ...
[ASSISTANT]:
```

The assistant span is the supervised target; prompt tokens are masked in labels (`-100`).

### Code-mixed language tag (optional)
To improve conditioning in code-mixed legal text, Stage 2 supports an optional prefix:

```bash
--lang-tag-prefix "[HI_EN_LEGAL]"
```

This prepends the tag to every input prompt (train and val) without changing dataset files.

## Stage 1: Supervised Fine-Tuning (SFT)

### Objective
Standard teacher-forced causal LM cross-entropy:

\[
L_{\text{gen}} = \text{CE}(\text{LM}(x), y)
\]

where \(x\) is the prompt and \(y\) is the assistant completion. Prompt tokens are masked out in the label tensor.

### Artifacts
Stage 1 outputs a Hugging Face-style `final/` directory that Stage 2 uses for initialization.

## Stage 2: Multi-Objective Training

### High-level objective
Stage 2 optimizes a weighted sum:

\[
L = L_{\text{gen}} + \lambda_e \, w_e(t)\,L_{\text{entail}} + \lambda_t \, w_t(t)\,L_{\text{triplet}}
\]

#### Dynamic weight scheduling
To avoid early instability from entailment spikes, the schedule uses step-progress \(t \in [0,1]\):

- **Early** (first 30%): \(w_e=0.3,\; w_t=0.3\)
- **Mid**: \(w_e=0.7,\; w_t=0.5\)
- **Late**: \(w_e=1.0,\; w_t=0.7\)

### \(L_{\text{entail}}\): NLI teacher KL
We use a frozen DeBERTa MNLI teacher to produce a 3-way distribution (contradiction, neutral, entailment) for:

\[
(\text{premise}=y^+,\; \text{hypothesis}=\hat{y})
\]

where \(y^+\) is the reference answer and \(\hat{y}\) is the model's greedy completion for the prompt.

The student head predicts logits from pooled LM hidden states; the loss is:

\[
L_{\text{entail}} = \text{KL}\big(p_{\text{teacher}} \,\|\, p_{\text{student}}\big)
\]

#### Cost reductions
To reduce training time:

- **Shorter greedy decode**: default `--entail-max-new-tokens 48` (recommended: 32–48)
- **Sparse computation**: `--entail-every N` computes entailment only every \(N\) optimizer steps (default: 2)
- **LRU caching**: `--entail-cache-size` caches greedy \(\hat{y}\) per prompt to avoid recomputation
- **Teacher device**: `--nli-on-cpu` runs DeBERTa on CPU to save VRAM (slower but safer on shared GPUs)

### \(L_{\text{triplet}}\): dynamic hard negatives + margin loss

#### Dynamic negative generation
Negatives are **not stored in dataset files**. For each (prompt \(x\), positive \(y^+\)) we form candidates:

- **Model negative**: sampled generation from the current policy
- **Rule corruption**: statute/procedure contradiction (e.g., IPC section swaps)
- **Cross-sample**: response from a different dialogue

#### Hard / semi-hard mining
We embed candidates with a frozen sentence encoder (SBERT/MPNet) and select a hard negative that is:

- sufficiently similar to the prompt (hard)
- less similar than the positive
- optionally constrained to a **semi-hard band** relative to \(\text{sim}(x,y^+)\)

#### Loss
Anchor is a learned projection of pooled LM hidden state; positives and negatives are sentence embeddings. With normalized vectors:

\[
L_{\text{triplet}} = \mathbb{E}\left[\max(0, d(a,p)-d(a,n)+m)\right]
\]

Default margin was reduced to \(m=0.2\) (config can override) to reduce saturation.

### Checkpointing and resume
Stage 2 writes a resumable checkpoint:

```text
output_dir/checkpoints/latest.pt
```

We optimized checkpoint size for large models by saving **PEFT adapter weights** (when applicable) rather than full 8B parameters.

### Training stability (AMP + gradients)

#### Correct grad norm under AMP
With `GradScaler`, gradients are scaled. We therefore:

- call `scaler.unscale_(optimizer)` before measuring/clipping
- clip gradients using `--grad-clip-max-norm` (default 1.0)
- log gradient sanity stats: `min_grad`, `max_grad`, `grad_norm`

#### Optional skip threshold
The “skip step” behavior is **disabled by default**. If desired:

```bash
--skip-grad-norm-threshold 300
```

This skips optimizer updates when the (unscaled, clipped) norm exceeds the threshold.

#### Update ratio proxy
We log:

\[
\text{update\_ratio} \approx \frac{\text{lr}\cdot\lVert g\rVert}{\lVert \theta\rVert}
\]

for trainable parameters, providing a compact stability signal.

#### Contribution ratios
Logs include per-step contributions:

- `loss_contrib_gen`
- `loss_contrib_entail` (already includes schedule and \(\lambda_e\))
- `loss_contrib_triplet` (includes schedule and \(\lambda_t\))
- `triplet_contrib_frac` = triplet contribution / total

### Logging and evaluation hooks
Training writes JSONL logs:

```text
output_dir/train_log.jsonl
```

Key fields include:

- step counters: `epoch`, `opt_step`, `micro_step`
- losses: `loss`, `loss_gen`, `loss_entail`, `loss_triplet`
- stability: `grad_norm`, `min_grad`, `max_grad`, `scaler_scale`, `update_ratio`
- rolling stats (window=50): moving averages and spike/saturation rates

#### Fixed qualitative evaluation
Every `--fixed-eval-every` optimizer steps (default 200), we run 5 fixed prompts and log:

- generated output
- teacher entailment probability (entailment label probability)

This provides a stable view of reasoning improvement during training.

### Debug-fast mode
`--debug-fast` runs a cheap end-to-end sanity pass:

- 100 training samples
- triplet disabled
- entailment computed every 5 steps
- generation caps reduced to 16 tokens

## Stage 3: DPO Preference Optimization

### Goal
Stage 3 performs preference optimization using:

- **Policy**: Stage 2 final model
- **Reference**: frozen copy of Stage 2 (or saved reference)
- **Chosen**: ground truth \(y^+\)
- **Rejected**: dynamic hard negatives \(y^-\) (same family as Stage 2)

### Beta sweep
We support sweeping \(\beta \in \{0.1, 0.5, 1.0\}\) (config) to tune the preference strength.

## Reproducibility and Project Layout

### Reproducibility
We set deterministic seeds and save run configuration files:

- `config_used.yaml`
- `run_args.json`

for each training run directory under `q1_3stage_pipeline/logs/checkpoints/`.

### Key directories

- `q1_3stage_pipeline/configs/`: YAML configuration
- `q1_3stage_pipeline/data/`: JSONL splits (dialogue-level)
- `q1_3stage_pipeline/logs/checkpoints/stage1/`: Stage 1 checkpoints
- `q1_3stage_pipeline/logs/checkpoints/stage2/`: Stage 2 checkpoints + logs
- `q1_3stage_pipeline/logs/checkpoints/stage3/`: Stage 3 outputs (DPO)

## Operational Notes

### Long epoch duration
Stage 2 is slow because it includes:

- on-the-fly generation for negatives
- greedy decoding for entailment supervision
- NLI teacher forward pass
- embedding encoder for triplet mining

The `--entail-every`, caching, and shorter `--entail-max-new-tokens` significantly reduce cost.

### Gated model warnings
If the environment lacks Hugging Face access to gated model metadata, warnings about fetching remote config files may appear. Training can still proceed if model weights are already available locally.

## Recommended Defaults (Stage 2)

```bash
--entail-max-new-tokens 32   # or 48
--entail-every 2             # or 3 if still slow
--entail-cache-size 2048
--grad-clip-max-norm 1.0
--skip-grad-norm-threshold 0 # disabled
```

