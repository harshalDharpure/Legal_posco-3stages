#!/usr/bin/env bash
set -euo pipefail

# Run the full 3-stage pipeline in the background on a selected GPU.
# This is a thin wrapper around `q1_3stage_pipeline/run_full_pipeline.py`.
#
# Example (runs on GPU 1, logs to logs/pipeline_runs/):
#   nohup bash q1_3stage_pipeline/run_full_pipeline.sh --gpu 1 --seed 43 > /dev/null 2>&1 &
#
# Example (online mode for downloading models):
#   HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 nohup bash q1_3stage_pipeline/run_full_pipeline.sh --gpu 0 --seed 43 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE_DIR="${ROOT_DIR}/q1_3stage_pipeline"

GPU=""
SEED="43"
CONFIG="${PIPE_DIR}/configs/pipeline_default.yaml"
RUN_NAME=""
RESUME_STAGE2="0"

# Stage2 defaults (can be overridden by passing extra args after --)
STAGE2_ENTAIL_MAX_NEW_TOKENS="32"
STAGE2_ENTAIL_EVERY="2"
STAGE2_ENTAIL_CACHE_SIZE="4096"
STAGE2_CHECKPOINT_EVERY="10"
STAGE2_GRAD_ACCUM="8"
STAGE2_GRAD_CLIP="1.0"
STAGE2_SKIP_GRAD_NORM_THRESHOLD="0"
STAGE2_LOAD_IN_4BIT="1"
STAGE2_NLI_ON_CPU="1"

usage() {
  cat <<'EOF'
Usage:
  bash q1_3stage_pipeline/run_full_pipeline.sh [--gpu N] [--seed S] [--config PATH] [--run-name NAME] [--resume-stage2]

Notes:
  - This wrapper writes logs under q1_3stage_pipeline/logs/pipeline_runs/.
  - For background run, use nohup:
      nohup bash q1_3stage_pipeline/run_full_pipeline.sh --gpu 1 --seed 43 > /dev/null 2>&1 &
  - If you need online model downloading:
      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 nohup bash q1_3stage_pipeline/run_full_pipeline.sh --gpu 0 --seed 43 &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-43}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --run-name) RUN_NAME="${2:-}"; shift 2 ;;
    --resume-stage2) RESUME_STAGE2="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "${PIPE_DIR}/logs/pipeline_runs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PIPE_DIR}/logs/pipeline_runs/pipeline_${TS}.log"

export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Default to offline mode (safe on shared clusters); override by env if needed.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

CMD=(python3 -u "${PIPE_DIR}/run_full_pipeline.py"
  --config "${CONFIG}"
  --seed "${SEED}"
  --stage2-entail-max-new-tokens "${STAGE2_ENTAIL_MAX_NEW_TOKENS}"
  --stage2-entail-every "${STAGE2_ENTAIL_EVERY}"
  --stage2-entail-cache-size "${STAGE2_ENTAIL_CACHE_SIZE}"
  --stage2-checkpoint-every "${STAGE2_CHECKPOINT_EVERY}"
  --stage2-grad-accum "${STAGE2_GRAD_ACCUM}"
  --stage2-grad-clip "${STAGE2_GRAD_CLIP}"
  --stage2-skip-grad-norm-threshold "${STAGE2_SKIP_GRAD_NORM_THRESHOLD}"
)

if [[ -n "${GPU}" ]]; then
  CMD+=(--gpu "${GPU}")
fi
if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi
if [[ "${RESUME_STAGE2}" == "1" ]]; then
  CMD+=(--resume-stage2)
fi
if [[ "${STAGE2_LOAD_IN_4BIT}" == "1" ]]; then
  CMD+=(--stage2-load-in-4bit)
fi
if [[ "${STAGE2_NLI_ON_CPU}" == "1" ]]; then
  CMD+=(--stage2-nli-on-cpu)
fi

# Allow extra args after "--" to pass through to Python (future-proof).
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "Started at: $(date)"
  echo "Repo root : ${ROOT_DIR}"
  echo "Log file  : ${LOG_FILE}"
  echo "GPU       : ${GPU:-<not set>}"
  echo "Offline   : HF_HUB_OFFLINE=${HF_HUB_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
  echo "Command   : ${CMD[*]}"
  echo
} | tee -a "${LOG_FILE}"

("${CMD[@]}" 2>&1) | tee -a "${LOG_FILE}"

