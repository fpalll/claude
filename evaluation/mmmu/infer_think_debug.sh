#!/bin/bash
#
# DEBUG duplicate of infer_think.sh: same OpenAI-compatible MMMU path, but
# run_mmmu.py infer subsamples the dataset (--max-samples / --sample-seed).
#
# Prompt: images first, then text with "Choices:\n(A) ..." and the JSON hint:
#   Please show your choice in the answer field with only the choice letter,
#   e.g., "answer": "C".
#
# Sampling (default run): max_tokens=81920, temperature=1.0, top_p=0.95,
# presence_penalty=1.5, extra_body.top_k=20 (run_mmmu.py OpenAI path).
# For Qwen3.6-27B, HF README recommends presence_penalty=0.0 — see per-model
# command templates in the comments below.
#
# Subsample (debug): set MAX_SAMPLES (default 32) and SAMPLE_SEED (default 0).
#
# Override any of these with environment variables before running:
#   export API_BASE=http://localhost:50038
#   export SERVED_MODEL_NAME=qwen3_5-35b-a3b
#   export OPENAI_API_KEY=EMPTY   # or dummy; many local servers ignore it
#   export MAX_SAMPLES=16
#   export SAMPLE_SEED=42
#   export NO_THINKING=0          # omit --no-thinking; use server default (often thinking on)
#   export REQUEST_TIMEOUT=7200   # default 3600; raise if you still see API timeouts
#   export NUM_WORKERS=4          # reduce GPU queueing if requests time out
#
# ---------------------------------------------------------------------------
# HF README — thinking mode, general tasks (Chat Completions–style sampling).
# Append to each line block: --max-samples "${MAX_SAMPLES}" --sample-seed "${SAMPLE_SEED}"
# Replace API_BASE / DATA_DIR / SERVED_MODEL_NAME / OUTPUT_FILE. HF also
# lists min_p=0.0 (not a run_mmmu.py flag; set on the server if supported).
# ---------------------------------------------------------------------------
#
# Qwen3.5-VL-35B-A3B-Instruct (Hub: Qwen/Qwen3.5-VL-35B-A3B-Instruct; gated):
#   python run_mmmu.py infer \
#     --api-base "${API_BASE}" --api-key "${OPENAI_API_KEY}" \
#     --served-model-name "${SERVED_MODEL_NAME}" \
#     --data-dir "${DATA_DIR}" --dataset MMMU_DEV_VAL --output-file "${OUTPUT_FILE}" \
#     --prompt-style qwen35 --num-workers 8 --request-timeout 600 \
#     --max-new-tokens 81920 --temperature 1.0 --top-p 0.95 --top-k 20 \
#     --presence-penalty 1.5 --repetition-penalty 1.0 \
#     --max-samples "${MAX_SAMPLES}" --sample-seed "${SAMPLE_SEED}"
#
# Qwen3.6-35B-A3B (Hub: Qwen/Qwen3.6-35B-A3B — same recipe as HF README):
#   python run_mmmu.py infer \
#     --api-base "${API_BASE}" --api-key "${OPENAI_API_KEY}" \
#     --served-model-name "${SERVED_MODEL_NAME}" \
#     --data-dir "${DATA_DIR}" --dataset MMMU_DEV_VAL --output-file "${OUTPUT_FILE}" \
#     --prompt-style qwen35 --num-workers 8 --request-timeout 600 \
#     --max-new-tokens 81920 --temperature 1.0 --top-p 0.95 --top-k 20 \
#     --presence-penalty 1.5 --repetition-penalty 1.0 \
#     --max-samples "${MAX_SAMPLES}" --sample-seed "${SAMPLE_SEED}"
#
# Qwen3.5-27B (Hub: Qwen/Qwen3.5-27B — README thinking/general matches 35B-A3B;
#   generation_config.json alone uses temperature=0.6 for default decoding):
#   python run_mmmu.py infer \
#     --api-base "${API_BASE}" --api-key "${OPENAI_API_KEY}" \
#     --served-model-name "${SERVED_MODEL_NAME}" \
#     --data-dir "${DATA_DIR}" --dataset MMMU_DEV_VAL --output-file "${OUTPUT_FILE}" \
#     --prompt-style qwen35 --num-workers 8 --request-timeout 600 \
#     --max-new-tokens 81920 --temperature 1.0 --top-p 0.95 --top-k 20 \
#     --presence-penalty 1.5 --repetition-penalty 1.0 \
#     --max-samples "${MAX_SAMPLES}" --sample-seed "${SAMPLE_SEED}"
#
# Qwen3.6-27B (Hub: Qwen/Qwen3.6-27B — HF README sets presence_penalty=0.0 for
#   thinking / general; differs from Qwen3.6-35B-A3B README):
#   python run_mmmu.py infer \
#     --api-base "${API_BASE}" --api-key "${OPENAI_API_KEY}" \
#     --served-model-name "${SERVED_MODEL_NAME}" \
#     --data-dir "${DATA_DIR}" --dataset MMMU_DEV_VAL --output-file "${OUTPUT_FILE}" \
#     --prompt-style qwen35 --num-workers 8 --request-timeout 600 \
#     --max-new-tokens 81920 --temperature 1.0 --top-p 0.95 --top-k 20 \
#     --presence-penalty 0.0 --repetition-penalty 1.0 \
#     --max-samples "${MAX_SAMPLES}" --sample-seed "${SAMPLE_SEED}"
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

API_BASE="${API_BASE:-http://localhost:50040}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_6_27b}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Data root: directory containing MMMU_DEV_VAL.tsv (and images cache under images/MMMU)
DATA_DIR="${DATA_DIR:-/data/winglam/qwen_a3b/MMMU}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/results/qwen36_27_mmmu_predictions_debug.jsonl}"

# Random subset of the TSV for quick runs (see run_mmmu.py --max-samples / --sample-seed)
MAX_SAMPLES="${MAX_SAMPLES:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

# Match MMLU-Pro evaluate_amend.py --no-thinking: vLLM extra_body chat_template_kwargs.
NO_THINKING="${NO_THINKING:-1}"
NO_THINKING_ARGS=()
if [[ "${NO_THINKING}" != "0" ]]; then
  NO_THINKING_ARGS=(--no-thinking)
fi

REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python run_mmmu.py infer \
  --api-base "${API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --data-dir "${DATA_DIR}" \
  --dataset MMMU_DEV_VAL \
  --output-file "${OUTPUT_FILE}" \
  --prompt-style qwen35 \
  --num-workers "${NUM_WORKERS}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --max-new-tokens 81920 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --presence-penalty 0.0 \
  --repetition-penalty 1.0 \
  --max-samples "${MAX_SAMPLES}" \
  --sample-seed "${SAMPLE_SEED}" \
  "${NO_THINKING_ARGS[@]}"

echo "Wrote predictions to ${OUTPUT_FILE}"

# --- Optional: judge / accuracy (same server can be used as the extractor judge) ---
export MIT_SPIDER_URL="${MIT_SPIDER_URL:-${API_BASE}/v1/chat/completions}"
export MIT_SPIDER_TOKEN="${MIT_SPIDER_TOKEN:-EMPTY}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${SCRIPT_DIR}/results/qwen36_27b_evaluation_debug.csv}"
python run_mmmu.py eval \
  --data-dir "${DATA_DIR}" \
  --input-file "${OUTPUT_FILE}" \
  --output-file "${EVAL_OUTPUT}" \
  --dataset MMMU_DEV_VAL \
  --eval-model "${SERVED_MODEL_NAME}" \
  --api-type mit \
  --nproc 16
# nohup bash infer_think_debug.sh > 36_27b_debug.log 2>&1 < /dev/null &
