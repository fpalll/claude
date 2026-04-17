#!/bin/bash
#
# MMMU inference against a local OpenAI-compatible server (e.g. SGLang) with
# Qwen3.5-style prompts and sampling (matches official chat.completions example).
#
# Prompt: images first, then text with "Choices:\n(A) ..." and the JSON hint:
#   Please show your choice in the answer field with only the choice letter,
#   e.g., "answer": "C".
#
# Sampling: max_tokens=81920, temperature=1.0, top_p=0.95, presence_penalty=1.5,
#           extra_body.top_k=20 (see run_mmmu.py run_inference_openai).
#
# Override any of these with environment variables before running:
#   export API_BASE=http://localhost:50038
#   export SERVED_MODEL_NAME=qwen3_5-35b-a3b
#   export OPENAI_API_KEY=EMPTY   # or dummy; many local servers ignore it
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

API_BASE="${API_BASE:-http://localhost:50038}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_5-35b-a3b}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# Data root: directory containing MMMU_DEV_VAL.tsv (and images cache under images/MMMU)
DATA_DIR="${DATA_DIR:-/data/winglam/dataset/MMMU}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/results/qwen35_a3b_mmmu_predictions.jsonl}"

python run_mmmu.py infer \
  --api-base "${API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --data-dir "${DATA_DIR}" \
  --dataset MMMU_DEV_VAL \
  --output-file "${OUTPUT_FILE}" \
  --prompt-style qwen35 \
  --num-workers 8 \
  --request-timeout 600 \
  --max-new-tokens 81920 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0

echo "Wrote predictions to ${OUTPUT_FILE}"

# --- Optional: judge / accuracy (same server can be used as the extractor judge) ---
export MIT_SPIDER_URL='http://localhost:50038/v1/chat/completions'
export MIT_SPIDER_TOKEN='EMPTY'
python run_mmmu.py eval \
  --data-dir "${DATA_DIR}" \
  --input-file "${OUTPUT_FILE}" \
  --output-file "${SCRIPT_DIR}/results/qwen35_a3b_evaluation.csv" \
  --dataset MMMU_DEV_VAL \
  --eval-model "${SERVED_MODEL_NAME}" \
  --api-type mit \
  --nproc 16
# nohup bash infer_think.sh > a3b.log 2>&1 < /dev/null &