#!/bin/bash
#
# VideoMME evaluation only: score predictions from infer_think.sh (or any compatible JSONL).
# Uses an OpenAI-compatible judge (same pattern as MMMU eval): set MIT_SPIDER_URL to the
# full chat/completions endpoint.
#
#   export DATA_DIR=/path/to/Video-MME   # default below: /data/winglam/dataset/Video-MME
#   export INPUT_FILE=./results/videomme_short_wo_subtitle_qwen35.jsonl
#   export SERVED_MODEL_NAME=qwen3_5-35b-a3b
#   export MIT_SPIDER_URL=http://localhost:50038/v1/chat/completions
#   export MIT_SPIDER_TOKEN=EMPTY
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-/data/winglam/dataset/Video-MME}"
INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/results/videomme_short_wo_subtitle_qwen35.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/results/videomme_short_wo_subtitle_qwen35_evaluation.csv}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_5-35b-a3b}"
EVAL_NPROC="${EVAL_NPROC:-16}"

export MIT_SPIDER_URL="${MIT_SPIDER_URL:-http://localhost:50038/v1/chat/completions}"
export MIT_SPIDER_TOKEN="${MIT_SPIDER_TOKEN:-EMPTY}"
export MIT_SPIDER_TIMEOUT="${MIT_SPIDER_TIMEOUT:-600}"

python run_videomme.py eval \
  --data-dir "${DATA_DIR}" \
  --input-file "${INPUT_FILE}" \
  --output-file "${OUTPUT_FILE}" \
  --eval-model "${SERVED_MODEL_NAME}" \
  --api-type mit \
  --nproc "${EVAL_NPROC}"

echo "Wrote ${OUTPUT_FILE}"
