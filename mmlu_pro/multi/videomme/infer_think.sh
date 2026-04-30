#!/bin/bash
#
# VideoMME inference via a local OpenAI-compatible server (e.g. SGLang / vLLM) using
# Qwen3.5-style prompts and sampling. Inference uses run_videomme_json_retry.py so qwen35
# runs retry until the visible completion ends with a valid "answer": "<letter>" snippet.
#
# Prompt (with --prompt-style qwen35): video first (as video_url / data URL), then text:
#   question + Choices:\n(A) ... + think step-by-step + JSON answer hint:
#   "Please show your choice in the answer field with only the choice letter, e.g., \"answer\": \"C\"."
#
# Generation (matches Qwen3.5 chat.completions example):
#   max_tokens=81920, temperature=1.0, top_p=0.95, presence_penalty=1.5,
#   extra_body.top_k=20,
#   extra_body.mm_processor_kwargs={ fps, do_sample_frames: true }
#   (vLLM documents fps/do_sample_frames via extra_body; other servers may ignore extras.)
#
# Override before running:
#   export API_BASE=http://localhost:50038
#   export SERVED_MODEL_NAME=qwen3_5-35b-a3b
#   export OPENAI_API_KEY=EMPTY
#   export DATA_DIR=/path/to/Video-MME   # default below: /data/winglam/dataset/Video-MME
#   export OUTPUT_FILE=.../predictions.jsonl
#
# Hugging Face zips (avoid unpacking all videos/ onto disk):
#   If VIDEOS_ZIPS / SUBTITLES_ZIPS are unset, the script auto-adds:
#     ${DATA_DIR}/videos_chunked_*.zip  and  ${DATA_DIR}/subtitle.zip (if present)
#   Override explicitly, e.g.:
#   export VIDEOS_ZIPS="/path/a.zip /path/b.zip"
#   export SUBTITLES_ZIPS="/path/subs.zip"
#   export NO_AUTO_VIDEOMME_ZIPS=1   # skip auto-detection
#
# Evaluation (runs after inference): judge via OpenAI-compatible chat/completions.
#   export MIT_SPIDER_URL=http://localhost:50038/v1/chat/completions
#   export MIT_SPIDER_TOKEN=EMPTY
#   export EVAL_OUTPUT_FILE=.../evaluation.csv
#   export RUN_EVAL=0   # skip eval if set to 0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

API_BASE="${API_BASE:-http://localhost:50038}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3_6-a3b}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
DATA_DIR="${DATA_DIR:-/data/winglam/dataset/Video-MME}"
OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/results/36_a3b.jsonl}"

EXTRA_INFER_ARGS=()
if [[ -n "${VIDEOS_ZIPS:-}" ]]; then
  # shellcheck disable=SC2206
  for z in ${VIDEOS_ZIPS}; do
    [[ -n "$z" ]] && EXTRA_INFER_ARGS+=(--videos-zip "$z")
  done
elif [[ -z "${NO_AUTO_VIDEOMME_ZIPS:-}" ]]; then
  shopt -s nullglob
  for z in "${DATA_DIR}"/videos_chunked_*.zip; do
    EXTRA_INFER_ARGS+=(--videos-zip "$z")
  done
  shopt -u nullglob
fi
if [[ -n "${SUBTITLES_ZIPS:-}" ]]; then
  # shellcheck disable=SC2206
  for z in ${SUBTITLES_ZIPS}; do
    [[ -n "$z" ]] && EXTRA_INFER_ARGS+=(--subtitles-zip "$z")
  done
elif [[ -z "${NO_AUTO_VIDEOMME_ZIPS:-}" && -f "${DATA_DIR}/subtitle.zip" ]]; then
  EXTRA_INFER_ARGS+=(--subtitles-zip "${DATA_DIR}/subtitle.zip")
fi

python run_videomme.py infer \
  --api-base "${API_BASE}" \
  --api-key "${OPENAI_API_KEY}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --data-dir "${DATA_DIR}" \
  --duration short \
  --output-file "${OUTPUT_FILE}" \
  --prompt-style qwen35 \
  --max-new-tokens 81920 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --fps 2 \
  --min-pixels 3584 \
  --max-pixels 401408 \
  --min-frames 4 \
  --max-frames 512 \
  --total-pixels 19267584 \
  --num-workers 2 \
  --request-timeout 3600 \
  "${EXTRA_INFER_ARGS[@]}"

echo "Wrote predictions to ${OUTPUT_FILE}"

# If the server errors on unknown extra_body keys, add: --no-extra-mm-processor-kwargs
# To match vLLM default do_sample_frames=false behavior: --no-video-do-sample-frames

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_OUTPUT_FILE="${EVAL_OUTPUT_FILE:-${SCRIPT_DIR}/results/debug/videomme_short_wo_subtitle_qwen36_evaluation_a3b_retry.csv}"
EVAL_NPROC="${EVAL_NPROC:-16}"

if [[ "${RUN_EVAL}" != "0" ]]; then
  export MIT_SPIDER_URL="${MIT_SPIDER_URL:-http://localhost:50038/v1/chat/completions}"
  export MIT_SPIDER_TOKEN="${MIT_SPIDER_TOKEN:-EMPTY}"
  # Judge HTTP read timeout (seconds). Default 600; raise if you see read timeout=60 under load.
  export MIT_SPIDER_TIMEOUT="${MIT_SPIDER_TIMEOUT:-600}"

  python run_videomme.py eval \
    --data-dir "${DATA_DIR}" \
    --input-file "${OUTPUT_FILE}" \
    --output-file "${EVAL_OUTPUT_FILE}" \
    --eval-model "${SERVED_MODEL_NAME}" \
    --api-type mit \
    --nproc "${EVAL_NPROC}"

  echo "Wrote evaluation to ${EVAL_OUTPUT_FILE} (and *_acc.json / .tsv alongside)"
fi
# nohup bash infer_think.sh > 36_a3b_video.log 2>&1 < /dev/null & 656863