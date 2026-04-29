#!/usr/bin/env bash
# Duplicate of retry.sh for debugging: same partial MMLU-Pro input on two OpenAI-compatible servers.
#
# Default comparison:
#   - qwen3_6_27b @ http://localhost:50040/v1/
#   - 35B-A3B (served name must match server) @ http://localhost:50041/v1/
#
# Override model id for 50041 if needed: NAME_35B="$(basename /data/models/qwen3.6-35b-a3b)" bash ...
# Or: curl -s http://localhost:50041/v1/models
#
# Env overrides: SAMPLE_SIZE, SUBJECT, SAMPLE_SEED, OUT_BASE, URL_27B, NAME_27B, URL_35B, NAME_35B, VENV_PYTHON
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="${VENV_PYTHON:-$(cd "$REPO_ROOT/.." && pwd)/venv/bin/python}"
if [[ -x "$VENV_PY" ]]; then
  PYTHON=("$VENV_PY")
else
  PYTHON=(python3)
fi

# Same partial MMLU-Pro slice for both runs (matches retry.sh spirit: small sample + one subject).
SAMPLE_SIZE="${SAMPLE_SIZE:-8}"
SUBJECT="${SUBJECT:-math}"
# Empty = first N questions per subject; set SAMPLE_SEED=42 for reproducible random N questions.
SAMPLE_SEED="${SAMPLE_SEED:-}"

URL_27B="${URL_27B:-http://localhost:50040/v1/}"
NAME_27B="${NAME_27B:-qwen3_6_27b}"

URL_35B="${URL_35B:-http://localhost:50041/v1/}"
# Served model id on :50041 — align with your vLLM/OpenAI server (path alone is not sent to the API).
NAME_35B="${NAME_35B:-/data/models/qwen3.6-35b-a3b}"

OUT_BASE="${OUT_BASE:-$REPO_ROOT/eval_results/mmlupro_compare}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
DIR_27="${OUT_BASE}/27b_${TS}"
DIR_35="${OUT_BASE}/35ba3b_${TS}"
LOG_27="${OUT_BASE}/debug_27b_${TS}.log"
LOG_35="${OUT_BASE}/debug_35ba3b_${TS}.log"

mkdir -p "$DIR_27" "$DIR_35"

run_eval() {
  local out_dir=$1 url=$2 model=$3 log=$4
  local -a cmd=(
    "${PYTHON[@]}" "$REPO_ROOT/evaluate_amend.py"
    --output_dir "$out_dir"
    --sample-size "$SAMPLE_SIZE"
    -a "$SUBJECT"
    --url "$url"
    -m "$model"
    --debug-print-output
    --num_workers 1
  )
  if [[ -n "$SAMPLE_SEED" ]]; then
    cmd+=(--sample-seed "$SAMPLE_SEED")
  fi
  "${cmd[@]}" 2>&1 | tee "$log"
}

echo "=== Run 1: ${NAME_27B} @ ${URL_27B}"
echo "    out: ${DIR_27}"
echo "    log: ${LOG_27}"
run_eval "$DIR_27" "$URL_27B" "$NAME_27B" "$LOG_27"

echo ""
echo "=== Run 2: ${NAME_35B} @ ${URL_35B}"
echo "    out: ${DIR_35}"
echo "    log: ${LOG_35}"
run_eval "$DIR_35" "$URL_35B" "$NAME_35B" "$LOG_35"

echo ""
echo "Done. Compare generation dumps ([debug] blocks) in:"
echo "  ${LOG_27}"
echo "  ${LOG_35}"
