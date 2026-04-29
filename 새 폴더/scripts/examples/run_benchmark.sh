#!/usr/bin/env bash
# Run MMLU-Pro eval in batch mode (no fullscreen TUI). Typical detached run:
#   nohup bash /path/to/MMLU-Pro/scripts/examples/run_benchmark.sh > benchmark.log 2>&1 < /dev/null &
# Extra CLI args are forwarded to evaluate_from_apiX.py (e.g. --num_workers 4).
set -euo pipefail
export PYTHONUNBUFFERED=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_PY="${VENV_PYTHON:-$(cd "$REPO_ROOT/.." && pwd)/venv/bin/python}"
if [[ -x "$VENV_PY" ]]; then
  PYTHON=("$VENV_PY")
else
  PYTHON=(python3)
fi

exec "${PYTHON[@]}" "$REPO_ROOT/evaluate_from_apiX.py" \
  --batch \
  --no-stream \
  --url 'http://localhost:50038/v1/' \
  -m 'qwen3_5-35b-a3b' \
  --output_dir "$REPO_ROOT/eval_results" \
  --assigned_subjects all \
  --num_workers 10 \
  "$@"
