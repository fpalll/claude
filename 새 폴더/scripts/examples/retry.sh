set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default: venv next to MMLU-Pro (e.g. winglam/venv). Override with VENV_PYTHON=/path/to/python
VENV_PY="${VENV_PYTHON:-$(cd "$REPO_ROOT/.." && pwd)/venv/bin/python}"
if [[ -x "$VENV_PY" ]]; then
  PYTHON=("$VENV_PY")
else
  PYTHON=(python3)
fi

"${PYTHON[@]}" "$REPO_ROOT/evaluate_amend.py" -o eval_debug/ --sample-size 8 -a math \
  --url 'http://localhost:50040/v1/' \
  -m 'qwen3_6_27b' \
  --output_dir "$REPO_ROOT/eval_results" \
  --max_tokens 512 \
  --no-thinking \
  --debug-print-output
  # --assigned_subjects all \
  # --num_workers 1 \
  # --max_tokens 512 

# python evaluate_amend.py -o eval_debug/ --sample-size 8 -a math
  # qwen3_5-27b qwen3_5-35b-a3b 2232996
  # nohup bash retry.sh > 27_0422.log 2>&1 < /dev/null & 598402
  # curl http://localhost:50037/v1/models