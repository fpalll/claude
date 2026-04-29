set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default: venv next to MMLU-Pro (e.g. winglam/venv). Override with VENV_PYTHON=/path/to/python
VENV_PY="${VENV_PYTHON:-$(cd "$REPO_ROOT/.." && pwd)/venv/bin/python}"
if [[ -x "$VENV_PY" ]]; then
  PYTHON=("$VENV_PY")
else
  PYTHON=(python3)
fi

"${PYTHON[@]}" "$REPO_ROOT/evaluate_from_apiX.py" \
  --url 'http://localhost:50037/v1/' \
  -m 'qwen3_5-27b' \
  --output_dir "$REPO_ROOT/eval_results/27b" \
  --assigned_subjects all \
  --num_workers 10

  # qwen3_5-27b qwen3_5-35b-a3b