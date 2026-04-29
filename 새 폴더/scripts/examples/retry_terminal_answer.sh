set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default: venv next to MMLU-Pro (e.g. winglam/venv). Override with VENV_PYTHON=/path/to/python
VENV_PY="${VENV_PYTHON:-$(cd "$REPO_ROOT/.." && pwd)/venv/bin/python}"
if [[ -x "$VENV_PY" ]]; then
  PYTHON=("$VENV_PY")
else
  PYTHON=(python3)
fi

# Uses evaluate_amend_terminal_answer.py: retries immediately until -r if the model
# does not end with Answer: (A-J) or The answer is (X). Adjust --output_dir / -m / --url as needed.

"${PYTHON[@]}" "$REPO_ROOT/evaluate_amend_terminal_answer.py" \
  --url 'http://localhost:50037/v1/' \
  -m 'qwen3_5-27b' \
  --output_dir "$REPO_ROOT/eval_results/27b_terminal_answer" \
  --assigned_subjects all \
  --num_workers 10 \
  -r 3

  # qwen3_5-27b qwen3_5-35b-a3b 2232996
  # nohup bash retry_terminal_answer.sh > 27b_terminal_answer.log 2>&1 < /dev/null &
