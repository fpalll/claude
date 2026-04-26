#!/bin/bash
set -e

echo "=== Setting up evaluation environment ==="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate with:  source venv/bin/activate"
echo ""
echo "Check your SGLang model name:"
echo "  curl http://localhost:50038/v1/models | python3 -m json.tool"
echo ""
echo "Example runs:"
echo "  # Debug run first (50 samples, ~5 min)"
echo "  python run_eval.py --model <model-name> --sample 50"
echo ""
echo "  # Full evaluation (thinking ON by default — matches HF scores)"
echo "  python run_eval.py --model <model-name>"
echo ""
echo "  # Single benchmark"
echo "  python run_eval.py --model <model-name> --benchmark mmlu_pro"
echo ""
echo "  # Disable thinking mode to compare"
echo "  python run_eval.py --model <model-name> --no-thinking"
