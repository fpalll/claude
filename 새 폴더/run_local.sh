export CUDA_VISIBLE_DEVICES=6,7

export HF_HOME=/data/winglam/hf_cache
export TRANSFORMERS_CACHE=/data/winglam/hf_cache
export HF_HUB_CACHE=/data/winglam/hf_cache

python evaluate_from_local.py \
  --model /data/models/qwen3.5-27b 