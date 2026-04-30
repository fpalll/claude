PORT=50038
MODEL_DIR=/data/models/qwen_36_27b

sudo docker run --gpus '"device=0,1"' \
    -p $PORT:8000 \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm --shm-size=32g \
    -v /data:/data \
    --name qwen36_27b \
    vllm/vllm-openai:latest \
    $MODEL_DIR \
        --served-model-name qwen36_27b \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.9 \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 32768 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder


#         curl http://localhost:50038/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "qwen36_27b",
#     "messages": [{"role": "user", "content": "Hello"}]
#   }'
