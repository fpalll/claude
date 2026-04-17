#!/bin/bash

PORT=50037
#model_dir=/embeddings/clip-vit-large-patch14-336
model_dir=/data/models/qwen3.5-27b

sudo docker run --gpus '"device=0,3"' -p $PORT:80 -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm \
           -v /data:/data --name qwen3_5_27b \
 	    lmsysorg/sglang:v0.5.10 \
	    python3 -m sglang.launch_server --model-path /data/models/qwen3.5-27b \
        --host 0.0.0.0 --port 80 --tensor-parallel-size 2 \
        --served-model-name qwen3_5-27b --tool-call-parser qwen3_coder --reasoning-parser qwen3 --context-length 262144

        
