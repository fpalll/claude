#!/bin/bash

# ==============================
# CONFIG
vim run_bench.sh
chmod +x run_bench.sh
./run_bench.sh
# ==============================

MODEL="Qwen/Qwen3-35B-A3B"
API_URL="http://localhost:8000/v1"
OUTPUT_DIR="results_$(date +%Y%m%d_%H%M%S)"

# concurrency sweep
RATES=(1 2 4 8 16 32 64)

# scenarios (input_len output_len num_prompts)
SCENARIOS=(
  "short 2000 512 1000"
  "medium 16000 1000 500"
  "long 64000 2000 200"
  "xl 256000 1000 100"
)

# cache modes
# format: name dataset num_prefixes
CACHE_MODES=(
  "nocache random 0"
  "fullcache prefix_repetition 1"
  "partialcache prefix_repetition 4"
)

mkdir -p $OUTPUT_DIR

echo "Results will be saved to $OUTPUT_DIR"

# ==============================
# RUN
# ==============================

for scenario in "${SCENARIOS[@]}"; do
  read NAME INPUT_LEN OUTPUT_LEN NUM_PROMPTS <<< $scenario

  for cache in "${CACHE_MODES[@]}"; do
    read CACHE_NAME DATASET NUM_PREFIXES <<< $cache

    for rate in "${RATES[@]}"; do

      RUN_NAME="${NAME}_${CACHE_NAME}_rate${rate}"
      LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}.log"

      echo "========================================"
      echo "Running: $RUN_NAME"
      echo "========================================"

      CMD="python -m vllm.benchmarks.benchmark_serving \
        --backend openai \
        --model $MODEL \
        --base-url $API_URL \
        --dataset $DATASET \
        --input-len $INPUT_LEN \
        --output-len $OUTPUT_LEN \
        --num-prompts $NUM_PROMPTS \
        --request-rate $rate"

      # add prefix config if needed
      if [ "$DATASET" = "prefix_repetition" ]; then
        CMD="$CMD --num-prefixes $NUM_PREFIXES"
      fi

      echo "CMD: $CMD"

      # run and save log
      eval $CMD > $LOG_FILE 2>&1

      echo "Saved to $LOG_FILE"
      echo ""

      # small sleep to stabilize
      sleep 5

    done
  done
done

echo "========================================"
echo "ALL DONE"
echo "========================================"
