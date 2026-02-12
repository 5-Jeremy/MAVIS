#!/bin/bash

# Usage: ./start_vllm_servers.sh <NUM_GPUS>
# NUM_GPUS specifies the tensor parallelism degree (number of GPUs to use)

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <NUM_GPUS>"
	exit 1
fi

NUM_GPUS=$1
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

echo "Starting vllm server with $NUM_GPUS GPUs for $MODEL_NAME..."

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port 8000 \
    --tensor-parallel-size $NUM_GPUS \
    --max-num-batched-tokens 8192 &