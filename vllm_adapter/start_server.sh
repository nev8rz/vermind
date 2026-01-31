#!/bin/bash
# Start vLLM server with VerMind model registration
# Supports both VerMind (language) and VerMind-V (vision-language) models

cd /root/vermind
source .venv/bin/activate

# Set PYTHONPATH to ensure vermind_models can be imported in subprocesses
export PYTHONPATH="/root/vermind:${PYTHONPATH}"

# Check if model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path> [vllm_options]"
    echo ""
    echo "Examples:"
    echo "  $0 /root/vermind/output/pretrain/pretrain_768/checkpoint_10000"
    echo "  $0 /root/vermind/output/vlm_pretrain/vlm_from_dpo_768/checkpoint_1000 --port 8080"
    echo ""
    echo "Using default model path..."
    MODEL_PATH="/root/vermind/output/pretrain/pretrain_768/checkpoint_10000"
else
    MODEL_PATH="$1"
    shift  # Remove first argument, pass rest to vLLM
fi

# Start vLLM server
exec python vllm_adapter/start_server.py "$MODEL_PATH" "$@"
