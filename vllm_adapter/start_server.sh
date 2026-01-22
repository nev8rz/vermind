#!/bin/bash
# Start vLLM server with VerMind model registration
# Reference: https://github.com/vllm-project/vllm/blob/main/tests/models/registry.py

cd /root/vermind
source .venv/bin/activate

# Set PYTHONPATH to ensure vermind_models can be imported in subprocesses
export PYTHONPATH="/root/vermind:$PYTHONPATH"

# Start vLLM server using Python script that ensures registration
exec python vllm_adapter/start_server.py
