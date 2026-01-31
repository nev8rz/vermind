# VerMind vLLM Adapter

vLLM adapter for VerMind models - enables high-performance inference with OpenAI-compatible API.

## Supported Models

- **VerMind**: Pure language model (GQA + SwiGLU)
- **VerMind-V**: Vision-language model (text-only mode in vLLM)

## Quick Start

### Start Server

```bash
cd /root/vermind
source .venv/bin/activate

# Method 1: Use start script with model path
bash vllm_adapter/start_server.sh /path/to/your/model

# Method 2: Use Python script directly
python vllm_adapter/start_server.py /path/to/your/model

# Method 3: Custom vLLM options
bash vllm_adapter/start_server.sh /path/to/model --port 8080 --max-model-len 4096
```

The server will start on `http://localhost:8000` by default.

### VerMind (Language Model)

```bash
# Start with pretrain checkpoint
bash vllm_adapter/start_server.sh /root/vermind/output/pretrain/pretrain_768/checkpoint_10000

# Start with DPO checkpoint
bash vllm_adapter/start_server.sh /root/vermind/output/dpo/dpo_768/checkpoint_1610
```

### VerMind-V (Vision-Language Model)

```bash
# Start VLM model (text-only mode in vLLM)
bash vllm_adapter/start_server.sh /root/vermind/output/vlm_pretrain/vlm_from_dpo_768/checkpoint_1000
```

**Note**: For full VLM inference with image support, use the standard inference script instead of vLLM.

## OpenAI-Compatible API Usage

### Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM doesn't require a real API key
)

# Chat completion
response = client.chat.completions.create(
    model="vermind",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
```

### cURL

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vermind",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vermind",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

## Directory Structure

```
vllm_adapter/
├── core/                      # Core VerMind model files
│   ├── configuration_vermind.py
│   ├── modeling_vermind.py
│   ├── vermind.py            # Main vLLM model implementation
│   └── register.py           # Model registration
├── vlm/                       # VerMind-V (VLM) files
│   ├── configuration_vermind_v.py
│   ├── modeling_vermind_v.py
│   └── vermind_v.py          # VLM model (text-only in vLLM)
├── plugin.py                  # vLLM plugin registration
├── start_server.py            # Server startup script
├── start_server.sh            # Shell wrapper
├── __init__.py
└── README.md
```

## Features

- ✅ Automatic plugin registration via vLLM plugin system
- ✅ Auto-configuration: automatically completes missing config files
- ✅ Supports both VerMind and VerMind-V models
- ✅ OpenAI-compatible API endpoints
- ✅ LoRA checkpoint detection and merge reminder

## Notes

- Model checkpoint must include `chat_template.jinja` for chat support
- Auto-configuration runs automatically on server startup
- Default port: 8000 (configurable via `--port`)
- VerMind-V runs in text-only mode in vLLM; for full VLM inference with images, use the standard inference script
