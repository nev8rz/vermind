# VerMind vLLM Adapter

vLLM adapter for VerMind models - enables high-performance inference with OpenAI-compatible API.

## Quick Start

### Start Server

```bash
cd /root/vermind
source .venv/bin/activate

# Method 1: Use start script (recommended)
bash vllm_adapter/start_server.sh

# Method 2: Use Python script directly
python vllm_adapter/start_server.py /path/to/your/model

# Method 3: Specify custom model path
python vllm_adapter/start_server.py /root/vermind/output/pretrain/pretrain_768/checkpoint_10000
```

The server will start on `http://localhost:8000` by default.

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
    model="/path/to/vermind/model",
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
    "model": "/path/to/vermind/model",
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
    "model": "/path/to/vermind/model",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### Verify Server Status

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

## Features

- ✅ Automatic plugin registration via vLLM plugin system
- ✅ Auto-configuration: automatically completes missing config files
- ✅ Chat model support with chat template
- ✅ OpenAI-compatible API endpoints

## Notes

- Model checkpoint must include `chat_template.jinja` for chat support
- Auto-configuration runs automatically on server startup
- Default port: 8000 (configurable via `--port`)
