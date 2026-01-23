# VerMind

<div align="center">
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python Logo" width="100" height="100"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace Logo" width="100" height="100"/>
</div>

VerMind is a high-performance language model with GQA (Grouped Query Attention) support, designed for efficient training and inference.

## Features

- ✅ **GQA (Grouped Query Attention)** - Efficient attention mechanism with grouped query and key-value heads
- ✅ **SwiGLU Activation** - Swish-Gated Linear Unit activation function
- ✅ **RoPE with YaRN Scaling** - Rotary Position Embedding with extended context support
- ✅ **HuggingFace Integration** - Compatible with Transformers library
- ✅ **vLLM Adapter** - High-performance inference with OpenAI-compatible API
- ✅ **Distributed Training** - Support for multi-GPU training with DDP
- ✅ **Checkpoint Management** - Automatic checkpoint saving and resuming

## Project Structure

```
vermind/
├── vermind_models/          # Core model implementation
│   ├── config/              # Model configuration
│   ├── GQA.py               # Grouped Query Attention
│   ├── FFN.py               # Feed Forward Network
│   └── base_module.py       # Base modules (RMSNorm, RoPE, etc.)
├── train/                   # Training scripts
│   ├── pretrain.py          # Pre-training script
│   ├── sft.py               # Supervised Fine-Tuning script
│   ├── train_tokenizer.py   # Tokenizer training
│   └── utils.py             # Training utilities
├── data_loader/             # Data loading modules
│   ├── pretrain_dataset.py  # Pre-training dataset
│   └── sft_dataset.py       # SFT dataset
├── scripts/                 # Utility scripts
│   ├── pretrain.sh          # Pre-training launch script
│   ├── sft.sh               # SFT launch script
│   └── eval_llm.py          # Model evaluation script
├── vllm_adapter/            # vLLM inference adapter
│   ├── start_server.py      # API server startup
│   └── README.md            # vLLM adapter documentation
└── docs/                    # Documentation notebooks
```

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (for training/inference)
- uv (package manager)

### Setup

```bash
# Clone the repository
git clone https://github.com/nev8rz/vermind.git
cd vermind

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

## Quick Start

### 1. Train Tokenizer

```bash
python train/train_tokenizer.py \
    --data_path /path/to/training_data.txt \
    --tokenizer_dir /path/to/tokenizer_output \
    --vocab_size 6400
```

### 2. Pre-training

```bash
# Using launch script
bash scripts/pretrain.sh

# Or directly
python train/pretrain.py \
    --data_path /path/to/pretrain_data.jsonl \
    --save_dir ./output/pretrain \
    --tokenizer_path /path/to/tokenizer \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --num_attention_heads 8 \
    --num_key_value_heads 2 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 1
```

### 3. Supervised Fine-Tuning (SFT)

```bash
# Using launch script
bash scripts/sft.sh

# Or directly
python train/sft.py \
    --data_path /path/to/sft_data.jsonl \
    --save_dir ./output/sft \
    --tokenizer_path /path/to/tokenizer \
    --from_weight /path/to/pretrain/checkpoint \
    --epochs 3 \
    --batch_size 128 \
    --learning_rate 5e-6
```

### 4. Model Evaluation

```bash
# Interactive evaluation
python scripts/eval_llm.py \
    --load_from /path/to/checkpoint \
    --max_new_tokens 2048 \
    --temperature 0.85 \
    --use_chat_template 1

# Auto-test mode (select [0] when prompted)
python scripts/eval_llm.py --load_from /path/to/checkpoint
```

### 5. Deploy with vLLM (High-Performance Inference)

```bash
# Start API server
cd /root/vermind
source .venv/bin/activate
bash vllm_adapter/start_server.sh

# Or with custom model path
python vllm_adapter/start_server.py /path/to/checkpoint
```

The server will start on `http://localhost:8000` with OpenAI-compatible API.

See [vLLM Adapter Documentation](vllm_adapter/README.md) for detailed API usage.

## Model Configuration

### Basic Model Sizes

- **Small (26M)**: `hidden_size=512, num_hidden_layers=8`
- **Base (104M)**: `hidden_size=768, num_hidden_layers=16`

### Key Parameters

- `hidden_size`: Model hidden dimension
- `num_hidden_layers`: Number of transformer layers
- `num_attention_heads`: Number of query attention heads
- `num_key_value_heads`: Number of key-value heads (for GQA)
- `max_position_embeddings`: Maximum sequence length (default: 32768)

## Training

### Pre-training

Pre-training script supports:
- Distributed Data Parallel (DDP) training
- Mixed precision training (bfloat16/float16)
- Gradient accumulation
- Learning rate scheduling (warmup + cosine annealing)
- Automatic checkpoint saving and resuming
- SwanLab experiment tracking

### Supervised Fine-Tuning

SFT script features:
- Chat template support
- Multi-turn conversation training
- Same training features as pre-training

### Checkpoint Management

- Checkpoints are saved in HuggingFace format
- Automatic checkpoint numbering (`checkpoint_1000`, `checkpoint_2000`, etc.)
- Supports automatic resuming from latest checkpoint
- Automatic cleanup of old checkpoints (keeps latest 3 by default)

## Inference

### Local Inference

Use `scripts/eval_llm.py` for local inference:

```bash
python scripts/eval_llm.py \
    --load_from /path/to/checkpoint \
    --max_new_tokens 2048 \
    --temperature 0.85 \
    --top_p 0.85 \
    --historys 4 \
    --use_chat_template 1
```

### vLLM API Server

Start the server and use OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="/path/to/model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=100,
)
```

See [vLLM Adapter README](vllm_adapter/README.md) for more details.

## Data Format

### Pre-training Data

JSONL format, one JSON object per line:
```json
{"text": "Your training text here..."}
```

### SFT Data

JSONL format with conversation structure:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Configuration Files

### Model Config (`config.json`)

```json
{
  "model_type": "vermind",
  "hidden_size": 768,
  "num_hidden_layers": 16,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "vocab_size": 6400,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0
}
```

## Dependencies

- `torch==2.8.0` - PyTorch
- `transformers>=4.57.6` - HuggingFace Transformers
- `vllm>=0.11.0` - vLLM for inference
- `datasets>=4.5.0` - Dataset handling
- `swanlab>=0.7.6` - Experiment tracking

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

## Citation

If you use VerMind in your research, please cite:

```bibtex
@software{vermind2026,
  title={VerMind: A High-Performance Language Model with GQA},
  author={[Your Name]},
  year={2026},
  url={https://github.com/nev8rz/vermind}
}
```
