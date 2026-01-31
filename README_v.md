# VerMind-V

<div align="center">
  <a href="https://github.com/nev8rz/vermind">
    <img src="https://raw.githubusercontent.com/nev8rz/vermind/main/docs/assets/vermind_logo.png" alt="VerMind-V Logo" width="120">
  </a>
  <h1 align="center">VerMind-V</h1>
  <p align="center">
    Vision-Language Model Powered by VerMind Architecture
    <br />
    <strong>See · Understand · Generate</strong>
  </p>
</div>

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch 2.8.0+](https://img.shields.io/badge/PyTorch-2.8.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

**VerMind-V** is a vision-language model (VLM) built on top of the VerMind architecture. It extends the powerful language understanding capabilities of VerMind with vision encoding, enabling multimodal understanding and generation tasks.

## ✨ Key Features

- 🔮 **Vision Encoder Integration**: Seamlessly integrates vision encoders (CLIP/SigLIP) with VerMind's language model
- 🧠 **Unified Multimodal Architecture**: Single model for both vision and language understanding
- ⚡ **Efficient Training**: Supports pre-training and instruction tuning for multimodal tasks
- 🎯 **Multiple Vision Tasks**: Image captioning, visual question answering, visual reasoning
- 🚀 **vLLM Compatible**: Deploy with vLLM for high-throughput multimodal inference

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Vision Encoder │────▶│  Projector      │────▶│  VerMind LLM    │
│  (CLIP/SigLIP)  │     │  (MLP/Linear)   │     │  (Decoder-only) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                                 │
         └───────────────── Image + Text ──────────────────┘
```

The architecture consists of three main components:

1. **Vision Encoder**: Pre-trained vision transformer (CLIP-ViT or SigLIP) that encodes images into visual tokens
2. **Projector**: Lightweight adapter that aligns visual features with the language model's embedding space
3. **VerMind LLM**: The powerful decoder-only language model backbone with GQA and SwiGLU

## 📊 Model Variants

| Model | Vision Encoder | LLM Parameters | Context Length | Resolution |
|-------|---------------|----------------|----------------|------------|
| VerMind-V-Base | SigLIP-SO | ~104M | 2K / 32K | 384×384 |
| VerMind-V-Large | SigLIP-SO | ~350M | 2K / 32K | 384×384 |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/nev8rz/vermind.git
cd vermind
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Inference Example

```python
from vermind_models import VerMindVLM, VLMConfig
from transformers import AutoTokenizer
from PIL import Image

# Load model
model = VerMindVLM.from_pretrained("nev8rz/vermind-v-base")
tokenizer = AutoTokenizer.from_pretrained("nev8rz/vermind-v-base")

# Load image
image = Image.open("example.jpg")

# Generate caption
prompt = "<image>\nDescribe this image in detail."
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass with image
outputs = model.generate(
    **inputs,
    images=[image],
    max_new_tokens=256
)

print(tokenizer.decode(outputs[0]))
```

## 🏃 Training Pipeline

### 1. Vision-Language Pre-training

Pre-train the projector on large-scale image-text pairs:

```bash
bash examples/vlm_pretrain.sh
```

### 2. Visual Instruction Tuning

Fine-tune on visual instruction-following data:

```bash
bash examples/vlm_sft.sh
```

### 3. Deploy

```bash
python vllm_adapter/start_server.py ./output/vlm/vlm_base
```

## 📁 Project Structure

```
vermind/
├── vermind_models/
│   ├── models/
│   │   ├── vermind_vlm.py      # VerMindVLM implementation
│   │   └── vision_encoder.py   # Vision encoder wrapper
│   └── config/
│       └── vlm_config.py       # VLM configuration
├── data_loader/
│   └── vlm_dataset.py          # Multimodal dataset classes
└── train/
    ├── vlm_pretrain.py         # Pre-training script
    └── vlm_sft.py              # Visual SFT script
```

## 🤝 Integration with VerMind

VerMind-V shares the same core architecture as VerMind:

- Same GQA and SwiGLU components
- Compatible tokenizer
- Unified training framework
- Shared vLLM adapter

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## 🔗 Links

- [VerMind Main Repo](https://github.com/nev8rz/vermind)
- [VerMind Documentation](https://nev8rz.github.io/vermind/)
- [Demo](https://nev8rz.github.io/vermind/vermind-v.html)

---

<p align="center">Powered by <a href="https://github.com/nev8rz/vermind">VerMind</a> ❤️</p>
