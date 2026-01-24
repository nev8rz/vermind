#
<div align="center">
  <a href="https://github.com/nev8rz/vermind">
    <img src="https://raw.githubusercontent.com/nev8rz/vermind/main/docs/assets/vermind_logo.png" alt="VerMind Logo" width="120">
  </a>
  <h1 align="center">VerMind</h1>
  <p align="center">
    A high-performance, lightweight, and modern language model built from the ground up in PyTorch.
    <br />
    <a href="https://nev8rz.github.io/vermind/"><strong>View Demo »</strong></a>
    ·
    <a href="https://github.com/nev8rz/vermind/issues">Report Bug</a>
    ·
    <a href="https://github.com/nev8rz/vermind/issues">Request Feature</a>
  </p>
</div>

<div align="center">

**English** · [简体中文](README_zh.md)

</div>

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch 2.8.0+](https://img.shields.io/badge/PyTorch-2.8.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/nev8rz/vermind?style=for-the-badge&logo=github)](https://github.com/nev8rz/vermind/stargazers)

</div>

---

**VerMind** is a comprehensive, end-to-end toolkit for building, training, and deploying custom language models. It features a state-of-the-art architecture, including **Grouped Query Attention (GQA)** and **SwiGLU**, designed for efficient training, fine-tuning, and high-throughput inference. This project is highly modular, extensively documented, and easy to customize, making it an ideal starting point for both research and production.

## ✨ Why VerMind?

-   🚀 **Performance & Efficiency**: Implements GQA and Flash Attention to reduce memory footprint and accelerate both training and inference.
-   🧠 **Modern Architecture**: Incorporates the latest advancements in LLM architecture, such as SwiGLU activation and Rotary Position Embedding (RoPE) with YaRN scaling.
-   🔧 **End-to-End Solution**: Provides a complete workflow from tokenizer training and pre-training to supervised fine-tuning (SFT), LoRA, and deployment with a vLLM adapter.
-   🧩 **Extensibility & Customization**: The modular design makes it easy to experiment with new ideas, swap components, and adapt the model to specific needs.
-   🎓 **Educational Value**: Serves as an excellent learning resource for understanding the inner workings of modern language models, with detailed code and documentation.

## 🛠️ Key Features

| Feature                               | Description                                                                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| ⚡ **Grouped Query Attention (GQA)**  | Reduces the memory bandwidth required for inference by sharing key-value heads, leading to significant speedups.                       |
| 🔥 **SwiGLU Activation**              | A modern activation function that often leads to better performance compared to traditional ReLU or GeLU.                                |
| 📐 **Rotary Position Embedding (RoPE)** | A relative position encoding scheme that has become standard in high-performance LLMs. Includes YaRN scaling for extending context length. |
| 🚀 **vLLM Adapter**                   | Enables blazing-fast inference speeds and an OpenAI-compatible API server out-of-the-box.                                                |
| 🎨 **LoRA Fine-Tuning**               | Supports parameter-efficient fine-tuning (PEFT) with Low-Rank Adaptation for rapid and memory-efficient customization.                 |
| 🌐 **Distributed Training**           | Built-in support for Distributed Data Parallel (DDP) to scale training across multiple GPUs.                                             |

## 🏗️ Architecture Overview

VerMind's architecture is a decoder-only transformer optimized for performance and scalability. The core components are designed to be both efficient and easy to understand.

<div align="center">
  <img src="https://raw.githubusercontent.com/nev8rz/vermind/main/docs/assets/architecture.png" alt="VerMind Architecture" width="800">
</div>

-   **RMSNorm**: Used for layer normalization, providing better stability.
-   **Rotary Position Embedding (RoPE)**: Applied to queries and keys to inject positional information.
-   **Grouped Query Attention (GQA)**: The attention block where multiple query heads attend to a single key-value head.
-   **SwiGLU Feed-Forward Network**: The FFN block uses the SwiGLU activation for better performance.

## 🚀 Getting Started

Get your local copy up and running in a few simple steps.

### Prerequisites

-   Python 3.12+
-   PyTorch 2.8.0+
-   `uv` package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/nev8rz/vermind.git
cd vermind

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## 🏃‍♀️ Usage Examples

VerMind provides a complete training pipeline with convenient shell scripts located in `scripts/`. The training workflow follows: **Tokenizer → Pre-training → SFT → LoRA → Deployment**.

### 1. Train Tokenizer

First, train a custom tokenizer on your corpus:

```bash
python train/train_tokenizer.py \
    --data_path /path/to/training_corpus.txt \
    --tokenizer_dir ./vermind_tokenizer \
    --vocab_size 6400
```

### 2. Pre-training

Pre-train the model from scratch on a large corpus. Use the provided script or run directly:

```bash
# Option 1: Use the launch script (runs in tmux)
bash scripts/pretrain.sh

# Option 2: Run directly with custom parameters
python train/pretrain.py \
    --data_path /path/to/pretrain_data.jsonl \
    --save_dir ./output/pretrain \
    --tokenizer_path ./vermind_tokenizer \
    --epochs 5 \
    --batch_size 128 \
    --learning_rate 1e-3
```

### 3. Supervised Fine-Tuning (SFT)

Fine-tune the pre-trained model on instruction-following data:

```bash
# Option 1: Use the launch script (runs in tmux)
bash scripts/sft.sh

# Option 2: Run directly with custom parameters
python train/sft.py \
    --data_path /path/to/sft_data.jsonl \
    --save_dir ./output/sft \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/pretrain/pretrain_768 \
    --epochs 3 \
    --learning_rate 5e-6
```

### 4. LoRA Fine-Tuning

For parameter-efficient fine-tuning, use LoRA to adapt the model with minimal resources:

```bash
# Option 1: Use the launch script (runs in tmux)
bash scripts/lora.sh

# Option 2: Run directly with custom parameters
python train/lora.py \
    --data_path /path/to/lora_data.jsonl \
    --save_dir ./output/lora \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/sft/full_sft_768 \
    --epochs 5 \
    --learning_rate 1e-4 \
    --lora_rank 16
```

### 5. Merge LoRA Weights

After LoRA training, merge the adapter weights into the base model:

```bash
python scripts/merge_lora.py \
    --model_path ./output/sft/full_sft_768 \
    --lora_path ./output/lora/lora_768
```

### 6. Model Evaluation

Evaluate your model interactively:

```bash
python scripts/eval_llm.py \
    --load_from ./output/lora/lora_768/checkpoint_merged \
    --use_chat_template 1
```

### 7. Deploy with vLLM

Start a high-performance API server compatible with OpenAI's client:

```bash
# Start the server
python vllm_adapter/start_server.py ./output/lora/lora_768/checkpoint_merged

# The server is now running at http://localhost:8000
```

### 8. Making API Requests

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="./output/lora/lora_768/checkpoint_merged",
    messages=[
        {"role": "user", "content": "Explain the importance of Grouped Query Attention."}
    ],
)
print(response.choices[0].message.content)
```

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✒️ Citation

If you use VerMind in your research or work, please consider citing it:

```bibtex
@software{vermind2026,
  title={VerMind: A High-Performance, Lightweight Language Model with GQA},
  author={Yijin Zhou},
  year={2026},
  url={https://github.com/nev8rz/vermind}
}
```

---

<p align="center">Made with ❤️ by Yijin Zhou</p>
