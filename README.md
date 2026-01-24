# VerMind

<div align="center">

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-View%20Demo-success?style=for-the-badge&logo=github)](https://nev8rz.github.io/vermind/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python Logo" width="80" height="80"/>
<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace Logo" width="80" height="80"/>

</div>

**VerMind** is a high-performance, lightweight, and modern language model built from the ground up in PyTorch. It features a state-of-the-art architecture, including **Grouped Query Attention (GQA)** and **SwiGLU**, designed for efficient training, fine-tuning, and high-throughput inference.

This project serves as a comprehensive, end-to-end toolkit for building, training, and deploying custom language models. It is highly modular, extensively documented, and easy to customize.

## Why VerMind?

- **Performance & Efficiency**: Implements GQA and Flash Attention to reduce memory footprint and accelerate both training and inference, making it suitable for resource-constrained environments.
- **Modern Architecture**: Incorporates the latest advancements in LLM architecture, such as SwiGLU activation and Rotary Position Embedding (RoPE) with YaRN scaling for long-context understanding.
- **End-to-End Solution**: Provides a complete workflow from tokenizer training and pre-training to supervised fine-tuning (SFT), LoRA, and high-performance deployment with a vLLM adapter.
- **Extensibility & Customization**: The modular design and clear code structure make it easy to experiment with new ideas, swap components, and adapt the model to specific needs.
- **Educational Value**: Serves as an excellent learning resource for understanding the inner workings of modern language models, with detailed code and documentation.

## Key Features

| Feature | Description |
|---|---|
| **Grouped Query Attention (GQA)** | Reduces the memory bandwidth required for inference by sharing key-value heads, leading to significant speedups. |
| **SwiGLU Activation** | A modern activation function that often leads to better performance compared to traditional ReLU or GeLU. |
| **Rotary Position Embedding (RoPE)** | A relative position encoding scheme that has become standard in high-performance LLMs. Includes YaRN scaling for extending context length. |
| **vLLM Adapter** | Enables blazing-fast inference speeds and an OpenAI-compatible API server out-of-the-box. |
| **LoRA Fine-Tuning** | Supports parameter-efficient fine-tuning (PEFT) with Low-Rank Adaptation, allowing for rapid and memory-efficient customization. |
| **Distributed Training** | Built-in support for Distributed Data Parallel (DDP) to scale training across multiple GPUs. |
| **Comprehensive Training Suite** | Includes scripts for pre-training, supervised fine-tuning (SFT), and LoRA, with features like mixed-precision, gradient accumulation, and experiment tracking. |

## Model Architecture

VerMind's architecture is a decoder-only transformer that is optimized for performance and scalability. The core components include:

1.  **RMSNorm**: Used for layer normalization, providing better stability and performance than standard LayerNorm.
2.  **Rotary Position Embedding (RoPE)**: Applied to queries and keys to inject positional information.
3.  **Grouped Query Attention (GQA)**: The attention block where multiple query heads attend to a single key-value head, striking a balance between Multi-Head and Multi-Query Attention.
4.  **SwiGLU Feed-Forward Network**: The FFN block uses the SwiGLU activation, which consists of three linear projections and a Swish activation function.

This design is heavily influenced by successful models like Llama and Mistral.

## Project Structure

```
vermind/
├── vermind_models/          # Core model implementation (GQA, FFN, RoPE)
├── train/                   # Training scripts (pre-train, SFT, LoRA)
├── data_loader/             # Data loading and processing modules
├── scripts/                 # Utility scripts (evaluation, merging LoRA)
├── vllm_adapter/            # Adapter for high-performance vLLM inference
├── docs/                    # GitHub Pages website and documentation
└── pyproject.toml           # Project configuration and dependencies
```

## Quick Start

### 1. Installation

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

### 2. Training (Example: LoRA)

LoRA is the most efficient way to adapt VerMind to your data.

```bash
python train/lora.py \
    --data_path /path/to/your_sft_data.jsonl \
    --save_dir ./output/lora \
    --tokenizer_path /path/to/base_model_tokenizer \
    --from_weight /path/to/base_model_checkpoint \
    --lora_rank 16
```

### 3. Deployment with vLLM

Start a high-performance API server compatible with OpenAI's client.

```bash
python vllm_adapter/start_server.py /path/to/your_finetuned_checkpoint

# The server is now running at http://localhost:8000
```

### 4. Making API Requests

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="/path/to/your_finetuned_checkpoint",
    messages=[
        {"role": "user", "content": "Explain the importance of Grouped Query Attention."}
    ],
)
print(response.choices[0].message.content)
```

For more detailed instructions on pre-training, SFT, and evaluation, please refer to the scripts in the `train/` and `scripts/` directories.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

Please make sure your code adheres to the project's coding style (Ruff for linting and formatting).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use VerMind in your research or work, please consider citing it:

```bibtex
@software{vermind2026,
  title={VerMind: A High-Performance, Lightweight Language Model with GQA},
  author={Yijin Zhou},
  year={2026},
  url={https://github.com/nev8rz/vermind}
}
```
