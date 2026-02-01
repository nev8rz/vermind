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
-   🔧 **End-to-End Solution**: Provides a complete workflow from tokenizer training and pre-training to supervised fine-tuning (SFT) with packed training support, DPO alignment, LoRA, and deployment with a vLLM adapter.
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
| 📦 **Packed SFT Training**            | Efficient sequence packing for SFT training with Varlen FlashAttention, reducing padding waste and improving GPU utilization by packing multiple sequences into batches. |
| 🎯 **Direct Preference Optimization (DPO)** | Align models with human preferences using preference pairs, improving output quality without requiring a separate reward model. |
| 🎮 **Proximal Policy Optimization (PPO)** | Advanced RLHF training with reward models for enhanced reasoning capabilities and response quality. |
| 🎯 **Group Relative Policy Optimization (GRPO)** | Efficient RL training without critic model, using group-relative advantages for policy optimization. |

## 🏗️ Architecture Overview

VerMind's architecture is a decoder-only transformer optimized for performance and scalability. The core components are designed to be both efficient and easy to understand.

<div align="center">
  <img src="https://raw.githubusercontent.com/nev8rz/vermind/main/docs/assets/architecture.png" alt="VerMind Architecture" width="800">
</div>

-   **RMSNorm**: Used for layer normalization, providing better stability.
-   **Rotary Position Embedding (RoPE)**: Applied to queries and keys to inject positional information.
-   **Grouped Query Attention (GQA)**: The attention block where multiple query heads attend to a single key-value head.
-   **SwiGLU Feed-Forward Network**: The FFN block uses the SwiGLU activation for better performance.

## 📊 Evaluation Results

VerMind has been evaluated on Chinese language understanding benchmarks (768 hidden size model):

| Benchmark | Version | SFT | DPO | PPO | GRPO |
|-----------|---------|-----|-----|-----|------|
| ACLUE | v1 | 25.67% ± 0.62% | 25.41% ± 0.62% | **25.82%** ± 0.62% | 25.76% ± 0.62% |
| CEval-Valid | v2 | 23.85% ± 1.17% | 23.55% ± 1.16% | **23.92%** ± 1.16% | 23.78% ± 1.16% |
| CMMLU | v1 | 24.79% ± 0.40% | **25.19%** ± 0.40% | 25.17% ± 0.40% | 24.95% ± 0.40% |
| TMMLUPlus | v2 | 25.15% ± 0.22% | **25.33%** ± 0.22% | 25.17% ± 0.22% | 25.21% ± 0.22% |

*Higher is better. Best results in bold.*

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

VerMind provides a complete training pipeline with convenient shell scripts located in `examples/`. The training workflow follows: **Tokenizer → Pre-training → SFT → DPO/PPO/GRPO (optional) → LoRA → Deployment**.

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
bash examples/pretrain.sh

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
bash examples/sft.sh

# Option 2: Run directly with custom parameters
python train/sft.py \
    --data_path /path/to/sft_data.jsonl \
    --save_dir ./output/sft \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/pretrain/pretrain_768 \
    --epochs 3 \
    --learning_rate 5e-6
```

#### Packed SFT Training

For more efficient training with better GPU utilization, use the packed SFT training mode which packs multiple sequences into a single batch using Varlen FlashAttention:

```bash
# Option 1: Use the launch script (runs in tmux)
bash examples/sft_packed.sh

# Option 2: Run directly with custom parameters
python train/sft_packed.py \
    --data_path /path/to/sft_data.jsonl \
    --parquet_path ./cache/sft_packed/sft.parquet \
    --save_dir ./output/sft_packed \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/pretrain/pretrain_768 \
    --epochs 3 \
    --learning_rate 5e-6 \
    --use_packed 1 \
    --max_seq_len 2048
```

**Preprocessing data for packed training:**

```bash
# First, preprocess your JSONL data into packed Parquet format
python scripts/pre_sftdatapacked.py \
    --jsonl_path /path/to/sft_data.jsonl \
    --output_path ./cache/sft_packed/sft.parquet \
    --tokenizer_path ./vermind_tokenizer \
    --max_seq_len 2048
```

Packed SFT training improves training efficiency by packing multiple sequences of varying lengths into fixed-size batches, reducing padding waste and improving GPU utilization. It uses Varlen FlashAttention for efficient attention computation without explicit attention masks.

### 4. LoRA Fine-Tuning

For parameter-efficient fine-tuning, use LoRA to adapt the model with minimal resources:

```bash
# Option 1: Use the launch script (runs in tmux)
bash examples/lora.sh

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

### 5. Direct Preference Optimization (DPO)

Fine-tune the model using DPO to align with human preferences by learning from preference pairs:

```bash
# Option 1: Use the launch script (runs in tmux, uses --dpo_aggregate mean)
bash examples/dpo.sh

# Option 2: Run directly with custom parameters
python train/dpo.py \
    --data_path /path/to/dpo_data.jsonl \
    --save_dir ./output/dpo \
    --tokenizer_path ./vermind_tokenizer \
    --ref_weight ./output/sft/full_sft_768 \
    --from_weight ./output/sft/full_sft_768 \
    --epochs 3 \
    --learning_rate 1e-6 \
    --beta 0.1 \
    --dpo_aggregate mean \
    --batch_size 16 \
    --max_seq_len 340
```

DPO training uses preference pairs (chosen vs rejected responses) to optimize the model's output quality without requiring a separate reward model. Use `--dpo_aggregate mean` (default for small models) or `sum` for sequence-level log-prob aggregation.

### 6. Proximal Policy Optimization (PPO)

For advanced RLHF training using PPO with a reward model to further improve model performance:

```bash
# Option 1: Use the launch script (runs in tmux)
bash examples/ppo.sh

# Option 2: Run directly with custom parameters
python train/ppo.py \
    --data_path /path/to/rlaif_data.jsonl \
    --save_dir ./output/ppo \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/sft/full_sft_768 \
    --ref_weight ./output/sft/full_sft_768 \
    --reward_model_path /path/to/reward_model \
    --epochs 3 \
    --learning_rate 1e-6 \
    --batch_size 8 \
    --max_seq_len 512 \
    --max_gen_len 1536 \
    --clip_epsilon 0.2 \
    --kl_coef 0.01
```

**PPO Key Parameters:**

- `--reward_model_path`: Path to the reward model for computing rewards
- `--clip_epsilon`: PPO clipping parameter (default: 0.2)
- `--kl_coef`: KL divergence penalty coefficient (default: 0.01)
- `--vf_coef`: Value function loss coefficient (default: 0.5)
- `--critic_lr_ratio`: Critic learning rate ratio to actor (default: 1.0)
- `--update_old_actor_freq`: Frequency to update the old actor (default: 10 steps)
- `--reasoning`: Set to 1 to enable reasoning mode with format rewards

PPO training uses a reward model to guide the policy optimization, making it suitable for complex alignment tasks. The training involves an actor-critic architecture with KL penalty to prevent the model from deviating too far from the reference policy.

### 7. Group Relative Policy Optimization (GRPO)

For efficient RL training without a critic model, using group-relative advantages:

```bash
# Option 1: Use the launch script (runs in tmux)
bash examples/grpo.sh

# Option 2: Run directly with custom parameters
python train/grpo.py \
    --data_path /path/to/rlaif_data.jsonl \
    --save_dir ./output/grpo \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/sft/full_sft_768 \
    --ref_weight ./output/sft/full_sft_768 \
    --reward_model_path /path/to/reward_model \
    --epochs 3 \
    --learning_rate 1e-6 \
    --batch_size 4 \
    --num_generations 4 \
    --max_seq_len 512 \
    --max_gen_len 1536 \
    --beta 0.04
```

**GRPO Key Parameters:**

- `--reward_model_path`: Path to the reward model for computing rewards
- `--num_generations`: Number of responses generated per prompt (default: 4)
- `--beta`: KL divergence penalty coefficient (default: 0.04)
- `--reasoning`: Set to 1 to enable reasoning mode with format rewards

GRPO eliminates the need for a critic model by computing relative advantages within groups of responses. This reduces memory usage and simplifies training while maintaining alignment quality.

### 8. Merge LoRA Weights

After LoRA training, merge the adapter weights into the base model:

```bash
python scripts/merge_lora.py \
    --model_path ./output/sft/full_sft_768 \
    --lora_path ./output/lora/lora_768
```

### 9. Model Evaluation

Evaluate your model interactively:

```bash
python scripts/eval_llm.py \
    --load_from ./output/lora/lora_768/checkpoint_merged \
    --use_chat_template 1
```

### 10. Vision-Language Model (VLM) Training

VerMind-V extends the base model with vision capabilities. Train your own VLM with the unified training script:

#### Stage 1: Vision-Language Pre-training

Pre-train the projector on large-scale image-text pairs (freeze LLM, train only vision projection):

```bash
bash examples/pretrain_vlm.sh

# Or run directly
python train/train_vlm.py \
    --stage pretrain \
    --from_weight ./output/sft/full_sft_768 \
    --data_path ./dataset/vlm_pretrain.parquet \
    --save_dir ./output/vlm_pretrain \
    --tokenizer_path ./vermind_tokenizer \
    --vision_encoder_path ./siglip-base-patch16-224 \
    --epochs 4 \
    --batch_size 64 \
    --learning_rate 5e-4 \
    --freeze_vision 1 \
    --freeze_llm 1
```

#### Stage 2: Visual Instruction Tuning

Fine-tune on visual instruction-following data (full model training):

```bash
bash examples/vlm_sft.sh

# Or run directly
python train/train_vlm.py \
    --stage sft \
    --from_weight ./output/vlm_pretrain/vlm_pretrain_768 \
    --data_path ./dataset/sft_data.parquet \
    --save_dir ./output/vlm_sft \
    --tokenizer_path ./vermind_tokenizer \
    --vision_encoder_path ./siglip-base-patch16-224 \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --freeze_vision 1 \
    --freeze_llm 0
```

**VLM Data Format**: Parquet files with `image_bytes` (binary) and `conversations` (JSON string) columns.

### 11. Deploy with vLLM

Start a high-performance API server compatible with OpenAI's client:

```bash
# Start the server
python vllm_adapter/start_server.py ./output/lora/lora_768/checkpoint_merged

# The server is now running at http://localhost:8000
```

### 11. Making API Requests

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
  author={nev8rz},
  year={2026},
  url={https://github.com/nev8rz/vermind}
}
```

---

<p align="center">Made with ❤️ by nev8rz</p>
