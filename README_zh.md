#
<div align="center">
  <a href="https://github.com/nev8rz/vermind">
    <img src="https://raw.githubusercontent.com/nev8rz/vermind/main/docs/assets/vermind_logo.png" alt="VerMind Logo" width="120">
  </a>
  <h1 align="center">VerMind</h1>
  <p align="center">
    一个从零开始、基于 PyTorch 构建的高性能、轻量级现代语言模型。
    <br />
    <a href="https://nev8rz.github.io/vermind/"><strong>查看演示 »</strong></a>
    ·
    <a href="https://github.com/nev8rz/vermind/issues">报告 Bug</a>
    ·
    <a href="https://github.com/nev8rz/vermind/issues">请求功能</a>
  </p>
</div>

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch 2.8.0+](https://img.shields.io/badge/PyTorch-2.8.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/nev8rz/vermind?style=for-the-badge&logo=github)](https://github.com/nev8rz/vermind/stargazers)

</div>

---

**VerMind** 是一个用于构建、训练和部署自定义语言模型的综合性端到端工具包。它采用了最先进的架构，包括 **分组查询注意力 (GQA)** 和 **SwiGLU**，专为高效训练、微调和高吞吐量推理而设计。该项目高度模块化，文档详尽，易于定制，是研究和生产的理想起点。

## ✨ 为什么选择 VerMind？

-   🚀 **性能与效率**: 实现 GQA 和 Flash Attention，减少内存占用，加速训练和推理。
-   🧠 **现代架构**: 融合了 LLM 架构的最新进展，如 SwiGLU 激活函数和用于长文本理解的 RoPE (旋转位置嵌入) 及 YaRN 缩放。
-   🔧 **端到端解决方案**: 提供从分词器训练、预训练到监督微调 (SFT)、LoRA 以及使用 vLLM 适配器进行部署的完整工作流。
-   🧩 **可扩展性与定制化**: 模块化设计使得实验新想法、更换组件和根据特定需求调整模型变得容易。
-   🎓 **教育价值**: 凭借详细的代码和文档，成为理解现代语言模型内部工作原理的绝佳学习资源。

## 🛠️ 核心功能

| 功能 | 描述 |
|---|---|
| ⚡ **分组查询注意力 (GQA)** | 通过共享键值头来减少推理所需的内存带宽，从而实现显著的速度提升。 |
| 🔥 **SwiGLU 激活函数** | 一种现代激活函数，通常比传统的 ReLU 或 GeLU 带来更好的性能。 |
| 📐 **旋转位置嵌入 (RoPE)** | 一种相对位置编码方案，已成为高性能语言模型的标配。包含 YaRN 缩放以扩展上下文长度。 |
| 🚀 **vLLM 适配器** | 支持极速推理，并提供与 OpenAI 兼容的 API 服务器，开箱即用。 |
| 🎨 **LoRA 微调** | 支持使用低秩自适应 (LoRA) 进行参数高效微调 (PEFT)，实现快速、低内存占用的定制化。 |
| 🌐 **分布式训练** | 内置对分布式数据并行 (DDP) 的支持，可将训练扩展到多个 GPU。 |

## 🏗️ 架构概览

VerMind 的架构是一个为性能和可扩展性而优化的仅解码器 Transformer 模型。核心组件设计得既高效又易于理解。

![VerMind Architecture](https://raw.githubusercontent.com/nev8rz/vermind/main/docs/assets/architecture.png)

-   **RMSNorm**: 用于层归一化，提供比标准 LayerNorm 更好的稳定性。
-   **旋转位置嵌入 (RoPE)**: 应用于查询和键，以注入位置信息。
-   **分组查询注意力 (GQA)**: 注意力模块，其中多个查询头关注单个键值头。
-   **SwiGLU 前馈网络**: FFN 模块使用 SwiGLU 激活函数以获得更好的性能。

## 🚀 快速开始

只需几个简单步骤即可在本地运行。

### 环境要求

-   Python 3.12+
-   PyTorch 2.8.0+
-   `uv` 包管理器 (推荐)

### 安装

1.  **克隆仓库**
    ```sh
    git clone https://github.com/nev8rz/vermind.git
    cd vermind
    ```
2.  **创建并激活虚拟环境**
    ```sh
    uv venv
    source .venv/bin/activate
    ```
3.  **安装依赖**
    ```sh
    uv pip install -e .
    ```

## 🏃‍♀️ 使用示例

VerMind 提供了一个完整的训练流程，并在 `examples/` 目录中提供了便捷的 Shell 脚本。训练工作流如下：**分词器 → 预训练 → SFT → LoRA → 部署**。

### 1. 训练分词器

首先，在你的语料库上训练一个自定义分词器：

```bash
python train/train_tokenizer.py \
    --data_path /path/to/training_corpus.txt \
    --tokenizer_dir ./vermind_tokenizer \
    --vocab_size 6400
```

### 2. 预训练

在大规模语料库上从头开始预训练模型。使用提供的脚本或直接运行：

```bash
# 方式一：使用启动脚本 (在 tmux 中运行)
bash examples/pretrain.sh

# 方式二：使用自定义参数直接运行
python train/pretrain.py \
    --data_path /path/to/pretrain_data.jsonl \
    --save_dir ./output/pretrain \
    --tokenizer_path ./vermind_tokenizer \
    --epochs 5 \
    --batch_size 128 \
    --learning_rate 1e-3
```

### 3. 监督微调 (SFT)

在指令遵循数据上对预训练模型进行微调：

```bash
# 方式一：使用启动脚本 (在 tmux 中运行)
bash examples/sft.sh

# 方式二：使用自定义参数直接运行
python train/sft.py \
    --data_path /path/to/sft_data.jsonl \
    --save_dir ./output/sft \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/pretrain/pretrain_768 \
    --epochs 3 \
    --learning_rate 5e-6
```

### 4. LoRA 微调

对于参数高效的微调，使用 LoRA 以最少的资源适配模型：

```bash
# 方式一：使用启动脚本 (在 tmux 中运行)
bash examples/lora.sh

# 方式二：使用自定义参数直接运行
python train/lora.py \
    --data_path /path/to/lora_data.jsonl \
    --save_dir ./output/lora \
    --tokenizer_path ./vermind_tokenizer \
    --from_weight ./output/sft/full_sft_768 \
    --epochs 5 \
    --learning_rate 1e-4 \
    --lora_rank 16
```

### 5. 合并 LoRA 权重

LoRA 训练后，将适配器权重合并到基础模型中：

```bash
python scripts/merge_lora.py \
    --model_path ./output/sft/full_sft_768 \
    --lora_path ./output/lora/lora_768
```

### 6. 模型评估

以交互方式或自动测试模式评估您的模型：

```bash
# 交互式聊天模式
python scripts/eval_llm.py \
    --load_from ./output/lora/lora_768/checkpoint_merged \
    --use_chat_template 1
```

### 7. 使用 vLLM 部署

启动与 OpenAI 客户端兼容的高性能 API 服务器：

```bash
# 启动服务器
python vllm_adapter/start_server.py ./output/lora/lora_768/checkpoint_merged

# 服务器现在运行在 http://localhost:8000
```

### 8. 发起 API 请求

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="./output/lora/lora_768/checkpoint_merged",
    messages=[
        {"role": "user", "content": "解释一下分组查询注意力的重要性。"}
    ],
)
print(response.choices[0].message.content)
```

## 🤝 贡献

欢迎各种贡献！

1.  Fork 本项目
2.  创建您的功能分支 (`git checkout -b feature/AmazingFeature`)
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  开启一个 Pull Request

## 📜 许可证

根据 MIT 许可证分发。详见 `LICENSE` 文件。

## ✒️ 引用

如果您在研究或工作中使用了 VerMind，请考虑引用：

```bibtex
@software{vermind2026,
  title={VerMind: A High-Performance, Lightweight Language Model with GQA},
  author={Yijin Zhou},
  year={2026},
  url={https://github.com/nev8rz/vermind}
}
```

---

<p align="center">由 Yijin Zhou 用 ❤️ 制作</p>
