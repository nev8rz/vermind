# VerMind

<p align="center">
  <a href="https://github.com/nev8rz/vermind">
    <img src="docs/assets/vermind_logo.png" alt="VerMind" width="120">
  </a>
</p>

<p align="center">
  <a href="https://github.com/nev8rz/vermind">
    <img src="https://cdn.simpleicons.org/github/ffffff" alt="GitHub" height="14"> GitHub
  </a> ·
  <a href="https://huggingface.co/nev8r/vermind">
    <img src="https://cdn.simpleicons.org/huggingface/ffd21e" alt="Hugging Face" height="14"> HF (LLM)
  </a> ·
  <a href="https://huggingface.co/nev8r/vermind-v">
    <img src="https://cdn.simpleicons.org/huggingface/ffd21e" alt="Hugging Face" height="14"> HF (VLM)
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=for-the-badge&logo=python" alt="Python 3.12+">
</p>

---

**VerMind** 是一个端到端的语言模型工具链，覆盖训练、微调与部署，并提供多模态 VerMind‑V（VLM）扩展以支持视觉理解。

## 亮点

- **LLM 训练流程**：分词器 → 预训练 → SFT（含打包训练）→ DPO / PPO / GRPO → LoRA
- **VLM 支持**：统一的图文训练脚本
- **推理能力**：OpenAI 兼容客户端 + vLLM 适配
- **Gradio Demo**：轻量级 VerMind‑V 网页界面

## 文档

- LLM：[docs/README_llm.md](https://github.com/nev8rz/vermind/blob/main/docs/README_llm.md) / [docs/README_llm_zh.md](https://github.com/nev8rz/vermind/blob/main/docs/README_llm_zh.md)
- VLM：[docs/README_vlm.md](https://github.com/nev8rz/vermind/blob/main/docs/README_vlm.md) / [docs/README_vlm_zh.md](https://github.com/nev8rz/vermind/blob/main/docs/README_vlm_zh.md)

## 快速开始

```bash
git clone https://github.com/nev8rz/vermind.git
cd vermind
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Demo

### VLM Web Demo（Gradio）

```bash
python scripts/web_demo.py --model_path /path/to/vlm_checkpoint --device cuda
```

### LLM API 客户端

需要 OpenAI 兼容服务（例如 `vllm_adapter/` 中的本地服务）。

```bash
python scripts/eval_llm.py --api_base http://localhost:8000/v1 --model vermind
```

## 项目结构（核心）

```
vermind/
├── data_loader/         # 数据集加载
├── train/               # 训练脚本
├── scripts/             # 工具与 Demo
├── vllm_adapter/        # OpenAI 兼容服务
├── docs/                # 文档与页面
└── vermind_models/      # 模型实现
```
