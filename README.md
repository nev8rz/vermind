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

**VerMind** is an end-to-end toolkit for building, training, and serving custom language models. It includes a core LLM stack and a multimodal VerMind‑V (VLM) extension with vision understanding.

## Highlights

- **LLM training pipeline**: tokenizer → pretrain → SFT (packed) → DPO / PPO / GRPO → LoRA
- **VLM support**: unified VLM training with image-text datasets
- **Inference**: OpenAI-compatible eval client and vLLM adapter
- **Gradio demo**: lightweight web UI for VerMind‑V

## Docs

- LLM: [docs/README_llm.md](https://github.com/nev8rz/vermind/blob/main/docs/README_llm.md) / [docs/README_llm_zh.md](https://github.com/nev8rz/vermind/blob/main/docs/README_llm_zh.md)
- VLM: [docs/README_vlm.md](https://github.com/nev8rz/vermind/blob/main/docs/README_vlm.md) / [docs/README_vlm_zh.md](https://github.com/nev8rz/vermind/blob/main/docs/README_vlm_zh.md)

## Quick Start

```bash
git clone https://github.com/nev8rz/vermind.git
cd vermind
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Demos

### VLM Web Demo (Gradio)

```bash
python scripts/web_demo.py --model_path /path/to/vlm_checkpoint --device cuda
```

### LLM API Client

Requires an OpenAI-compatible server (for example the local server in `vllm_adapter/`).

```bash
python scripts/eval_llm.py --api_base http://localhost:8000/v1 --model vermind
```

## Project Structure (Core)

```
vermind/
├── data_loader/         # Dataset loaders
├── train/               # Training scripts
├── scripts/             # Utilities & demos
├── vllm_adapter/        # OpenAI-compatible server adapter
├── docs/                # Documentation & pages
└── vermind_models/      # Model implementations
```
