# VerMind LLM

This document covers the **text-only** VerMind language model workflow: training, fine-tuning, and local/API inference.

## What’s in this repo (LLM)

- Training scripts: `train/pretrain.py`, `train/sft.py`, `train/sft_packed.py`, `train/lora.py`, `train/dpo.py`, `train/ppo.py`, `train/grpo.py`
- Tokenizer training: `train/train_tokenizer.py`
- Inference / eval (OpenAI-compatible client): `scripts/eval_llm.py`
- LoRA merge utility: `scripts/merge_lora.py`
- vLLM adapter: `vllm_adapter/`

## Install

```bash
git clone https://github.com/nev8rz/vermind.git
cd vermind
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Data formats (LLM)

These formats are derived from the dataset loaders in `data_loader/`.

- **Pretrain** (`data_loader/pretrain_dataset.py`)
  - JSONL with a `text` field
- **SFT** (`data_loader/sft_dataset.py`)
  - JSONL with `conversations: [{"role": "user|assistant|system", "content": "..."}, ...]`
- **Packed SFT**
  - Use `scripts/pre_sftdatapacked.py` to convert SFT JSONL → Parquet with `input_ids`, `labels`, `boundaries`
- **DPO** (`data_loader/dpo_dataset.py`)
  - JSONL with `chosen` and `rejected` arrays, each in the same `conversations` format as SFT
- **PPO/RLAIF** (`data_loader/rlaif_dataset.py`)
  - JSONL with `conversations: [{"content": "..."}, ...]` (roles are inferred by turn order)

## Training entry points

Use the provided scripts directly or via `examples/`:

```bash
# Tokenizer
python train/train_tokenizer.py --data_path /path/to/corpus.txt --tokenizer_dir ./vermind_tokenizer --vocab_size 6400

# Pretrain
python train/pretrain.py --data_path /path/to/pretrain.jsonl --save_dir ./output/pretrain --tokenizer_path ./vermind_tokenizer

# SFT
python train/sft.py --data_path /path/to/sft.jsonl --save_dir ./output/sft --from_weight ./output/pretrain/pretrain_768

# Packed SFT (requires preprocessing)
python scripts/pre_sftdatapacked.py --jsonl_path /path/to/sft.jsonl --output_path ./cache/sft_packed/sft.parquet --tokenizer_path ./vermind_tokenizer
python train/sft_packed.py --data_path /path/to/sft.jsonl --parquet_path ./cache/sft_packed/sft.parquet --save_dir ./output/sft_packed --from_weight ./output/pretrain/pretrain_768

# LoRA
python train/lora.py --data_path /path/to/lora.jsonl --save_dir ./output/lora --from_weight ./output/sft/full_sft_768

# DPO / PPO / GRPO
python train/dpo.py  --data_path /path/to/dpo.jsonl  --save_dir ./output/dpo  --from_weight ./output/sft/full_sft_768 --ref_weight ./output/sft/full_sft_768
python train/ppo.py  --data_path /path/to/ppo.jsonl  --save_dir ./output/ppo  --from_weight ./output/sft/full_sft_768 --ref_weight ./output/sft/full_sft_768
python train/grpo.py --data_path /path/to/grpo.jsonl --save_dir ./output/grpo --from_weight ./output/sft/full_sft_768 --ref_weight ./output/sft/full_sft_768
```

## Inference

### OpenAI-compatible client demo

`scripts/eval_llm.py` connects to an OpenAI-compatible endpoint (for example the local vLLM server in `vllm_adapter/`).

```bash
python scripts/eval_llm.py --api_base http://localhost:8000/v1 --model vermind
```

### vLLM adapter

See `vllm_adapter/` for the OpenAI-compatible server implementation.
