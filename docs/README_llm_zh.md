# VerMind LLM

本文件覆盖 **纯文本 LLM** 的训练、微调与推理流程。

## 项目内已有内容（LLM）

- 训练脚本：`train/pretrain.py`、`train/sft.py`、`train/sft_packed.py`、`train/lora.py`、`train/dpo.py`、`train/ppo.py`、`train/grpo.py`
- 分词器训练：`train/train_tokenizer.py`
- 推理 / 评测（OpenAI 兼容客户端）：`scripts/eval_llm.py`
- LoRA 合并：`scripts/merge_lora.py`
- vLLM 适配：`vllm_adapter/`

## 安装

```bash
git clone https://github.com/nev8rz/vermind.git
cd vermind
uv venv && source .venv/bin/activate
uv pip install -e .
```

## 数据格式（LLM）

以下格式来自 `data_loader/` 的实际实现。

- **预训练**（`data_loader/pretrain_dataset.py`）
  - JSONL，包含 `text` 字段
- **SFT**（`data_loader/sft_dataset.py`）
  - JSONL，包含 `conversations: [{"role": "user|assistant|system", "content": "..."}, ...]`
- **打包式 SFT**
  - 先用 `scripts/pre_sftdatapacked.py` 把 SFT JSONL 转为 Parquet，包含 `input_ids`、`labels`、`boundaries`
- **DPO**（`data_loader/dpo_dataset.py`）
  - JSONL，包含 `chosen` 与 `rejected` 两个对话数组（格式同 SFT）
- **PPO/RLAIF**（`data_loader/rlaif_dataset.py`）
  - JSONL，包含 `conversations: [{"content": "..."}, ...]`（角色按顺序推断）

## 训练入口

可直接运行脚本或使用 `examples/`：

```bash
# 分词器
python train/train_tokenizer.py --data_path /path/to/corpus.txt --tokenizer_dir ./vermind_tokenizer --vocab_size 6400

# 预训练
python train/pretrain.py --data_path /path/to/pretrain.jsonl --save_dir ./output/pretrain --tokenizer_path ./vermind_tokenizer

# SFT
python train/sft.py --data_path /path/to/sft.jsonl --save_dir ./output/sft --from_weight ./output/pretrain/pretrain_768

# 打包式 SFT（需预处理）
python scripts/pre_sftdatapacked.py --jsonl_path /path/to/sft.jsonl --output_path ./cache/sft_packed/sft.parquet --tokenizer_path ./vermind_tokenizer
python train/sft_packed.py --data_path /path/to/sft.jsonl --parquet_path ./cache/sft_packed/sft.parquet --save_dir ./output/sft_packed --from_weight ./output/pretrain/pretrain_768

# LoRA
python train/lora.py --data_path /path/to/lora.jsonl --save_dir ./output/lora --from_weight ./output/sft/full_sft_768

# DPO / PPO / GRPO
python train/dpo.py  --data_path /path/to/dpo.jsonl  --save_dir ./output/dpo  --from_weight ./output/sft/full_sft_768 --ref_weight ./output/sft/full_sft_768
python train/ppo.py  --data_path /path/to/ppo.jsonl  --save_dir ./output/ppo  --from_weight ./output/sft/full_sft_768 --ref_weight ./output/sft/full_sft_768
python train/grpo.py --data_path /path/to/grpo.jsonl --save_dir ./output/grpo --from_weight ./output/sft/full_sft_768 --ref_weight ./output/sft/full_sft_768
```

## 推理

### OpenAI 兼容客户端

`scripts/eval_llm.py` 连接 OpenAI 兼容接口（例如本地 vLLM 服务）。

```bash
python scripts/eval_llm.py --api_base http://localhost:8000/v1 --model vermind
```

### vLLM 适配

OpenAI 兼容服务位于 `vllm_adapter/`。
