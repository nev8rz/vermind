# VerMind-V（VLM）

本文覆盖 **视觉-语言模型** 的训练、评测与 Web Demo。

<p align="center">
  <img src="assets/vlm_sample.png" alt="VerMind-V Web Demo" width="90%">
</p>

## 项目内已有内容（VLM）

- 训练脚本：`train/train_vlm.py`
- 数据集加载：`data_loader/vlm_dataset.py`
- Web Demo：`scripts/web_demo.py`
- 推理/评测：`scripts/eval_vlm.py`

## 安装

```bash
git clone https://github.com/nev8rz/vermind.git
cd vermind
uv venv && source .venv/bin/activate
uv pip install -e .
```

## 数据格式（VLM）

来自 `data_loader/vlm_dataset.py`：

- Parquet 文件包含列：
  - `image_bytes`：图像二进制
  - `conversations`：JSON 字符串，对话列表，每条包含 `content`
- 数据集会将 `<image>` 替换为模型图像 token 序列。

## Web Demo（Gradio）

```bash
python scripts/web_demo.py \
  --model_path /path/to/vlm_checkpoint \
  --device cuda
```

注意：`--model_path` 目录需包含权重与 `config.json`。

## 评测 / 推理

### 本地推理

```bash
python scripts/eval_vlm.py --model_path /path/to/vlm_checkpoint --device cuda
```

### OpenAI 兼容 API 模式

若已启动 OpenAI 兼容服务：

```bash
python scripts/eval_vlm.py --use_api --api_base http://localhost:8000/v1 --model vermind-v
```

## 训练入口

```bash
python train/train_vlm.py --stage pretrain --from_weight ./output/sft/full_sft_768 --data_path ./dataset/vlm_pretrain.parquet
python train/train_vlm.py --stage sft --from_weight ./output/vlm_pretrain/vlm_pretrain_768 --data_path ./dataset/sft_data.parquet
```
