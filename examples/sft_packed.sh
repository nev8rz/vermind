#!/bin/bash
# VerMind SFT Packed (打包数据集训练) 训练脚本

tmux new-session -d -s sft_packed "
cd /root/vermind
source .venv/bin/activate

uv run python train/sft_packed.py \
    --data_path /root/vermind/dataset/sft.jsonl \
    --save_dir /root/vermind/output/sft_packed \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --epochs 3 \
    --batch_size 64 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --num_workers 6 \
    --accumulation_steps 16 \
    --max_seq_len 2048 \
    --save_interval 2000 \
    --from_weight /root/vermind/output/pretrain/pretrain_768 \
    --use_packed 1 \
    --use_swanlab
"

# tmux attach -t sft_packed
# kill session
# tmux kill-session -t sft_packed
