#!/bin/bash
# VerMind SFT (Supervised Fine-Tuning) 训练脚本

tmux new-session -d -s sft "
cd /root/vermind
source .venv/bin/activate

uv run python train/sft.py \
    --data_path /root/vermind/dataset/sft_2048.jsonl \
    --save_dir /root/vermind/output/sft_2048 \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --num_workers 6 \
    --accumulation_steps  8\
    --max_seq_len 1600 \
    --save_interval 2000 \
    --from_weight /root/vermind/output/sft_1024/full_sft_768 \
    --use_swanlab
"

# tmux attach -t sft
# kill session
# tmux kill-session -t sft