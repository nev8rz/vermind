#!/bin/bash
# VerMind Reasoning Model SFT 训练脚本
# 对 <think>, </think>, <answer>, </answer> 标签赋予更高权重

tmux new-session -d -s reason2 "
cd /root/vermind
source .venv/bin/activate

uv run python train/reason.py \
    --data_path /root/vermind/dataset/r1_mix_1024.jsonl \
    --save_dir /root/vermind/output/reason \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --from_weight /root/vermind/output/dpo/dpo_768/checkpoint_1610 \
    --epochs 3 \
    --batch_size 32 \
    --accumulation_steps 8 \
    --learning_rate 1e-5 \
    --max_seq_len 1024 \
    --tag_weight 10.0 \
    --save_interval 1000 \
    --use_swanlab
"

# tmux attach -t reason
# kill session
# tmux kill-session -t reason
