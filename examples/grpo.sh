#!/bin/bash
# VerMind GRPO (Group Relative Policy Optimization) 训练脚本
# GRPO 使用组内相对优势，无需critic模型

tmux new-session -d -s grpo "
cd /root/vermind
source .venv/bin/activate

uv run python train/grpo.py \
    --data_path /root/vermind/dataset/rlaif-mini.jsonl \
    --save_dir /root/vermind/output/grpo \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --from_weight /root/vermind/output/reason/reason_768/checkpoint_35651 \
    --ref_weight /root/vermind/output/reason/reason_768/checkpoint_35651 \
    --reward_model_path /root/vermind/Skywork-Reward-V2-Qwen3-4B \
    --epochs 3 \
    --batch_size 8 \
    --num_generations 4 \
    --reasoning 1 \
    --accumulation_steps 2 \
    --learning_rate 1e-6 \
    --max_seq_len 512 \
    --max_gen_len 1536 \
    --beta 0.05 \
    --save_interval 100 \
    --use_swanlab
"

# tmux attach -t grpo
# kill session
# tmux kill-session -t grpo
