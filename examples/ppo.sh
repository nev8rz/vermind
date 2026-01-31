#!/bin/bash
# VerMind PPO (Proximal Policy Optimization) 训练脚本

tmux new-session -d -s ppo "
cd /root/vermind
source .venv/bin/activate

uv run python train/ppo.py \
    --data_path /root/vermind/dataset/rlaif-mini.jsonl \
    --save_dir /root/vermind/output/ppo \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --from_weight /root/vermind/output/reason/reason_768/checkpoint_17825 \
    --ref_weight /root/vermind/output/reason/reason_768/checkpoint_17825 \
    --reward_model_path /root/vermind/Skywork-Reward-V2-Qwen3-4B \
    --epochs 3 \
    --batch_size 8 \
    --accumulation_steps 2\
    --learning_rate 1e-6 \
    --reasoning 1 \
    --max_seq_len 512 \
    --max_gen_len 1536 \
    --save_interval 100 \
    --use_swanlab
"

# tmux attach -t ppo
# kill session
# tmux kill-session -t ppo
