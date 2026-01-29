#!/bin/bash
# VerMind PPO (Proximal Policy Optimization) 训练脚本

tmux new-session -d -s ppo "
cd /root/vermind
source .venv/bin/activate

uv run python train/ppo.py \
    --data_path /root/vermind/dataset/rlaif-mini.jsonl \
    --save_dir /root/vermind/output/ppo \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --from_weight /root/vermind/output/sft_packed/sft_packed_768_4/checkpoint_339044 \
    --ref_weight /root/vermind/output/sft_packed/sft_packed_768_4/checkpoint_339044 \
    --reward_model_path /root/vermind/internlm2-1_8b-reward \
    --epochs 3 \
    --batch_size 8 \
    --accumulation_steps 2\
    --learning_rate 1e-6 \
    --max_seq_len 512 \
    --max_gen_len 1536 \
    --save_interval 100\
    --use_swanlab
"

# tmux attach -t ppo
# kill session
# tmux kill-session -t ppo
