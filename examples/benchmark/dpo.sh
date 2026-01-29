#!/bin/bash
# VerMind DPO (Direct Preference Optimization) 训练脚本

tmux new-session -d -s dpo "
cd /root/vermind
source .venv/bin/activate

uv run python train/dpo.py \
    --data_path /root/vermind/dataset/dpo.jsonl \
    --save_dir /root/vermind/output/dpo \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --ref_weight /root/vermind/output/sft/full_sft_768 \
    --from_weight /root/vermind/output/sft/full_sft_768 \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.03 \
    --num_workers 6 \
    --accumulation_steps 1 \
    --max_seq_len 512 \
    --save_interval 1000 \
    --beta 0.1 \
    --dpo_aggregate mean \
    --use_swanlab
"

# tmux attach -t dpo
# kill session
# tmux kill-session -t dpo
