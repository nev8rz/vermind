#!/bin/bash
# VerMind LoRA Fine-Tuning 训练脚本

tmux new-session -d -s lora "
cd /root/vermind
source .venv/bin/activate

uv run python train/lora.py \
    --data_path /root/vermind/dataset/lora_self_cognition.jsonl \
    --save_dir /root/vermind/output/lora \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.03 \
    --num_workers 6 \
    --accumulation_steps 1 \
    --save_interval 100 \
    --from_weight /root/vermind/output/sft/sft_768 \
    --lora_rank 16 \
    --lora_target_modules 'q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj' \
    --use_swanlab
"

# tmux attach -t lora
