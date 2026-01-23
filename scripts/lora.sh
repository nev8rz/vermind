#!/bin/bash
# VerMind LoRA Fine-Tuning 训练脚本

tmux new-session -d -s lora "
cd /root/vermind
source .venv/bin/activate

uv run python train/lora.py \
    --data_path /root/vermind/dataset/lora_self_cognition.jsonl \
    --save_dir /root/vermind/output/lora \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --num_workers 6 \
    --accumulation_steps 1 \
    --save_interval 100 \
    --from_weight /root/vermind/output/sft/full_sft_768 \
    --lora_rank 16 \
    --log_interval 1 \
    --lora_target_modules 'q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj' \
    --use_swanlab
"

# tmux attach -t lora

#  python scripts/merge_lora.py     --model_path /root/vermind/output/sft/full_sft_768   --lora_path /root/vermind/output/lora/lora_768

# python vllm_adapter/start_server.py ./output/lora/lora_768/checkpoint_79_merged/