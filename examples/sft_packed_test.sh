#!/bin/bash
# VerMind SFT Packed 测试脚本（小数据集快速验证）

tmux new-session -d -s sft_packed_test "
cd /root/vermind
source .venv/bin/activate

uv run python train/sft_packed.py \
    --data_path /root/vermind/dataset/sft_sample_01pct.jsonl \
    --save_dir /root/vermind/output/sft_packed_test \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.03 \
    --num_workers 2 \
    --accumulation_steps 1 \
    --max_seq_len 1024 \
    --save_interval 999999 \
    --log_interval 20 \
    --monitor_steps 3 \
    --from_weight /root/vermind/output/pretrain \
    --use_packed 1 \
    --num_hidden_layers 4
"

# tmux attach -t sft_packed_test
# kill session
# tmux kill-session -t sft_packed_test
