tmux new-session -d -s pretrain "
cd /root/vermind
source .venv/bin/activate

uv run python train/pretrain.py \
    --data_path /root/vermind/dataset/pretrain_hq.jsonl \
    --save_dir /root/vermind/output/pretrain \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --epochs 5 \
    --num_workers 6 \
    --warmup_ratio 0.03 \
    --learning_rate 1e-3 \
    --batch_size 128 \
    --accumulation_steps 16 \
    --save_interval 2000 \
    --use_swanlab
"

# tmux attach -t pretrain