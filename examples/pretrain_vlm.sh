# VerMind-V (VLM) 预训练脚本
# 使用说明:
# 1. 先准备好 Parquet 格式的图文数据 (包含 image_bytes 和 conversations 列)
# 2. 修改 --data_path 指向你的数据文件
# 3. 如需从已有权重继续训练，设置 --from_weight 参数

tmux new-session -d -s vlm_pretrain "
cd /root/vermind
source .venv/bin/activate

uv run python train/pretrain_vlm.py \
    --from_weight /root/vermind/output/pretrain/pretrain_768/checkpoint_10000 \
    --data_path /root/vermind/dataset/vlm_pretrain.parquet \
    --save_dir /root/vermind/output/vlm_pretrain \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --vision_encoder_path /root/vermind/siglip-base-patch16-224 \
    --epochs 4 \
    --batch_size 64 \
    --accumulation_steps 8 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.03 \
    --max_seq_len 768 \
    --num_workers 8 \
    --save_interval 5000 \
    --log_interval 50 \
    --freeze_vision 1 \
    --freeze_llm 1 \
    --use_swanlab
"
#   uv run python train/pretrain_vlm.py \
#       --from_weight /root/vermind/output/dpo/dpo_768/checkpoint_1610 \
#       --data_path /root/vermind/dataset/vlm_pretrain.parquet \
#       --save_dir /root/vermind/output/vlm_pretrain \
#       --freeze_vision 1 \
#       --freeze_llm 1 \
#       --warmup_ratio 0.00 \
#       --learning_rate 4e-4 
# 常用参数说明:
# --freeze_vision 1      # 冻结 Vision Encoder (推荐)
# --freeze_llm 0         # 不冻结 LLM (全量训练) 或 1 (只训练 projection)
# --from_weight ./output/vlm_pretrain/vlm_pretrain_768  # 从已有 checkpoint 继续训练
# --from_resume 1        # 自动检测并续训

echo "VLM 预训练已启动，使用: tmux attach -t vlm_pretrain 查看日志"
