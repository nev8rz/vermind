# VerMind-V (VLM) SFT 脚本
# 使用说明:
# 1. 先准备好 Parquet 格式的图文指令数据 (包含 image_bytes 和 conversations 列)
# 2. 修改 --data_path 指向你的数据文件
# 3. 从 pretrain_vlm 的 checkpoint 继续训练

tmux new-session -d -s vlm_sft "
cd /root/vermind
source .venv/bin/activate

uv run python train/train_vlm.py \
    --stage sft \
    --from_weight /root/vermind/output/vlm_pretrain/vlm_pretrain_768 \
    --data_path /root/vermind/dataset/sft_data.parquet \
    --save_dir /root/vermind/output/vlm_sft \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --vision_encoder_path /root/vermind/siglip-base-patch16-224 \
    --epochs 3 \
    --batch_size 32 \
    --accumulation_steps 8 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --max_seq_len 768 \
    --num_workers 8 \
    --save_interval 2000 \
    --log_interval 50 \
    --freeze_vision 1 \
    --freeze_llm 0 \
    --use_swanlab
"

# 常用参数说明:
# --freeze_vision 1      # 冻结 Vision Encoder (推荐)
# --freeze_llm 0         # 解冻 LLM (SFT阶段全量训练)
# --from_resume 1        # 自动检测并续训

echo "VLM SFT 训练已启动，使用: tmux attach -t vlm_sft 查看日志"
