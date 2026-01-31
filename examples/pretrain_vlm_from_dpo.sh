tmux new-session -d -s vlm_pretrain "
cd /root/vermind
source .venv/bin/activate

# VerMind-V (VLM) 预训练脚本 - 从 DPO checkpoint 加载 LLM
# 使用 DPO 训练后的权重作为 LLM 基础，添加 Vision 能力

uv run python train/pretrain_vlm.py \
    --from_weight /root/vermind/output/dpo/dpo_768/checkpoint_1610 \
    --data_path /root/vermind/dataset/vlm_pretrain.parquet \
    --save_dir /root/vermind/output/vlm_pretrain \
    --save_weight vlm_from_dpo \
    --tokenizer_path /root/vermind/vermind_tokenizer \
    --vision_encoder_path /root/vermind/siglip-base-patch16-224 \
    --epochs 3 \
    --batch_size 8 \
    --accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --max_seq_len 512 \
    --num_workers 4 \
    --save_interval 1000 \
    --log_interval 100 \
    --freeze_vision 1 \
    --freeze_llm 0 \
    --use_swanlab
"

# 参数说明:
# --from_weight /root/vermind/output/dpo/dpo_768/checkpoint_1610
#     从 DPO checkpoint 加载 LLM 权重
#     会自动加载 model 和 lm_head，Vision Proj 随机初始化
#
# --freeze_vision 1       # 冻结 SigLIP
# --freeze_llm 0          # 训练 LLM + Vision Proj
#
# 如果想先只训练 Vision Proj，保持 LLM 冻结:
# --freeze_llm 1 --learning_rate 1e-3

echo "VLM 预训练已启动（从 DPO checkpoint），使用: tmux attach -t vlm_pretrain 查看日志"
