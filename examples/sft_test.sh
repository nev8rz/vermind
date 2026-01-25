#!/bin/bash
# 测试 SFT 训练脚本（packed 和非 packed 数据）

# 设置默认参数
DATA_PATH="../dataset/sft_sample_01pct.jsonl"
TOKENIZER_PATH="/root/vermind/vermind_tokenizer"
SAVE_DIR="../out"
BATCH_SIZE=4
MAX_SEQ_LEN=2048
EPOCHS=1
LOG_INTERVAL=5
MONITOR_STEPS=5

echo "=========================================="
echo "测试 1: 普通数据集（非 packed）"
echo "=========================================="
python train/sft_copy.py \
    --data_path $DATA_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --epochs $EPOCHS \
    --log_interval $LOG_INTERVAL \
    --monitor_steps $MONITOR_STEPS \
    --use_packed 0 \
    --num_hidden_layers 8 \
    --save_weight sft_test_normal

echo ""
echo "=========================================="
echo "测试 2: 打包数据集（packed）"
echo "=========================================="
python train/sft_copy.py \
    --data_path $DATA_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --epochs $EPOCHS \
    --log_interval $LOG_INTERVAL \
    --monitor_steps $MONITOR_STEPS \
    --use_packed 1 \
    --num_hidden_layers 8 \
    --save_weight sft_test_packed

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
