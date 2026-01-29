# 新建conda 环境 安装 lm_eval 测试
# pip install lm_eval[api] transformers  torch==2.8.0 datasets==2.18.0
export MODEL_NAME=vermind
export TOKENIZER_PATH=./output/sft_packed/sft_packed_768_4/checkpoint_339044
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

lm_eval run \
  --model local-completions \
  --model_args model=$MODEL_NAME,tokenizer=$TOKENIZER_PATH,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,timeout=120 \
  --tasks "ceval*,cmmlu*,aclue*,tmmluplus*" \
  --batch_size 8 \
  --output_path chinese_benchmark.json \
  --trust_remote_code
