import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
    AddedToken,
)
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq  

# 语料文件路径
root_path = Path(__file__).parent.parent
data_file = root_path / "vermind_dataset/tokenizer_data/fineweb_edu_sampled_6_percent"
save_path = root_path / "vermind_tokenizer"
save_path.mkdir(parents=True, exist_ok=True)
template_file = root_path / "vermind_dataset/tokenizer_data/chat_template.jinja2"

# 初始化tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) 
tokenizer.decoder = decoders.ByteLevel() 

# 定义特殊token
special_tokens = ["<|endoftext|>"]  # 这个特殊token相当于原生的特殊token，会被分配的id是0

# 设置训练器并添加特殊token
trainer = trainers.BpeTrainer(
    vocab_size = 32000,
    special_tokens = special_tokens,
    show_progress = True,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),  # 使用ByteLevel的默认字母表
    min_frequency = 3,
)

# 修改后的读取文件函数：保持函数名不变，内部改为流式读取以节省内存
def read_data(data_file):
    """
    流式读取目录下的所有 parquet 文件，避免一次性加载导致的内存溢出
    """
    for parquet_file in data_file.glob("**/*.parquet"):
        # 使用 pyarrow 逐块读取 RowGroup
        with pq.ParquetFile(parquet_file) as pf:
            for i in range(pf.num_row_groups):
                # 仅加载当前 RowGroup 的 'text' 列
                table = pf.read_row_group(i, columns=['text'])
                # 遍历当前块的数据
                for text in table.column('text'):
                    yield text.as_py()

# 训练tokenizer
print("Training tokenizer..., this may take a while.")
tokenizer.train_from_iterator(
    iterator=read_data(data_file),
    trainer=trainer
)

# 添加额外token (逻辑保持不变)
added_content = ["<|im_start|>", "<|im_end|>","<think>", "</think>"]  # 会沿着最大id继续分配id
special_flags = [True, True, False, False]  # 是否是特殊token

added_tokens = [
    AddedToken(content,
               single_word=False, lstrip=False, rstrip=False,
               normalized=False, special=sp)
    for content, sp in zip(added_content, special_flags)
]

num_added_tokens = tokenizer.add_tokens(added_tokens)
print(f"{num_added_tokens} new tokens added to the tokenizer.")

# 保存tokenizer
tokenizer.save(str(save_path / "tokenizer.json"))
tokenizer.model.save(str(save_path))

# 聊天模板
with open(str(template_file), 'r', encoding='utf-8') as f:
    chat_template = f.read()
tokenizer.chat_template = chat_template

# 创建配置文件 (字典结构和 ID 分配完全保留原样)
config = {
    "add_prefix_space": False,    
    "added_tokens_decoder": {
        "0": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "32000": {
            "content": "<|im_start|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "32001": {
            "content": "<|im_end|>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "32002": {
            "content": "<think>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": False
            },
        "32003": {
            "content": "</think>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": False
            }
        },
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
    ],
    "bos_token": None,
    "eos_token": "<|im_end|>",  # <|im_end|>作为对话结束标记
    "pad_token": "<|endoftext|>",
    "unk_token": None,
    "chat_template": chat_template,
    "clean_up_tokenization_spaces": False,
    "errors": "replace",
    "model_max_length": 10000000,
    "split_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "add_bos_token": False,
    }

# 保存配置文件
with open(str(save_path / "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, ensure_ascii=False, indent=4)

print("Tokenizer trained and saved successfully!")
