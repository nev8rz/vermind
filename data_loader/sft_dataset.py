
import os
import bisect
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import pyarrow.parquet as pq
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_CACHE_DIR = (PROJECT_ROOT / ".cache" / "sft_dataset").as_posix()


class SFTDataset(Dataset):
    """普通 SFT 数据集（非 packed）"""
    def __init__(self, jsonl_path, tokenizer, max_length=2048, cache_dir=SFT_CACHE_DIR):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SFTDatasetPacked(Dataset):
    """
    Packed SFT 数据集
    从预处理好的 Parquet 文件加载数据（由 scripts/pre_sftdatapacked.py 生成）
    使用 PyArrow 流式读取，避免一次性加载整个文件到内存
    """
    def __init__(self, parquet_path: str, tokenizer=None, max_length: int = 2048):
        """
        Args:
            parquet_path: 预处理好的 Parquet 文件路径（由 pre_sftdatapacked.py 生成）
            tokenizer: 可选，仅用于兼容性（实际不需要）
            max_length: 最大序列长度（从 Parquet 中读取，此参数仅用于验证）
        """
        super().__init__()
        self.parquet_path = parquet_path
        self.max_length = max_length
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"Parquet 文件不存在: {parquet_path}\n"
                f"请先运行 scripts/pre_sftdatapacked.py 预处理数据"
            )
        
        print(f"[SFTDatasetPacked] 打开 Parquet 文件: {parquet_path}")
        self.parquet_file = pq.ParquetFile(parquet_path)
        
        schema = self.parquet_file.schema_arrow
        required_cols = ['input_ids', 'labels', 'boundaries']
        missing_cols = [col for col in required_cols if col not in schema.names]
        if missing_cols:
            raise ValueError(f"Parquet 文件缺少必要的列: {missing_cols}")
        
        metadata = self.parquet_file.metadata
        self.num_rows = metadata.num_rows
        
        row_group_offsets = [0]
        for i in range(metadata.num_row_groups):
            row_group_offsets.append(row_group_offsets[-1] + metadata.row_group(i).num_rows)
        self.row_group_offsets = row_group_offsets
        
        if self.num_rows > 0:
            first_table = self.parquet_file.read_row_group(0, columns=required_cols)
            first_row = first_table.to_pandas().iloc[0]
            first_seq_len = len(first_row['input_ids'])
            if first_seq_len != max_length:
                print(f"[SFTDatasetPacked] 警告: Parquet 中的序列长度 ({first_seq_len}) 与 max_length ({max_length}) 不匹配")
                self.max_length = first_seq_len
        else:
            self.max_length = max_length
        
        print(f"[SFTDatasetPacked] 文件打开成功: {self.num_rows:,} 个序列")
        print(f"[SFTDatasetPacked] 序列长度: {self.max_length}")
        print(f"[SFTDatasetPacked] Row groups: {metadata.num_row_groups}")
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, index):
        row_group_idx = bisect.bisect_right(self.row_group_offsets, index) - 1
        row_in_group = index - self.row_group_offsets[row_group_idx]
        
        table = self.parquet_file.read_row_group(row_group_idx, columns=['input_ids', 'labels', 'boundaries'])
        df = table.to_pandas()
        row = df.iloc[row_in_group]
        
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        labels = torch.tensor(row['labels'], dtype=torch.long)
        boundaries = row['boundaries']
        return input_ids, labels, boundaries


def collate_fn_packed(batch):
    all_input_ids = []
    all_labels = []
    all_position_ids = []
    batch_cu_seqlens = [0]
    
    for item in batch:
        input_ids, labels, boundaries = item
        seq_len = len(input_ids)
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        
        position_ids = torch.zeros(seq_len, dtype=torch.long)
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], min(boundaries[i + 1], seq_len)
            position_ids[s:e] = torch.arange(0, e - s, dtype=torch.long)
        all_position_ids.append(position_ids)
        
        batch_cu_seqlens.append(batch_cu_seqlens[-1] + seq_len)
    
    input_ids = torch.cat(all_input_ids, dim=0)
    labels = torch.cat(all_labels, dim=0)
    position_ids = torch.cat(all_position_ids, dim=0)
    cu_seqlens = torch.tensor(batch_cu_seqlens, dtype=torch.int32)
    
    return input_ids, labels, cu_seqlens, position_ids
