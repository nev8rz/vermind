"""
预处理 SFT 数据为 Packed Parquet 格式 (完整修复版)
功能：
1. 读取 JSONL -> Tokenize -> 生成 Labels
2. Greedy Packing -> 暂存为分片 (Shards)
3. 使用 PyArrow 流式合并分片 (解决 OOM 问题)

适配命令：
python scripts/pre_sftdatapacked.py merge --temp_dir ... --output_path ... --keep_shards
"""

import json
import gc
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# ==========================================

# ==========================================

def process_jsonl_to_packed_parquet(
    jsonl_path: str,
    output_parquet_path: str,
    tokenizer_path: str,
    max_seq_len: int = 2048,
    ignore_index: int = -100,
    pad_token_id: Optional[int] = None,
):
    """
    预处理主流程
    """
    print(f"[PreSFT] 开始处理: {jsonl_path}")
    print(f"[PreSFT] 目标输出: {output_parquet_path}")
    print(f"[PreSFT] Max Seq Len: {max_seq_len}")
    

    try:
        print("[PreSFT] 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"[Error] 加载 Tokenizer 失败: {e}")
        return


    if pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = 0 # Fallback
            


    temp_dir = Path(output_parquet_path).parent / "temp_packing"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    

    batch_size = 20000
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "shards": 0
    }
    
    current_batch = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = 0

    print("[PreSFT] 开始读取与 Packing...")
    

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            try:
                line = line.strip()
                if not line: continue
                
                data = json.loads(line)

                messages = data.get('conversations', data.get('messages', []))
                
                if not messages:
                    stats["skipped"] += 1
                    continue
                

                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except Exception:
                    stats["skipped"] += 1
                    continue
                
                # Tokenize
                tokenized = tokenizer(
                    text,
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors=None
                )
                
                input_ids = tokenized['input_ids']
                
                if not input_ids:
                    stats["skipped"] += 1
                    continue
                

                if len(input_ids) > max_seq_len:
                    input_ids = input_ids[:max_seq_len]
                

                labels = list(input_ids)
                

                current_batch.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "length": len(input_ids)
                })
                stats["processed"] += 1
                

                if len(current_batch) >= batch_size:
                    save_shard(current_batch, temp_dir, stats["shards"], max_seq_len, pad_token_id, ignore_index)
                    stats["shards"] += 1
                    current_batch = []
                    gc.collect()

            except json.JSONDecodeError:
                stats["errors"] += 1
            except Exception:
                stats["errors"] += 1
    

    if current_batch:
        save_shard(current_batch, temp_dir, stats["shards"], max_seq_len, pad_token_id, ignore_index)
        stats["shards"] += 1
        del current_batch
        gc.collect()

    print(f"\n[PreSFT] 预处理统计: {stats}")
    

    merge_shards_stream(temp_dir, output_parquet_path, keep_shards=False)


def save_shard(batch_samples: List[Dict], temp_dir: Path, shard_idx: int, max_seq_len: int, pad_token_id: int, ignore_index: int):
    """
    对一个 Batch 进行 Packing 并保存为 Parquet 分片
    """
    if not batch_samples:
        return


    packed_data = greedy_pack_samples(batch_samples, max_seq_len, pad_token_id, ignore_index)
    
    if not packed_data:
        return

    shard_path = temp_dir / f"shard_{shard_idx:05d}.parquet"
    

    df = pd.DataFrame(packed_data)
    

    df.to_parquet(shard_path, index=False, engine='pyarrow', compression='snappy')


def greedy_pack_samples(samples: List[Dict], max_seq_len: int, pad_token_id: int, ignore_index: int) -> List[Dict]:
    """
    贪婪打包算法 (Best-fit / First-fit desc)
    """

    samples.sort(key=lambda x: x['length'], reverse=True)
    
    bins = []
    
    for sample in samples:
        placed = False
        sample_len = sample['length']
        

        for bin_item in bins:
            if bin_item['current_len'] + sample_len <= max_seq_len:
                bin_item['samples'].append(sample)
                bin_item['current_len'] += sample_len
                placed = True
                break
        

        if not placed:
            bins.append({
                'samples': [sample],
                'current_len': sample_len
            })
    
    packed_result = []
    for bin_item in bins:
        input_ids = []
        labels = []
        boundaries = [0]
        

        for samp in bin_item['samples']:
            input_ids.extend(samp['input_ids'])
            labels.extend(samp['labels'])
            
            boundaries.append(len(input_ids))
        
        # Padding
        seq_len = len(input_ids)
        pad_len = max_seq_len - seq_len
        
        if pad_len > 0:
            input_ids.extend([pad_token_id] * pad_len)
            labels.extend([ignore_index] * pad_len)
        

        packed_result.append({
            "input_ids": input_ids,
            "labels": labels,
            "boundaries": boundaries
        })
        
    return packed_result


def merge_shards_stream(temp_dir: Path, output_path: str, keep_shards: bool = False):
    """
    【内存安全】流式合并分片
    使用 PyArrow.ParquetWriter 逐个写入，内存占用极低。
    """
    temp_dir = Path(temp_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[Merge] 来源目录: {temp_dir}")
    print(f"[Merge] 输出文件: {output_path}")
    

    shard_files = sorted(list(temp_dir.glob("shard_*.parquet")))
    if not shard_files:
        print("[Merge] 错误: 未找到分片文件，请检查路径。")
        return

    print(f"[Merge] 找到 {len(shard_files)} 个分片，准备合并...")

    try:

        first_table = pq.read_table(shard_files[0])
        schema = first_table.schema
    except Exception as e:
        print(f"[Merge] 读取第一个分片失败: {e}")
        return

    try:
        with pq.ParquetWriter(output_path, schema, compression='snappy') as writer:
            

            writer.write_table(first_table)
            del first_table
            gc.collect()
            

            for shard_path in tqdm(shard_files[1:], desc="Merging Shards"):
                try:

                    table = pq.read_table(shard_path)
                    

                    if table.schema != schema:
                        table = table.cast(schema)
                    

                    writer.write_table(table)
                    

                    del table
                    gc.collect()
                    
                except Exception as e:
                    print(f"[Merge] 警告: 分片 {shard_path.name} 合并失败: {e}")
                    
        print(f"[Merge] 合并成功!")
        file_size_gb = output_path.stat().st_size / (1024**3)
        print(f"[Merge] 最终文件大小: {file_size_gb:.2f} GB")


        if not keep_shards:
            print("[Merge] 正在清理临时分片...")
            shutil.rmtree(temp_dir)
            print("[Merge] 清理完成。")
        else:
            print(f"[Merge] 保留分片文件于: {temp_dir}")
            
    except Exception as e:
        print(f"[Merge] 致命错误: {e}")


# ==========================================

# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 数据预处理与打包工具")
    subparsers = parser.add_subparsers(dest='command', required=True, help="子命令: preprocess 或 merge")
    
    # --------------------------------------
    # --------------------------------------
    pre_parser = subparsers.add_parser('preprocess', help='预处理 JSONL 并打包')
    pre_parser.add_argument("--jsonl_path", type=str, required=True, help="输入 JSONL 文件路径")
    pre_parser.add_argument("--output_path", type=str, required=True, help="输出 Parquet 文件路径")
    pre_parser.add_argument("--tokenizer_path", type=str, required=True, help="HuggingFace Tokenizer 路径")
    pre_parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    pre_parser.add_argument("--ignore_index", type=int, default=-100)
    
    # --------------------------------------
    # --------------------------------------
    merge_parser = subparsers.add_parser('merge', help='合并已有的分片')
    merge_parser.add_argument("--temp_dir", type=str, required=True, help="包含 shard_xxx.parquet 的临时目录")
    merge_parser.add_argument("--output_path", type=str, required=True, help="最终输出的 Parquet 文件路径")
    merge_parser.add_argument("--keep_shards", action="store_true", help="是否保留临时分片文件 (默认删除)")
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        process_jsonl_to_packed_parquet(
            jsonl_path=args.jsonl_path,
            output_parquet_path=args.output_path,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len,
            ignore_index=args.ignore_index
        )
        
    elif args.command == 'merge':

        merge_shards_stream(
            temp_dir=args.temp_dir,
            output_path=args.output_path,
            keep_shards=args.keep_shards
        )