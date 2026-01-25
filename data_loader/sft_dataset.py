from torch.utils.data import Dataset
import torch
import os
import pickle
import hashlib
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_CACHE_DIR = (PROJECT_ROOT / ".cache" / "sft_dataset").as_posix()
PACKED_CACHE_DIR = (PROJECT_ROOT / ".cache" / "sft_packed").as_posix()

class SFTDataset(Dataset):
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
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

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

class SFTDatasetPacked(SFTDataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, cache_dir=SFT_CACHE_DIR, use_cache=True):
        super().__init__(jsonl_path, tokenizer, max_length, cache_dir)
        
        self.cache_dir = PACKED_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # 生成缓存 Key
        cache_key = self._generate_cache_key(jsonl_path, max_length)
        self.cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        # 2. 尝试加载缓存（仅 use_cache 且文件存在时）
        if use_cache:
            if os.path.exists(self.cache_path):
                print(f"[SFTDatasetPacked] 发现缓存文件: {os.path.basename(self.cache_path)}")
                try:
                    with open(self.cache_path, 'rb') as f:
                        data = pickle.load(f)
                    if self._validate_cached_format(data):
                        self.packed_samples = data
                        print(f"[SFTDatasetPacked] 加载缓存: {os.path.basename(self.cache_path)}")
                        print(f"[SFTDatasetPacked] 加载成功: {len(self.packed_samples)} 个打包序列")
                        return
                    print("[SFTDatasetPacked] 缓存格式过旧或无效，将重新打包")
                    try:
                        os.remove(self.cache_path)
                    except OSError:
                        pass
                except Exception as e:
                    print(f"[SFTDatasetPacked] 缓存损坏，重新打包: {e}")
                    try:
                        os.remove(self.cache_path)
                    except OSError:
                        pass
            else:
                print(f"[SFTDatasetPacked] 缓存文件不存在: {os.path.basename(self.cache_path)}，将重新打包")

        # 3. 无缓存或加载失败，执行打包
        self._pack_samples()

        # 4. 保存缓存
        if use_cache:
            self._save_cache()

    def _generate_cache_key(self, jsonl_path, max_length):
        jsonl_path = os.path.abspath(jsonl_path)
        stat = os.stat(jsonl_path)
        # 添加 version 后缀以防逻辑变更导致缓存不兼容
        # v6: 延迟生成 attention_mask_2d（不保存在缓存中，节省内存）
        cache_str = f"{jsonl_path}_{stat.st_size}_{stat.st_mtime}_{max_length}_v6"
        return hashlib.md5(cache_str.encode()).hexdigest()[:16]

    def _validate_cached_format(self, data):
        """检查缓存是否为有效格式（含 boundaries，attention_mask_2d 可选）"""
        if not data or not isinstance(data, list):
            return False
        s = data[0]
        if not isinstance(s, dict):
            return False
        # boundaries 必需，attention_mask_2d 可选（新格式延迟生成）
        return "boundaries" in s

    def _create_2d_attention_mask(self, input_ids, boundaries):
        """
        创建 2D attention mask（向量化实现），确保：
        1. 不同样本之间不能互相 attend
        2. 每个样本内部使用 causal mask（只能看到前面的 tokens）
        3. Padding 位置不能 attend 也不能被 attend
        Args:
            input_ids: 打包后的 input_ids 列表
            boundaries: 样本边界位置列表，例如 [0, 512, 1067, 1835] 表示3个样本
        
        Returns:
            2D attention mask: shape (seq_len, seq_len)，1 表示可以 attend，0 表示不能
        """
        seq_len = len(input_ids)
        pad_id = self.tokenizer.pad_token_id
        # 向量化：避免 list comprehension + 单次 tensor 构造
        ids = torch.tensor(input_ids, dtype=torch.long) if isinstance(input_ids, list) else input_ids
        is_padding = (ids == pad_id)  # (seq_len,)

        # 每个位置所属的 segment（0, 1, ..., num_segments-1）
        segment_id = torch.zeros(seq_len, dtype=torch.long)
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            segment_id[s:e] = i

        # (seq_len, 1) vs (1, seq_len) -> 同段才可 attend
        same_segment = (segment_id.unsqueeze(1) == segment_id.unsqueeze(0))
        # Causal: k <= q
        q = torch.arange(seq_len, dtype=torch.long)
        causal = (q.unsqueeze(0).T >= q.unsqueeze(0))
        # 非 padding 的 q 可 attend，非 padding 的 k 可被 attend
        valid = (~is_padding).unsqueeze(1) & (~is_padding).unsqueeze(0)

        mask = (same_segment & causal & valid).long()
        return mask

    def _pack_samples(self):
        """
        优化后的打包逻辑：
        1. 批量 tokenize（显著加速）
        2. 流式打包 + 向量化 2D mask
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kw):
                return x

        print("[SFTDatasetPacked] 开始打包...")
        tokenize_batch_size = 128
        pad_id = self.tokenizer.pad_token_id

        # 流式处理：tokenize 一批后立即打包，避免内存爆炸
        self.packed_samples = []
        current_input_ids = []
        current_labels = []
        current_boundaries = [0]

        # 批量 tokenize 并立即打包
        num_batches = (len(self.samples) + tokenize_batch_size - 1) // tokenize_batch_size
        print(f"[SFTDatasetPacked] 样本数: {len(self.samples)}, batch_size: {tokenize_batch_size}, 批次数量: {num_batches}")
        for batch_idx in tqdm(range(num_batches), desc="Processing"):
            start_idx = batch_idx * tokenize_batch_size
            end_idx = min(start_idx + tokenize_batch_size, len(self.samples))
            
            # 批量生成 prompt（使用索引访问）
            batch_prompts = [
                self.create_chat_prompt(self.samples[i]["conversations"])
                for i in range(start_idx, end_idx)
            ]
            
            # 批量 tokenize
            out = self.tokenizer(
                batch_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_tensors=None,
            )
            
            # 立即处理这一批：生成 labels 并打包
            for ids in out["input_ids"]:
                ids = ids[: self.max_length]
                labels = self.generate_labels(ids)
                
                # 流式打包逻辑
                if len(current_input_ids) + len(ids) > self.max_length:
                    pad_len = self.max_length - len(current_input_ids)
                    if pad_len > 0:
                        current_input_ids += [pad_id] * pad_len
                        current_labels += [-100] * pad_len
                    # 延迟生成 attention_mask_2d，只在 __getitem__ 时按需生成
                    self.packed_samples.append({
                        "input_ids": current_input_ids,
                        "labels": current_labels,
                        "boundaries": current_boundaries,
                    })
                    current_input_ids = list(ids)
                    current_labels = list(labels)
                    current_boundaries = [0, len(ids)]
                else:
                    current_input_ids.extend(ids)
                    current_labels.extend(labels)
                    current_boundaries.append(len(current_input_ids))

        if current_input_ids:
            pad_len = self.max_length - len(current_input_ids)
            if pad_len > 0:
                current_input_ids += [pad_id] * pad_len
                current_labels += [-100] * pad_len
            # 延迟生成 attention_mask_2d
            self.packed_samples.append({
                "input_ids": current_input_ids,
                "labels": current_labels,
                "boundaries": current_boundaries,
            })

        print(f"[SFTDatasetPacked] 打包完成: {len(self.samples)} -> {len(self.packed_samples)} 序列")

    def _save_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.packed_samples, f)
            print(f"[SFTDatasetPacked] 缓存已保存: {os.path.basename(self.cache_path)} ({len(self.packed_samples)} 个序列)")
        except Exception as e:
            print(f"[SFTDatasetPacked] 缓存保存失败: {e}")

    def __len__(self):
        """返回打包后的序列数量，而不是原始样本数量"""
        if not hasattr(self, 'packed_samples'):
            raise RuntimeError("packed_samples 未初始化，请检查 __init__ 方法")
        return len(self.packed_samples)

    def _create_position_ids(self, boundaries, seq_len):
        """
        为打包数据创建正确的 position_ids。
        每个样本应该从位置 0 开始，而不是使用序列中的绝对位置。
        
        Args:
            boundaries: 样本边界位置列表，例如 [0, 512, 1067, 1835]
            seq_len: 序列总长度
        
        Returns:
            position_ids: shape (seq_len,)，每个位置的值表示该 token 在其样本内的相对位置
        """
        position_ids = torch.zeros(seq_len, dtype=torch.long)
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            # 每个样本从位置 0 开始
            position_ids[start:end] = torch.arange(0, end - start, dtype=torch.long)
        
        return position_ids

    def __getitem__(self, index):
        # 直接获取预处理好的数据
        data = self.packed_samples[index]
        input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        
        # 获取必需的字段
        boundaries = data['boundaries']
        
        # 按需生成 attention_mask_2d（延迟生成，节省内存）
        if 'attention_mask_2d' in data:
            # 兼容旧缓存格式（如果存在）
            attention_mask_2d = data['attention_mask_2d']
            if not isinstance(attention_mask_2d, torch.Tensor):
                attention_mask_2d = torch.tensor(attention_mask_2d, dtype=torch.long)
        else:
            # 新格式：按需生成
            attention_mask_2d = self._create_2d_attention_mask(data['input_ids'], boundaries)
        
        # 生成正确的 position_ids：每个样本从位置 0 开始
        position_ids = self._create_position_ids(boundaries, len(input_ids))
        
        # 返回 2D attention mask 和 position_ids
        return input_ids, labels, attention_mask_2d, boundaries, position_ids


def collate_fn_packed(batch):
    """
    自定义 collate_fn 用于处理打包数据集
    处理 5 个返回值：input_ids, labels, attention_mask_2d, boundaries, position_ids
    
    Args:
        batch: 一个列表，每个元素是一个元组 (input_ids, labels, attention_mask_2d, boundaries, position_ids)
    
    Returns:
        input_ids: (batch_size, seq_len) 堆叠后的 input_ids
        labels: (batch_size, seq_len) 堆叠后的 labels
        attention_mask: (batch_size, seq_len, seq_len) 堆叠后的 2D attention mask
        boundaries: List[List[int]] 保持为列表，因为不同样本的边界数量可能不同
        position_ids: (batch_size, seq_len) 堆叠后的 position_ids
    """
    import torch
    
    # batch 是一个列表，每个元素是一个元组 (input_ids, labels, attention_mask_2d, boundaries, position_ids)
    input_ids_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    attention_mask_list = [item[2] for item in batch]
    boundaries_list = [item[3] for item in batch]  # 保持为列表，不堆叠
    position_ids_list = [item[4] for item in batch]
    
    # 堆叠张量
    input_ids = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    
    # boundaries 保持为列表（因为不同样本的边界数量可能不同）
    return input_ids, labels, attention_mask, boundaries_list, position_ids