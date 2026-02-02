import os
import sys
import json
import torch
import io
from PIL import Image
from torch.utils.data import Dataset
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vermind_models import VerMindVLM, VLMConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLMDataset(Dataset):
    """
    Vision-Language Model Dataset
    从 Parquet 文件加载图文对数据，用于 VerMind-V 训练
    """
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        vision_encoder_path: str = "google/siglip-base-patch16-224",
        max_length: int = 512,
        image_special_token: str = '<image>',
    ):
        """
        Args:
            parquet_path: Parquet 文件路径，需包含 'image_bytes' 和 'conversations' 列
            tokenizer: VerMind tokenizer
            vision_encoder_path: SigLIP 模型路径
            max_length: 最大序列长度
            image_special_token: 图像占位符，会被替换为 image_ids
        """
        super().__init__()
        self.table = pq.read_table(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_special_token = image_special_token
        

        vlm_config = VLMConfig()
        self.image_ids = vlm_config.image_ids
        

        _, self.processor = VerMindVLM.get_vision_model(vision_encoder_path)
        if self.processor is None:
            raise ValueError(f"无法加载视觉模型: {vision_encoder_path}")
        

        self.bos_str = f'{tokenizer.bos_token}assistant\n'
        self.eos_str = f'{tokenizer.eos_token}\n'
        self.bos_id = tokenizer(self.bos_str, add_special_tokens=False).input_ids
        self.eos_id = tokenizer(self.eos_str, add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def create_chat_prompt(self, conversations: list) -> str:
        """
        将对话转换为 prompt，替换 <image> 为特殊 token
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            content = turn['content'].replace('<image>', self.image_special_token)
            messages.append({"role": role, "content": content})
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def insert_image_tokens(self, input_ids: list) -> list:
        """
        将 <image> 的 tokenizer 输出替换为 image_ids
        """

        image_token_ids = self.tokenizer(self.image_special_token, add_special_tokens=False).input_ids
        
        new_input_ids = []
        i = 0
        while i < len(input_ids):

            if input_ids[i:i+len(image_token_ids)] == image_token_ids:

                new_input_ids.extend(self.image_ids)
                i += len(image_token_ids)
            else:
                new_input_ids.append(input_ids[i])
                i += 1
        
        return new_input_ids

    def generate_labels(self, input_ids: list) -> list:
        """
        只计算 assistant 回复部分的 loss
        视觉 token 是输入特征，不应参与 next token prediction
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                found_eos = False
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        found_eos = True
                        break
                    end += 1
                
                if found_eos:

                    for j in range(i, min(end + len(self.eos_id), len(input_ids))):
                        labels[j] = input_ids[j]
                    i = end + len(self.eos_id)
                else:

                    for j in range(i, min(len(input_ids), self.max_length)):
                        labels[j] = input_ids[j]
                    i = len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int):

        conversations = json.loads(self.table['conversations'][index].as_py())
        image_bytes = self.table['image_bytes'][index].as_py()
        

        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids
        

        input_ids = self.insert_image_tokens(input_ids)
        

        input_ids = input_ids[:self.max_length]
        

        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        

        labels = self.generate_labels(input_ids)
        

        image = Image.open(io.BytesIO(image_bytes))
        pixel_values = VerMindVLM.image2tensor(image, self.processor)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values.squeeze(0),
        }


def vlm_collate_fn(batch):
    """
    VLM DataLoader 的 collate 函数
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
    }
