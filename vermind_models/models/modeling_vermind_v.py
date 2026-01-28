# ==================== VLM (Vision Language Model) 相关代码 ====================
# 该文件包含基于 SigLIP 的视觉语言模型扩展

import os
import torch
import warnings
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipModel, SiglipProcessor
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

warnings.filterwarnings('ignore')

# 导入基础模型和 VLM 配置
from .modeling_vermind import VerMindForCausalLM
from ..config import VerMindConfig, VLMConfig


class VisionProj(nn.Module):
    """视觉投影层：将 SigLIP 的输出映射到 LLM 的隐藏空间，并进行 token 压缩"""
    
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        
        # 1. 升级为 MLP (Linear -> GELU -> Linear)
        # 增加非线性特征变换，帮助对齐 SigLIP 和 LLM 的空间
        self.mlp = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        # image_encoders shape: [Batch, 196, 768] (来自 SigLIP-224)
        
        # 1. MLP 映射
        x = self.mlp(image_encoders)  # [Batch, 196, hidden_size]
        
        # 2. Token 压缩 (2x2 Average Pooling)
        # 这一步对于 Vermind 这种小参数模型至关重要！
        # 将 196 个 token 压缩为 49 个，减少 75% 的序列长度，大幅降低计算压力
        B, L, C = x.shape
        # SigLIP 224x224 -> 14x14 patches
        H_dim = int(L**0.5) # 14
        
        # Reshape 为图像格式 [B, C, 14, 14] 以便做 2d pooling
        x = x.view(B, H_dim, H_dim, C).permute(0, 3, 1, 2)
        
        # Pooling: 14x14 -> 7x7
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        
        # 变回序列格式 [B, 49, C]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class VerMindVLM(VerMindForCausalLM):
    """基于 SigLIP 的视觉语言模型"""
    
    config_class = VLMConfig

    def __init__(self, params: VLMConfig = None, vision_model_path="google/siglip-base-patch16-224"):
        super().__init__(params)
        if not params: params = VLMConfig()
        self.params = params
        
        # 加载 SigLIP Vision Encoder
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        
        # 初始化新的 Projection 层
        self.vision_proj = VisionProj(ve_hidden_size=768, hidden_size=params.hidden_size)

    @staticmethod
    def get_vision_model(model_path: str):
        """加载 SigLIP 视觉模型"""
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        
        # 兼容本地路径和 HuggingFace Hub 路径
        if not os.path.exists(model_path) and not "/" in model_path:
             return None, None
              
        print(f"[VerMind-V] Loading Vision Encoder: {model_path}...")
        try:
            # 使用 SiglipModel 加载
            model = SiglipModel.from_pretrained(model_path)
            processor = SiglipProcessor.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading SigLIP: {e}")
            return None, None

        # 冻结 vision_encoder 的所有参数
        for param in model.parameters():
            param.requires_grad = False
            
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        """将 PIL 图像转换为模型输入张量"""
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        # SiglipProcessor 会自动处理归一化和 Resize (224x224)
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        """提取图像特征"""
        # image_tensors shape: [Batch, 3, 224, 224]
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        
        # SigLIP 直接输出 [Batch, 196, 768]，没有 CLS token，不需要切片
        img_embedding = outputs.last_hidden_state  
        return img_embedding

    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        """将视觉特征注入到文本嵌入中"""
        # 这里的逻辑通过 image_ids (长度49) 来查找插入位置
        def find_indices(tokens, image_ids):
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in
                            matches[batch_idx].nonzero(as_tuple=True)[0]]
                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()
            } or None

        image_indices = find_indices(tokens, self.params.image_ids)
        
        if vision_tensors is not None and image_indices:
            # vision_proj 输出形状现在是 [Batch, 49, Hidden]
            vision_proj = self.vision_proj(vision_tensors)
            
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
                
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        # 取出对应的图像特征
                        current_vision_embeds = vision_proj[i] if vision_proj.shape[0] > 1 else vision_proj[0]
                        
                        # 简单的单图注入逻辑
                        if img_idx < 1: 
                             h_i = torch.cat((h_i[:start_idx], current_vision_embeds, h_i[end_idx + 1:]), dim=0)[:seqlen]
                        img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        """前向传播，支持图像输入"""
        
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 仅在首个 token 位置且有图片输入时，注入视觉特征
        if pixel_values is not None and start_pos == 0:
            if len(pixel_values.shape) == 5: 
                pixel_values = pixel_values[:, 0, :, :, :] 
            
            # 1. 提取特征 (SigLIP)
            vision_tensors = VerMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            
            # 2. 投影 + Pooling + 替换文本 Embedding
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, vision_tensors=vision_tensors,
                                                   seqlen=input_ids.shape[1])

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = 0
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
