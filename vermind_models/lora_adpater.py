import torch
import torch.nn as nn
import os
from typing import Optional, List

# 定义LoRA网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


class LoRAConfig:
    """LoRA配置类"""
    def __init__(
        self,
        rank: int = 8,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Args:
            rank: LoRA的秩，控制低秩矩阵的大小
            target_modules: 要应用LoRA的模块名称列表，如 ['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                           如果为None，则自动应用到所有方阵Linear层
        """
        self.rank = rank
        self.target_modules = target_modules


def apply_lora(model, rank=16, target_modules=None):
    """
    将LoRA应用到模型的指定层
    
    Args:
        model: VerMindForCausalLM 模型
        rank: LoRA的秩
        target_modules: 要应用LoRA的模块名称列表，如果为None，则自动应用到所有方阵Linear层
    """
    # 获取模型设备
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 如果指定了target_modules，只对匹配的模块应用LoRA
            if target_modules is not None:
                # 检查模块名是否包含target_modules中的任何一个
                if not any(target in name for target in target_modules):
                    continue
            
            # 如果是方阵Linear层，或者指定了target_modules，则应用LoRA
            if target_modules is not None or module.weight.shape[0] == module.weight.shape[1]:
                lora = LoRA(module.in_features, module.out_features, rank=rank).to(device)
                setattr(module, "lora", lora)  # 给 module 加一个 lora 成员变量
                original_forward = module.forward  # 保存原始 forward 方法

                # 构造新 forward：原始输出 + LoRA 输出
                def forward_with_lora(x, layer1=original_forward, layer2=lora):
                    return layer1(x) + layer2(x)

                module.forward = forward_with_lora  # 替换 forward 方法


def load_lora(model, lora_path: str):
    """
    加载LoRA权重
    
    Args:
        model: VerMindForCausalLM 模型（已应用LoRA）
        lora_path: LoRA权重文件路径
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weight file not found: {lora_path}")
    
    state_dict = torch.load(lora_path, map_location=next(model.parameters()).device)
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # replace用于隐掉{name}.lora.，因为load的执行者是module.lora.，不去掉会重复
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            if lora_state:
                module.lora.load_state_dict(lora_state)
    
    print(f"[LoRA] Loaded LoRA weights from: {lora_path}")


def save_lora(model, save_path: str):
    """
    保存LoRA权重
    
    Args:
        model: VerMindForCausalLM 模型（已应用LoRA）
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            for k, v in module.lora.state_dict().items():
                state_dict[f"{name}.lora.{k}"] = v
    
    torch.save(state_dict, save_path)
    print(f"[LoRA] Saved {len(state_dict)} params to: {save_path}")
