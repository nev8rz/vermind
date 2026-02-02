import torch
import torch.nn as nn
import os
import json
from typing import Optional, List


SAFETENSORS_AVAILABLE = False
save_file = None
load_file = None

try:
    from safetensors.torch import save_file, load_file  # type: ignore
    SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank * 2
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)

        self.B.weight.data.zero_()

    def forward(self, x):


        scaling = self.alpha / self.rank if self.rank > 0 else 1.0
        return scaling * self.B(self.A(x))


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


def apply_lora(model, rank=16, alpha=None, target_modules=None):
    """
    将LoRA应用到模型的指定层
    
    Args:
        model: VerMindForCausalLM 模型
        rank: LoRA的秩
        alpha: LoRA的alpha参数（缩放因子），如果为None则使用rank*2
        target_modules: 要应用LoRA的模块名称列表，如果为None，则自动应用到所有方阵Linear层
    """

    device = next(model.parameters()).device
    

    if alpha is None:
        alpha = rank * 2
    

    modules_to_modify = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):

            if target_modules is not None:

                if not any(target in name for target in target_modules):
                    continue
            

            if target_modules is not None or module.weight.shape[0] == module.weight.shape[1]:
                modules_to_modify.append((name, module))
    

    applied_count = 0
    for name, module in modules_to_modify:
        lora = LoRA(module.in_features, module.out_features, rank=rank, alpha=alpha).to(device)
        setattr(module, "lora", lora)
        original_forward = module.forward



        def make_forward_with_lora(orig_forward, lora_module):
            def forward_with_lora(x):
                return orig_forward(x) + lora_module(x)
            return forward_with_lora

        module.forward = make_forward_with_lora(original_forward, lora)
        applied_count += 1
    
    print(f"[LoRA] Applied LoRA to {applied_count} modules (rank={rank}, alpha={alpha})")


def load_lora_config(lora_path: str):
    """
    加载 LoRA 配置
    
    Args:
        lora_path: LoRA 路径（目录或文件路径）
    
    Returns:
        dict: LoRA 配置，包含 lora_rank, lora_alpha, target_modules
    """

    if os.path.isdir(lora_path):
        config_file = os.path.join(lora_path, "adapter_config.json")
    else:

        config_file = os.path.join(os.path.dirname(lora_path), "adapter_config.json")
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    return None


def load_lora(model, lora_path: str):
    """
    加载LoRA权重（支持从 checkpoint 目录或文件路径加载）
    
    Args:
        model: VerMindForCausalLM 模型（已应用LoRA）
        lora_path: LoRA权重路径，可以是：
                   - checkpoint 目录路径（如 base_path/checkpoint_1000/）
                   - LoRA 文件路径（如 adapter_model.safetensors 或 adapter_model.bin）
                   - 旧的 .pth 或 .safetensors 文件路径（向后兼容）
    """
    from torch.nn.parallel import DistributedDataParallel
    

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    

    if os.path.isdir(lora_path):

        safetensors_file = os.path.join(lora_path, "adapter_model.safetensors")
        bin_file = os.path.join(lora_path, "adapter_model.bin")
        
        if os.path.exists(safetensors_file):
            lora_file = safetensors_file
        elif os.path.exists(bin_file):
            lora_file = bin_file
        else:

            import glob
            safetensors_files = glob.glob(os.path.join(lora_path, "*.safetensors"))
            pth_files = glob.glob(os.path.join(lora_path, "*.pth"))
            
            if safetensors_files:
                lora_file = safetensors_files[0]
            elif pth_files:
                lora_file = pth_files[0]
            else:
                raise FileNotFoundError(f"No LoRA weight file found in directory: {lora_path}")
    else:

        lora_file = lora_path
    

    device = next(raw_model.parameters()).device
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device
    

    if lora_file.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors library is not installed. Install it with: pip install safetensors")

        if device_str.startswith('cuda:'):

            state_dict = load_file(lora_file, device='cpu')

            state_dict = {k: v.to(device) for k, v in state_dict.items()}
        else:
            state_dict = load_file(lora_file, device=device_str)
        format_type = "safetensors"
    else:

        state_dict = torch.load(lora_file, map_location=device)
        format_type = "pth"
    

    loaded_count = 0
    failed_count = 0
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):

            prefix = f'{name}.lora.'
            lora_state = {k.replace(prefix, ''): v for k, v in state_dict.items() if prefix in k}
            if lora_state:
                try:

                    missing_keys, unexpected_keys = module.lora.load_state_dict(lora_state, strict=False)
                    if missing_keys:
                        print(f"[LoRA] Warning: {name} missing keys: {missing_keys}")
                        failed_count += 1
                    elif unexpected_keys:
                        print(f"[LoRA] Warning: {name} unexpected keys: {unexpected_keys}")
                    else:
                        loaded_count += 1
                except Exception as e:
                    print(f"[LoRA] Error loading {name}: {e}")
                    failed_count += 1
            else:

                print(f"[LoRA] Warning: No weights found for {name} (looking for prefix: {prefix})")
                failed_count += 1
    
    print(f"[LoRA] Loaded LoRA weights from: {lora_file} (format: {format_type})")
    print(f"[LoRA] Successfully loaded: {loaded_count}, Failed: {failed_count}")


def save_lora(model, base_path: str, checkpoint_num: int = None, use_safetensors: bool = True, 
               lora_rank: int = None, lora_alpha: int = None, target_modules: List[str] = None):
    """
    保存LoRA权重和配置（使用与 save_checkpoint 相同的目录结构）
    
    Args:
        model: VerMindForCausalLM 模型（已应用LoRA）
        base_path: 基础保存路径（目录）
        checkpoint_num: checkpoint 编号，如果为 None，则保存到 base_path 目录
        use_safetensors: 是否使用 safetensors 格式（默认 True）
        lora_rank: LoRA 的秩（用于保存配置）
        lora_alpha: LoRA 的 alpha 参数（用于保存配置）
        target_modules: LoRA 目标模块列表（用于保存配置）
    """
    from torch.nn.parallel import DistributedDataParallel
    

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    

    if checkpoint_num is not None:

        actual_save_path = os.path.join(base_path, f"checkpoint_{checkpoint_num}")
    else:

        actual_save_path = base_path
    
    os.makedirs(actual_save_path, exist_ok=True)
    

    state_dict = {}
    lora_modules = []
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            lora_modules.append(name)
            for k, v in module.lora.state_dict().items():
                state_dict[f"{name}.lora.{k}"] = v
    

    if lora_rank is None and lora_modules:

        first_lora_name = lora_modules[0]
        for name, module in raw_model.named_modules():
            if name == first_lora_name and hasattr(module, 'lora'):
                lora_rank = module.lora.rank
                break
    
    if target_modules is None and lora_modules:
        target_modules = []
        for name in lora_modules:
            parts = name.split('.')
            for part in parts:
                if any(target in part for target in ['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj']):
                    if part not in target_modules:
                        target_modules.append(part)
    
    if lora_alpha is None and lora_rank is not None:

        lora_alpha = lora_rank * 2
    

    adapter_config = {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules if target_modules else [],
        "task_type": "CAUSAL_LM"
    }
    config_file = os.path.join(actual_save_path, "adapter_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    

    if use_safetensors:
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors library is not installed. Install it with: pip install safetensors")
        lora_file = os.path.join(actual_save_path, "adapter_model.safetensors")
        save_file(state_dict, lora_file)
        format_type = "safetensors"
    else:
        lora_file = os.path.join(actual_save_path, "adapter_model.bin")
        torch.save(state_dict, lora_file)
        format_type = "pth"
    
    print(f"[LoRA] Saved {len(state_dict)} params to: {lora_file} (format: {format_type})")
    print(f"[LoRA] Saved config to: {config_file}")
    return lora_file
