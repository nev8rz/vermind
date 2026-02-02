#!/usr/bin/env python3
"""
合并 LoRA 适配器到基础模型
将 LoRA 权重合并到基础模型权重中，生成一个完整的模型
"""

import os
import sys
import argparse
import torch
import torch.nn as nn


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.utils import load_checkpoint, Logger
from vermind_models.lora_adpater import apply_lora, load_lora, load_lora_config


def merge_lora_weights(model, lora_path: str):
    """
    合并 LoRA 权重到基础模型
    
    Args:
        model: VerMindForCausalLM 模型（已应用 LoRA 并加载了权重）
        lora_path: LoRA 路径（用于日志）
    
    Returns:
        list: 合并的模块名称列表
    """
    Logger(f'Starting LoRA weight merging...')
    
    merged_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and isinstance(module, nn.Linear):

            original_weight = module.weight.data.clone()
            lora_A = module.lora.A.weight.data  # (in_features, rank)
            lora_B = module.lora.B.weight.data  # (rank, out_features)
            

            lora_alpha = module.lora.alpha
            lora_rank = module.lora.rank
            scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
            


            lora_delta = torch.matmul(lora_B, lora_A)  # (out_features, in_features)
            merged_weight = original_weight + scaling * lora_delta
            

            assert merged_weight.shape == original_weight.shape, \
                f"Weight shape mismatch for {name}: {merged_weight.shape} vs {original_weight.shape}"
            

            weight_changed = not torch.allclose(original_weight, merged_weight, rtol=1e-5, atol=1e-8)
            assert weight_changed, \
                f"Weight for {name} did not change after merging! LoRA might be zero or not loaded correctly."
            

            module.weight.data = merged_weight
            



            delattr(module, 'lora')
            

            import types
            module.forward = types.MethodType(nn.Linear.forward, module)
            
            merged_modules.append(name)
            Logger(f'Merged LoRA weights for: {name}')
    
    Logger(f'Successfully merged {len(merged_modules)} LoRA adapters')
    return merged_modules


def infer_lora_config(lora_path: str):
    """
    从 LoRA 权重文件中推断 rank 和 target_modules
    
    Args:
        lora_path: LoRA 路径（目录或文件）
    
    Returns:
        rank: LoRA 的秩（如果无法推断则返回 None）
        target_modules: 目标模块列表（如果无法推断则返回 None）
    """

    try:
        if lora_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(lora_path)
        elif lora_path.endswith('.bin') or lora_path.endswith('.pth'):
            state_dict = torch.load(lora_path, map_location='cpu')
        elif os.path.isdir(lora_path):

            safetensors_file = os.path.join(lora_path, "adapter_model.safetensors")
            bin_file = os.path.join(lora_path, "adapter_model.bin")
            if os.path.exists(safetensors_file):
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_file)
            elif os.path.exists(bin_file):
                state_dict = torch.load(bin_file, map_location='cpu')
            else:
                return None, None
        else:
            return None, None
        


        rank = None
        target_modules = set()
        
        for key in state_dict.keys():
            if '.lora.A.weight' in key:

                rank = state_dict[key].shape[1]

                module_name = key.split('.lora.A.weight')[0]

                parts = module_name.split('.')
                for part in parts:
                    if any(target in part for target in ['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj']):
                        target_modules.add(part)
                break
        
        target_modules = list(target_modules) if target_modules else None
        return rank, target_modules
    
    except Exception as e:
        Logger(f'Warning: Could not infer LoRA config from {lora_path}: {e}')
        return None, None


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器到基础模型")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="基础模型路径（可以是基础路径如 /root/vermind/output/sft/full_sft_768，会自动找最新的 checkpoint）"
    )
    parser.add_argument(
        '--lora_path',
        type=str,
        required=True,
        help="LoRA 适配器路径（可以是基础路径如 /root/vermind/output/lora/lora_768，会自动找最新的 checkpoint）"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="合并后模型的保存路径（默认：lora_path + '_merged'）"
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=None,
        help="LoRA 的秩（如果未指定，将尝试从权重中推断）"
    )
    parser.add_argument(
        '--lora_target_modules',
        type=str,
        default=None,
        help="LoRA 目标模块，用逗号分隔（如果未指定，将尝试从权重中推断）"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="运行设备"
    )
    args = parser.parse_args()
    

    import glob
    model_path = args.model_path
    if os.path.isdir(model_path):
        checkpoint_pattern = os.path.join(model_path, "checkpoint_*")
        checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
            latest_checkpoint = checkpoints[-1]
            Logger(f'找到 {len(checkpoints)} 个模型 checkpoint，使用最新的: {os.path.basename(latest_checkpoint)}')
            model_path = latest_checkpoint
    

    lora_path = args.lora_path
    if os.path.isdir(lora_path):
        checkpoint_pattern = os.path.join(lora_path, "checkpoint_*")
        checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
            latest_checkpoint = checkpoints[-1]
            Logger(f'找到 {len(checkpoints)} 个 LoRA checkpoint，使用最新的: {os.path.basename(latest_checkpoint)}')
            lora_path = latest_checkpoint
    

    if not os.path.exists(model_path):
        Logger(f'❌ 错误: 模型路径不存在: {model_path}')
        Logger(f'   请检查路径是否正确')
        sys.exit(1)
    
    if not os.path.exists(lora_path):
        Logger(f'❌ 错误: LoRA 路径不存在: {lora_path}')
        Logger(f'   请检查路径是否正确')
        sys.exit(1)
    

    if args.output_path is None:
        if os.path.isdir(lora_path):
            args.output_path = lora_path.rstrip('/') + '_merged'
        else:
            args.output_path = os.path.splitext(lora_path)[0] + '_merged'
    
    Logger(f'Model path: {model_path}')
    Logger(f'LoRA path: {lora_path}')
    Logger(f'Output path: {args.output_path}')
    

    Logger(f'Loading base model from {model_path}...')
    model, tokenizer, _ = load_checkpoint(model_path, device=args.device, load_training_state=False)
    Logger(f'Base model loaded')
    

    Logger(f'Saving pre-merge weights for verification...')
    pre_merge_weights = {}
    for name, param in model.named_parameters():
        pre_merge_weights[name] = param.data.clone()
    Logger(f'Saved {len(pre_merge_weights)} parameter tensors')
    

    lora_config = load_lora_config(lora_path)
    
    if lora_config:

        lora_rank = lora_config.get('lora_rank')
        lora_alpha = lora_config.get('lora_alpha')
        lora_target_modules = lora_config.get('target_modules', [])
        Logger(f'Loaded LoRA config from adapter_config.json: rank={lora_rank}, alpha={lora_alpha}, target_modules={lora_target_modules}')
    else:

        Logger(f'No adapter_config.json found, trying to infer or use provided parameters...')
        inferred_rank, inferred_target_modules = infer_lora_config(lora_path)
        
        lora_rank = args.lora_rank if args.lora_rank is not None else inferred_rank
        if lora_rank is None:
            raise ValueError("Could not determine LoRA rank. Please specify --lora_rank or ensure adapter_config.json exists")
        
        if args.lora_target_modules:
            lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        elif inferred_target_modules:
            lora_target_modules = inferred_target_modules
        else:

            lora_target_modules = ['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            Logger(f'Warning: Could not infer target_modules, using default: {lora_target_modules}')
        
        Logger(f'LoRA config: rank={lora_rank}, target_modules={lora_target_modules}')
    

    Logger(f'Applying LoRA structure to model...')
    lora_alpha = lora_config.get('lora_alpha') if lora_config else None
    apply_lora(model, rank=lora_rank, alpha=lora_alpha, target_modules=lora_target_modules)
    Logger(f'LoRA structure applied (rank={lora_rank}, alpha={lora_alpha})')
    

    Logger(f'Loading LoRA weights from {lora_path}...')
    load_lora(model, lora_path)
    Logger(f'LoRA weights loaded')
    

    Logger(f'Merging LoRA weights into base model...')
    merged_modules = merge_lora_weights(model, lora_path)
    Logger(f'LoRA weights merged')
    

    Logger(f'Verifying weight changes after merging...')
    changed_count = 0
    unchanged_count = 0
    changed_modules = []
    unchanged_modules = []
    
    for name, param in model.named_parameters():
        if name in pre_merge_weights:
            pre_weight = pre_merge_weights[name]
            post_weight = param.data
            

            if torch.allclose(pre_weight, post_weight, rtol=1e-5, atol=1e-8):
                unchanged_count += 1
                unchanged_modules.append(name)
            else:
                changed_count += 1
                changed_modules.append(name)

                diff = torch.abs(post_weight - pre_weight)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                Logger(f'  ✓ {name}: changed (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})')
    
    Logger(f'Weight verification results:')
    Logger(f'  - Changed: {changed_count} parameters')
    Logger(f'  - Unchanged: {unchanged_count} parameters')
    

    assert changed_count > 0, f"ERROR: No weights changed after merging! Expected at least {merged_modules} modules to change."
    Logger(f'✅ Assertion passed: {changed_count} parameters changed as expected')
    

    if lora_config and lora_config.get('target_modules'):
        target_modules = lora_config.get('target_modules', [])
        changed_targets = [m for m in changed_modules if any(target in m for target in target_modules)]
        Logger(f'  - Changed target modules: {len(changed_targets)}/{len(target_modules)}')
        if len(changed_targets) == 0:
            Logger(f'⚠️  Warning: No target modules changed! This might indicate a problem.')
    

    Logger(f'Saving merged model to {args.output_path}...')
    os.makedirs(args.output_path, exist_ok=True)
    

    embed_weight = model.model.embed_tokens.weight
    lm_head_weight = model.lm_head.weight
    is_tied = embed_weight.data_ptr() == lm_head_weight.data_ptr()
    
    if is_tied:

        model.model.embed_tokens.weight = nn.Parameter(embed_weight.clone())
    
    try:

        model.save_pretrained(args.output_path, safe_serialization=True)
        

        if tokenizer is not None:
            tokenizer.save_pretrained(args.output_path)
    finally:

        if is_tied:
            model.model.embed_tokens.weight = lm_head_weight
    
    Logger(f'Merged model saved to {args.output_path}')
    Logger(f'Done!')


if __name__ == "__main__":
    main()
