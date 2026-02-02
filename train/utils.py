
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(step, total_steps, lr, warmup_ratio=0.0):
    warmup_steps = int(total_steps * warmup_ratio)
    if warmup_steps > 0 and step < warmup_steps:
        return lr * step / warmup_steps
    return lr * (0.05 + 0.475 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_save_path(original_save_path):
    """
    确定基础保存路径（在训练开始时调用一次）
    如果原始路径已存在且有 checkpoint，则创建新的编号路径
    
    Args:
        original_save_path: 原始保存路径，如 pretrain_768
    
    Returns:
        确定的基础保存路径
    """
    import glob
    
    base_dir = os.path.dirname(original_save_path)
    base_name = os.path.basename(original_save_path)
    

    base_path = original_save_path
    if os.path.exists(original_save_path):

        pattern = os.path.join(original_save_path, "checkpoint_*")
        has_checkpoints = len([p for p in glob.glob(pattern) if os.path.isdir(p)]) > 0
        
        if has_checkpoints:


            base_pattern = os.path.join(base_dir, f"{base_name}_*")
            numbered_bases = []
            for path in glob.glob(base_pattern):
                if os.path.isdir(path):
                    path_name = os.path.basename(path)
                    if path_name.startswith(f"{base_name}_"):
                        try:
                            num = int(path_name.replace(f"{base_name}_", ""))
                            numbered_bases.append(num)
                        except ValueError:
                            pass
            

            if numbered_bases:
                next_base_num = max(numbered_bases) + 1
            else:
                next_base_num = 1
            
            base_path = os.path.join(base_dir, f"{base_name}_{next_base_num}")
    

    os.makedirs(base_path, exist_ok=True)
    
    return base_path


def save_checkpoint(model, tokenizer, config, save_path, optimizer=None, scaler=None, epoch=0, step=0, swanlab=None, max_checkpoints=3, save_interval=1000, steps_per_epoch=None, **kwargs):
    """
    保存模型、tokenizer 和配置（使用标准 transformers 格式，safetensors）
    支持多 checkpoint 管理和自动编号
    
    Args:
        model: VerMindForCausalLM 模型
        tokenizer: tokenizer
        config: VerMindConfig 配置
        save_path: 保存路径（目录，基础路径，应该已经在训练开始时确定）
        optimizer: 优化器（可选）
        scaler: 梯度缩放器（可选）
        epoch: 当前 epoch
        step: 当前 step（当前 epoch 内的步数）
        swanlab: swanlab run 对象（可选）
        max_checkpoints: 最大保留的 checkpoint 数量（默认3）
        save_interval: 保存间隔，用于计算 checkpoint 编号
        steps_per_epoch: 每个 epoch 的总步数（用于计算全局步数）
        **kwargs: 其他需要保存的状态（可包含 is_last_step: bool，表示是否是最后一个step）
    """
    from torch.nn.parallel import DistributedDataParallel
    import glob
    import shutil
    

    base_path = save_path
    


    if steps_per_epoch is not None:
        global_step = epoch * steps_per_epoch + step
    else:
        global_step = step
    




    is_last_step = kwargs.get('is_last_step', False)
    if global_step < save_interval:
        checkpoint_num = global_step
    elif global_step % save_interval == 0:

        checkpoint_num = (global_step // save_interval) * save_interval
    else:

        checkpoint_num = global_step
    

    actual_save_path = os.path.join(base_path, f"checkpoint_{checkpoint_num}")
    
    os.makedirs(actual_save_path, exist_ok=True)
    

    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    



    embed_weight = raw_model.model.embed_tokens.weight
    lm_head_weight = raw_model.lm_head.weight
    is_tied = embed_weight.data_ptr() == lm_head_weight.data_ptr()
    
    if is_tied:

        raw_model.model.embed_tokens.weight = nn.Parameter(embed_weight.clone())
    
    try:

        raw_model.save_pretrained(actual_save_path, safe_serialization=True)
        

        if tokenizer is not None:
            tokenizer.save_pretrained(actual_save_path)
    finally:

        if is_tied:
            raw_model.model.embed_tokens.weight = lm_head_weight
    

    training_state = {
        'epoch': epoch,
        'step': step,
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    }
    
    if optimizer is not None:
        training_state['optimizer'] = optimizer.state_dict()
    
    if scaler is not None:
        training_state['scaler'] = scaler.state_dict()
    
    # swanlab id
    swanlab_id = None
    if swanlab:
        if hasattr(swanlab, 'get_run'):
            run = swanlab.get_run()
            swanlab_id = getattr(run, 'id', None) if run else None
        else:
            swanlab_id = getattr(swanlab, 'id', None)
    training_state['swanlab_id'] = swanlab_id
    

    for key, value in kwargs.items():
        if value is not None:
            if hasattr(value, 'state_dict'):
                raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                raw_value = getattr(raw_value, '_orig_mod', raw_value)
                training_state[key] = raw_value.state_dict()
            else:
                training_state[key] = value
    

    training_state_path = os.path.join(actual_save_path, 'training_state.pt')
    training_state_tmp = training_state_path + '.tmp'
    torch.save(training_state, training_state_tmp)
    os.replace(training_state_tmp, training_state_path)
    

    all_checkpoints = []
    pattern = os.path.join(base_path, "checkpoint_*")
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            all_checkpoints.append(path)
    

    all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    

    removed_count = 0
    if len(all_checkpoints) > max_checkpoints:
        for old_cp in all_checkpoints[max_checkpoints:]:
            Logger(f'Removing old checkpoint: {os.path.basename(old_cp)}')
            shutil.rmtree(old_cp)
            removed_count += 1
    
    Logger(f'Model saved to {os.path.basename(base_path)}/{os.path.basename(actual_save_path)} (transformers format, safetensors)')
    Logger(f'Total checkpoints in {os.path.basename(base_path)}: {len(all_checkpoints) - removed_count}/{max_checkpoints}')


def load_checkpoint(model_path, device='cuda', load_training_state=True):
    """
    加载模型和 tokenizer（从 checkpoint 目录，使用 transformers from_pretrained）
    
    Args:
        model_path: 模型路径（目录）
        device: 设备
        load_training_state: 是否加载训练状态
    
    Returns:
        model, tokenizer, training_state (如果 load_training_state=True)
    """
    from vermind_models.models.modeling_vermind import VerMindForCausalLM
    from transformers import AutoTokenizer
    

    model_path = os.path.abspath(model_path) if not os.path.isabs(model_path) else model_path
    



    if os.path.exists(model_path) and os.path.isdir(model_path):


        try:
            model = VerMindForCausalLM.from_pretrained(
                model_path, 
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:


            Logger(f'Warning: Failed to load with local_files_only=True: {e}')
            Logger(f'Trying without local_files_only...')
            model = VerMindForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )
    else:

        model = VerMindForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    


    if hasattr(model, 'lm_head') and hasattr(model.model, 'embed_tokens'):
        if model.lm_head.weight.shape == model.model.embed_tokens.weight.shape:
            model.model.embed_tokens.weight = model.lm_head.weight
    
    model = model.to(device)
    

    tokenizer = None
    if os.path.exists(model_path):

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            pass
    
    if tokenizer is None:

        tokenizer = AutoTokenizer.from_pretrained('/root/vermind/vermind_tokenizer')
    
    training_state = None
    if load_training_state:
        training_state_path = os.path.join(model_path, 'training_state.pt')
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location='cpu')
            Logger(f'Training state loaded from {training_state_path}')
        else:
            Logger(f'No training state found at {training_state_path}')
    

    return model, tokenizer, training_state


def resume_training(model_path, device='cuda'):
    """
    恢复训练（从 checkpoint 目录加载）
    
    Args:
        model_path: 模型路径（目录）
        device: 设备
    
    Returns:
        model, tokenizer, training_state
    """
    model, tokenizer, training_state = load_checkpoint(model_path, device, load_training_state=True)
    
    saved_ws = training_state.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws:
        training_state['step'] = training_state['step'] * saved_ws // current_ws
        Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{training_state["step"]}')
    
    return model, tokenizer, training_state


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)