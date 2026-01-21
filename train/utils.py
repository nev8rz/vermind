"""
训练工具函数集合
"""
import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from vermind_models.models.modeling_vermind import VerMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0)) # 路由专家数量
    n_active = getattr(config, 'num_experts_per_tok', 0) # 每个token选择的专家数量
    n_shared = getattr(config, 'n_shared_experts', 0) # 共享专家数量
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6 # 路由专家参数数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6 # 共享专家参数数量
    base = total - (expert * n_routed) - (shared_expert * n_shared) # 基础参数数量
    active = base + (expert * n_active) + (shared_expert * n_shared) # 活跃参数数量
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M') # 打印模型参数
    else: Logger(f'Model Params: {total:.2f}M') # 打印模型参数


def is_main_process(): # 判断是否是主进程
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content): # 打印日志
    if is_main_process():
        print(content)


def get_lr(step, total_steps, lr, warmup_ratio=0.0):
    warmup_steps = int(total_steps * warmup_ratio) # 预热步数
    if warmup_steps > 0 and step < warmup_steps:
        return lr * step / warmup_steps
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式，world_size为1

    dist.init_process_group(backend="nccl") # 初始化分布式训练环境
    local_rank = int(os.environ["LOCAL_RANK"]) # 获取本地rank
    torch.cuda.set_device(local_rank) # 设置本地设备
    return local_rank


def setup_seed(seed: int): # 设置随机种子，保证训练结果可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, swanlab=None, save_dir='../checkpoints', **kwargs):
    # 两个功能 1. 保存模型权重 2. 恢复模型权重
    os.makedirs(save_dir, exist_ok=True) # 创建保存目录
    moe_path = '_moe' if lm_config.use_moe else '' 
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth' # 保存路径
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth' # 恢复路径

    if model is not None: # 保存模型权重
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model # 获取原始模型，DistributedDataParallel会被包裹一层module
        raw_model = getattr(raw_model, '_orig_mod', raw_model) # 获取原始模型，torch.compile会被重命名为_orig_mod
        state_dict = raw_model.state_dict() # 获取模型权重
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()} # 将模型权重转换为半精度
        ckp_tmp = ckp_path + '.tmp' # 临时保存路径
        torch.save(state_dict, ckp_tmp) # 保存模型权重
        os.replace(ckp_tmp, ckp_path) # 原子替换，防止在写入过程中程序崩溃导致原有的 Checkpoint 文件损坏
        swanlab_id = None # swanlab id
        if swanlab: # 如果swanlab存在
            if hasattr(swanlab, 'get_run'): # 如果swanlab有get_run方法，
                run = swanlab.get_run() # 获取swanlab run
                swanlab_id = getattr(run, 'id', None) if run else None # 获取swanlab id
            else:
                swanlab_id = getattr(swanlab, 'id', None) # 获取swanlab id

        resume_data = { # 恢复数据，dict类型
            'model': state_dict,
            'optimizer': optimizer.state_dict(), # 优化器状态
            'epoch': epoch,
            'step': step, # 当前步数
            'world_size': dist.get_world_size() if dist.is_initialized() else 1, # world_size
            'swanlab_id': swanlab_id # swanlab id
        }
        for key, value in kwargs.items(): # 遍历kwargs，将kwargs中的状态保存到resume_data中
            if value is not None:
                if hasattr(value, 'state_dict'): # 如果value有state_dict方法，则保存value的状态
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value) # 获取原始模型
                    resume_data[key] = raw_value.state_dict() # 保存模型状态
                else: # 如果value没有state_dict方法，则保存value，一般是静态参数
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp' # 临时恢复路径
        torch.save(resume_data, resume_tmp) # 保存恢复数据
        os.replace(resume_tmp, resume_path) # 原子替换，防止在写入过程中程序崩溃导致原有的 Checkpoint 文件损坏
        del state_dict, resume_data # 删除临时数据
        torch.cuda.empty_cache() # 清空缓存
    else:  # resume模式
        if os.path.exists(resume_path): # 如果恢复路径存在
            ckp_data = torch.load(resume_path, map_location='cpu') # 加载恢复数据
            saved_ws = ckp_data.get('world_size', 1) # 获取保存的world_size
            current_ws = dist.get_world_size() if dist.is_initialized() else 1 # 获取当前的world_size
            if saved_ws != current_ws: # 如果保存的world_size不等于当前的world_size
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws # 计算当前的step
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}') # 打印日志
            return ckp_data # 返回恢复数据
        return None # 返回None

# 初始化模型，支持分布式训练
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../vermind_tokenizer', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = VerMindForCausalLM(lm_config) # 初始化模型

    if from_weight!= 'none': # 如果from_weight不为none，则加载模型权重
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config) # 打印模型参数
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M') # 打印可训练参数数量
    return model.to(device), tokenizer # 返回模型和tokenizer


class SkipBatchSampler(Sampler): # 跳过批次采样器，跳过前skip_batches个批次，resume时使用
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