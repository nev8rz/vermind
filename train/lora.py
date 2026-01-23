import os
import argparse
import time
import warnings
import torch
import torch.distributed as dist
import swanlab
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from vermind_models.config import VerMindConfig
from data_loader import SFTDataset
from utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer
from vermind_models.models.modeling_vermind import VerMindForCausalLM
from vermind_models.lora_adpater import apply_lora, load_lora, save_lora

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, tokenizer=None, lm_config=None, base_save_path=None):
    start_time = time.time() # 开始时间
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device) # 将input_ids移动到设备
        labels = labels.to(args.device) # 将labels移动到设备
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio) # 获取学习率
        for param_group in optimizer.param_groups: # 遍历优化器参数组
            param_group['lr'] = lr # 设置学习率

        with autocast_ctx: # 混合精度训练
            res = model(input_ids, labels=labels)
            loss = res.loss
            loss = loss / args.accumulation_steps # 损失除以梯度累积步数

        scaler.scale(loss).backward() # 梯度累积

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab: swanlab.log({"loss": current_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 保存LoRA权重（使用全局步数）
            global_step = epoch * iters + step
            lora_save_path = os.path.join(base_save_path, f'lora_{global_step}.pth')
            save_lora(model, lora_save_path)
            Logger(f'LoRA weights saved to {lora_save_path}')

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind LoRA Fine-Tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='lora', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率（LoRA通常使用较大学习率）")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热步数比例（0.0-1.0）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="LoRA权重保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数（query heads）")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="键值头数（key-value heads）")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft.jsonl", help="SFT训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="../vermind_tokenizer", help="tokenizer路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练（支持目录路径或旧格式文件名）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA的秩（rank）")
    parser.add_argument('--lora_target_modules', default='q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj', type=str, help="要应用LoRA的模块，用逗号分隔，如果为None则自动应用到所有方阵Linear层")
    parser.add_argument('--lora_load_from', default='none', type=str, help="从已有的LoRA权重加载（none表示不加载）")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-LoRA", help="swanlab项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数 ==========
    os.makedirs(args.save_dir, exist_ok=True) # 创建保存目录
    lm_config = VerMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=False
    ) # 配置模型参数
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=dtype)
    
    # ========== 4. 配置swanlab ==========
    swanlab_run = None
    if args.use_swanlab and is_main_process():
        swanlab_run_name = f"VerMind-LoRA-Rank{args.lora_rank}-Alpha{args.lora_alpha}-Epoch{args.epochs}-BatchSize{args.batch_size}-LR{args.learning_rate}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name)
        swanlab_run = swanlab.get_run()
    
    # ========== 5. 加载模型和tokenizer ==========
    # 从基础模型加载
    if args.from_weight != 'none':
        if os.path.isdir(args.from_weight):
            # 如果是目录，检查是否是基础路径（包含 checkpoint_* 子目录）还是具体的 checkpoint 路径
            import glob
            checkpoint_pattern = os.path.join(args.from_weight, "checkpoint_*")
            checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
            
            if checkpoints:
                # 如果是基础路径（包含多个 checkpoint），自动选择最新的
                checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
                latest_checkpoint = checkpoints[-1]
                Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(latest_checkpoint)}')
                model, tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
                Logger(f'Model and tokenizer loaded from {latest_checkpoint}')
            else:
                # 如果是具体的 checkpoint 目录，直接加载
                model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
                Logger(f'Model and tokenizer loaded from {args.from_weight}')
        else:
            # 兼容旧格式：从 .pth 文件加载
            model = VerMindForCausalLM(lm_config)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            weight_path = f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_size}.pth'
            if os.path.exists(weight_path):
                weights = torch.load(weight_path, map_location=args.device)
                if isinstance(weights, dict) and 'model' in weights:
                    model.load_state_dict(weights['model'], strict=False)
                else:
                    model.load_state_dict(weights, strict=False)
                Logger(f'Model weights loaded from {weight_path}')
            model = model.to(args.device)
    else:
        raise ValueError("LoRA training requires a base model. Please specify --from_weight")
    
    # 应用LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(',')] if args.lora_target_modules else None
    apply_lora(model, rank=args.lora_rank, target_modules=target_modules)
    Logger(f'LoRA applied with rank={args.lora_rank}, target_modules={target_modules}')
    
    # 加载已有的LoRA权重（如果指定）
    if args.lora_load_from != 'none':
        if os.path.exists(args.lora_load_from):
            load_lora(model, args.lora_load_from)
            Logger(f'LoRA weights loaded from {args.lora_load_from}')
        else:
            Logger(f'Warning: LoRA weight file not found: {args.lora_load_from}')
    
    # 只优化LoRA参数
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
    total_lora_params = sum(p.numel() for p in lora_params)
    total_model_params = sum(p.numel() for p in model.parameters())
    Logger(f'LoRA parameters: {total_lora_params / 1e6:.2f}M / {total_model_params / 1e6:.2f}M ({total_lora_params / total_model_params * 100:.2f}%)')
    
    if args.use_compile == 1: 
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)  # 只优化LoRA参数
    
    # ========== 6. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 7. 确定基础保存路径 ==========
    original_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path determined: {os.path.basename(base_save_path)}')
    
    # ========== 8. 开始训练 ==========
    for epoch in range(args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader), 0, swanlab_run, tokenizer, lm_config, base_save_path)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
