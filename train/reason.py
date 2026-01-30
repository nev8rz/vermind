"""
Reasoning Model SFT 训练脚本
对 <think>, </think>, <answer>, </answer> 标签赋予更高权重
"""
import os
import argparse
import time
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
import swanlab
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from vermind_models.config import VerMindConfig
from data_loader import SFTDataset
from train.utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer
from vermind_models.models.modeling_vermind import VerMindForCausalLM

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, swanlab=None, base_save_path=None):
    """
    训练一个 epoch，对推理标签赋予更高权重
    """
    # 获取特殊标签的 token ids
    start_of_think_ids = tokenizer('<think>', add_special_tokens=False).input_ids
    end_of_think_ids = tokenizer('</think>', add_special_tokens=False).input_ids
    start_of_answer_ids = tokenizer('<answer>', add_special_tokens=False).input_ids
    end_of_answer_ids = tokenizer('</answer>', add_special_tokens=False).input_ids
    
    # 合并所有特殊 token ids
    special_token_ids = start_of_think_ids + end_of_think_ids + start_of_answer_ids + end_of_answer_ids
    special_token_ids_tensor = torch.tensor(special_token_ids, device=args.device)
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # 处理 batch
        if len(batch) == 2:
            input_ids, labels = batch
            attention_mask = None
        elif len(batch) == 3:
            input_ids, labels, attention_mask = batch
        else:
            input_ids, labels = batch[0], batch[1]
            attention_mask = None
        
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        
        # 学习率调整
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播
            res = model(input_ids, attention_mask=attention_mask)
            
            # 计算 logits 和 labels
            shift_logits = res.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算每个 token 的损失
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            # 创建基础 loss mask（忽略 -100 的位置）
            loss_mask = (shift_labels != -100).float()
            
            # 对特殊标签赋予更高权重
            # 将 shift_labels 展平，检查哪些位置是特殊 token
            shift_labels_flat = shift_labels.view(-1)
            is_special = torch.isin(shift_labels_flat, special_token_ids_tensor)
            
            # 应用权重：特殊标签使用 tag_weight，其他使用 1.0
            loss_mask_flat = loss_mask.view(-1)
            loss_mask_sum = loss_mask_flat.sum()
            
            # 特殊标签权重增强
            loss_mask_flat[is_special] = loss_mask_flat[is_special] * args.tag_weight
            loss_mask = loss_mask_flat.view(shift_labels.size())
            
            # 计算加权损失
            logits_loss = (loss * loss_mask).sum() / (loss_mask_sum + 1e-8)
            
            # 加上辅助损失（MoE）
            aux_loss = res.aux_loss if hasattr(res, 'aux_loss') and res.aux_loss is not None else 0.0
            total_loss = logits_loss + aux_loss
            total_loss = total_loss / args.accumulation_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = total_loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                   f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.1f}min')
            
            if swanlab:
                swanlab.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 保存 checkpoint
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                config=lm_config,
                save_path=base_save_path,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                swanlab=swanlab_run,
                max_checkpoints=args.max_checkpoints,
                save_interval=args.save_interval,
                steps_per_epoch=iters
            )
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind Reasoning Model SFT")
    parser.add_argument("--save_dir", type=str, default="../output/reason", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热步数比例（0.0-1.0）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="最大保留的 checkpoint 数量")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数（query heads）")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="键值头数（key-value heads）")
    parser.add_argument('--max_seq_len', default=2048, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/reason.jsonl", help="推理训练数据路径（包含<think><answer>标签）")
    parser.add_argument("--tokenizer_path", type=str, default="../vermind_tokenizer", help="tokenizer路径")
    parser.add_argument('--from_weight', default='sft', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--tag_weight', default=10.0, type=float, help="特殊标签(<think></think><answer></answer>)的权重倍数")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-Reason", help="swanlab项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = VerMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=bool(args.use_moe)
    )
    
    # 检查 resume checkpoint
    training_state = None
    resume_path = None
    if args.from_resume == 1:
        moe_suffix = '_moe' if lm_config.use_moe else ''
        resume_path = f'../checkpoints/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
        if os.path.exists(resume_path) and os.path.isdir(resume_path):
            try:
                _, _, training_state = resume_training(resume_path, device=args.device)
                Logger(f'Resumed from: {resume_path}')
            except Exception as e:
                Logger(f'Failed to resume: {e}')
                training_state = None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=dtype)
    
    # ========== 4. 配置 swanlab ==========
    swanlab_run = None
    if args.use_swanlab and is_main_process():
        swanlab_id = training_state.get('swanlab_id') if training_state else None
        resume = 'must' if swanlab_id else None
        swanlab_run_name = f"VerMind-Reason-Epoch-{args.epochs}-LR-{args.learning_rate}-TagW-{args.tag_weight}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()
    
    # ========== 5. 定义模型、数据、优化器 ==========
    if training_state is not None:
        model, tokenizer, _ = load_checkpoint(resume_path, device=args.device, load_training_state=False)
        Logger('Model loaded from resume checkpoint')
    elif args.from_weight != 'none':
        if os.path.isdir(args.from_weight):
            import glob
            checkpoint_pattern = os.path.join(args.from_weight, "checkpoint_*")
            checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
            
            if checkpoints:
                checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
                latest_checkpoint = checkpoints[-1]
                Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(latest_checkpoint)}')
                model, tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
                Logger(f'Model loaded from {latest_checkpoint}')
            else:
                model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
                Logger(f'Model loaded from {args.from_weight}')
        else:
            model = VerMindForCausalLM(lm_config)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            moe_suffix = '_moe' if lm_config.use_moe else ''
            weight_path = f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if os.path.exists(weight_path):
                weights = torch.load(weight_path, map_location=args.device)
                if isinstance(weights, dict) and 'model' in weights:
                    model.load_state_dict(weights['model'], strict=False)
                else:
                    model.load_state_dict(weights, strict=False)
                Logger(f'Model weights loaded from {weight_path}')
            model = model.to(args.device)
    else:
        model = VerMindForCausalLM(lm_config)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model = model.to(args.device)
        Logger('Model initialized from scratch')
    
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # 打印特殊标签 token 信息
    think_start_ids = tokenizer('<think>', add_special_tokens=False).input_ids
    think_end_ids = tokenizer('</think>', add_special_tokens=False).input_ids
    answer_start_ids = tokenizer('<answer>', add_special_tokens=False).input_ids
    answer_end_ids = tokenizer('</answer>', add_special_tokens=False).input_ids
    Logger(f'Special tags token ids: <think>={think_start_ids}, </think>={think_end_ids}, '
           f'<answer>={answer_start_ids}, </answer>={answer_end_ids}')
    Logger(f'Special tag weight: {args.tag_weight}x')
    
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 ckp 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if training_state:
        optimizer.load_state_dict(training_state['optimizer'])
        scaler.load_state_dict(training_state['scaler'])
        start_epoch = training_state['epoch']
        start_step = training_state.get('step', 0)
    
    # ========== 7. DDP 包装模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 确定基础保存路径 ==========
    moe_suffix = '_moe' if lm_config.use_moe else ''
    original_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path: {os.path.basename(base_save_path)}')
    
    # ========== 9. 开始训练 ==========
    Logger(f'Starting Reasoning Model training: epochs={args.epochs}, tag_weight={args.tag_weight}x')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step')
            train_epoch(epoch, loader, len(loader) + skip, tokenizer, lm_config, start_step, swanlab_run, base_save_path)
        else:
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, swanlab_run, base_save_path)
    
    # ========== 10. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
    
    Logger('Reasoning model training completed!')
