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
from data_loader import SFTDataset, SFTDatasetPacked, collate_fn_packed
from utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer
from vermind_models.models.modeling_vermind import VerMindForCausalLM

warnings.filterwarnings('ignore')


monitoring_data = {
    'cu_seqlens_info': [],
    'position_ids_info': [],
    'attention_output_stats': []
}

def monitor_attention_hook(module, input, output):
    if isinstance(output, tuple):
        attn_output = output[0]
    else:
        attn_output = output
    
    if attn_output is not None:
        stats = {
            'shape': list(attn_output.shape),
            'mean': attn_output.mean().item(),
            'std': attn_output.std().item(),
            'min': attn_output.min().item(),
            'max': attn_output.max().item(),
        }
        monitoring_data['attention_output_stats'].append(stats)


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, tokenizer=None, lm_config=None, base_save_path=None, 
                use_packed=False, monitor_steps=5):
    start_time = time.time()
    monitored_steps = 0
    
    for step, batch in enumerate(loader, start=start_step + 1):
        if len(batch) == 2:
            input_ids, labels = batch
            cu_seqlens = None
            position_ids = None
        elif len(batch) == 4:
            input_ids, labels, cu_seqlens, position_ids = batch
        else:
            input_ids, labels, cu_seqlens, _, position_ids = batch
        
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(args.device)
        if position_ids is not None:
            position_ids = position_ids.to(args.device)
        
        if monitored_steps < monitor_steps:
            Logger(f"\n=== Step {step} 监控信息 ===")
            Logger(f"Input shape: {input_ids.shape}")
            Logger(f"Labels shape: {labels.shape}")
            
            if cu_seqlens is not None:
                cu_info = {
                    'shape': list(cu_seqlens.shape),
                    'dtype': str(cu_seqlens.dtype),
                    'values': cu_seqlens.tolist()[:10] if len(cu_seqlens) > 10 else cu_seqlens.tolist(),
                    'batch_size': len(cu_seqlens) - 1,
                    'total_tokens': cu_seqlens[-1].item() if len(cu_seqlens) > 0 else 0,
                }
                monitoring_data['cu_seqlens_info'].append(cu_info)
                Logger(f"cu_seqlens: {cu_info}")
            
            if position_ids is not None:
                pos_info = {
                    'shape': list(position_ids.shape),
                    'dtype': str(position_ids.dtype),
                    'min': position_ids.min().item(),
                    'max': position_ids.max().item(),
                    'mean': position_ids.float().mean().item(),
                }
                if position_ids.shape[0] > 0:
                    unique_starts = []
                    for i in range(min(100, len(position_ids))):
                        if i == 0 or position_ids[i] == 0:
                            unique_starts.append(i)
                    pos_info['sample_starts'] = unique_starts[:5]
                
                monitoring_data['position_ids_info'].append(pos_info)
                Logger(f"Position IDs: {pos_info}")
            else:
                Logger("Position IDs: None (使用默认绝对位置)")
            
            
            monitored_steps += 1
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, cu_seqlens=cu_seqlens, position_ids=position_ids)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab: swanlab.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

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
    parser = argparse.ArgumentParser(description="VerMind SFT Packed (支持打包数据集训练)")
    parser.add_argument("--save_dir", type=str, default="../output/sft_packed", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_packed', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热步数比例")
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
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="键值头数")
    parser.add_argument('--max_seq_len', default=2048, type=int, help="训练的最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_512.jsonl", help="SFT训练数据路径（use_packed=0 时用）")
    parser.add_argument("--parquet_path", type=str, default="", help="Packed Parquet 路径（use_packed=1 时用）；不填则从 data_path 推导：.cache/sft_packed/<stem>.parquet")
    parser.add_argument("--tokenizer_path", type=str, default="/root/vermind/vermind_tokenizer", help="tokenizer路径")
    parser.add_argument('--from_weight', default='/root/vermind/output/pretrain', type=str, help="基于哪个权重训练（默认从pretrain加载）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--use_packed", default=1, type=int, choices=[0, 1], help="是否使用打包数据集（0=否，1=是，默认1）")
    parser.add_argument("--monitor_steps", type=int, default=0, help="监控前N个step的详细信息（0=不监控）")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-SFT-Packed", help="swanlab项目名")
    args = parser.parse_args()


    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = VerMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=bool(args.use_moe)
    )
    
    training_state = None
    resume_path = None
    if args.from_resume == 1:
        moe_suffix = '_moe' if lm_config.use_moe else ''
        resume_path = f'../checkpoints/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
        if os.path.exists(resume_path) and os.path.isdir(resume_path):
            try:
                _, _, training_state = resume_training(resume_path, device=args.device)
                Logger(f'Resumed from transformers checkpoint: {resume_path}')
            except Exception as e:
                Logger(f'Failed to resume from {resume_path}: {e}')
                training_state = None
    

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=dtype)
    

    swanlab_run = None
    if args.use_swanlab and is_main_process():
        swanlab_id = training_state.get('swanlab_id') if training_state else None
        resume = 'must' if swanlab_id else None
        swanlab_run_name = f"VerMind-SFT-Packed-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()
    

    if training_state is not None:
        model, tokenizer, _ = load_checkpoint(resume_path, device=args.device, load_training_state=False)
        Logger('Model and tokenizer loaded from resume checkpoint')
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
                Logger(f'Model and tokenizer loaded from {latest_checkpoint}')
            else:
                model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
                Logger(f'Model and tokenizer loaded from {args.from_weight}')
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
    
    if len(model.model.layers) > 0:
        first_attn = model.model.layers[0].self_attn
        first_attn.register_forward_hook(monitor_attention_hook)
        Logger('Registered attention monitoring hook')
    

    if args.use_packed:
        Logger('使用打包数据集 (SFTDatasetPacked)')
        parquet_path = args.parquet_path.strip()
        if not parquet_path:
            base = os.path.splitext(os.path.basename(args.data_path))[0]
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "sft_packed")
            parquet_path = os.path.join(cache_dir, f"{base}.parquet")
            Logger(f'未指定 --parquet_path，使用: {parquet_path}')
        train_ds = SFTDatasetPacked(parquet_path=parquet_path, max_length=args.max_seq_len)
        collate_fn = collate_fn_packed
    else:
        Logger('使用普通数据集 (SFTDataset)')
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        collate_fn = None
    
    Logger(f'数据集大小: {len(train_ds)} 个样本/序列')
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    

    start_epoch, start_step = 0, 0
    if training_state:
        optimizer.load_state_dict(training_state['optimizer'])
        scaler.load_state_dict(training_state['scaler'])
        start_epoch = training_state['epoch']
        start_step = training_state.get('step', 0)
    

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    

    moe_suffix = '_moe' if lm_config.use_moe else ''
    original_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path determined: {os.path.basename(base_save_path)}')
    

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, 
                           pin_memory=True, collate_fn=collate_fn)
        
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, tokenizer, lm_config, base_save_path,
                       use_packed=bool(args.use_packed), monitor_steps=args.monitor_steps)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, tokenizer, lm_config, base_save_path,
                       use_packed=bool(args.use_packed), monitor_steps=args.monitor_steps)
    

    if is_main_process():
        Logger('\n' + '='*80)
        Logger('监控总结:')
        Logger(f'cu_seqlens 信息: {len(monitoring_data["cu_seqlens_info"])} 条记录')
        Logger(f'Position IDs 信息: {len(monitoring_data["position_ids_info"])} 条记录')
        Logger(f'Attention 输出统计: {len(monitoring_data["attention_output_stats"])} 条记录')
        Logger('='*80)
    

    if dist.is_initialized(): dist.destroy_process_group()
