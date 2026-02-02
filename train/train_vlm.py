"""
VerMind-V (VLM) Pretraining Script
视觉-语言模型预训练，冻结 Vision Encoder，只训练 Vision Proj 和 LLM
"""
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
from vermind_models.config import VLMConfig
from data_loader.vlm_dataset import VLMDataset, vlm_collate_fn
from train.utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer, AutoModel
from vermind_models.models.modeling_vermind_v import VerMindVLM

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, swanlab=None, tokenizer=None, lm_config=None, base_save_path=None, args=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        pixel_values = batch["pixel_values"].to(args.device)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values
            )
            loss = res.loss / args.accumulation_steps

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
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab: swanlab.log({"loss": current_loss, "learning_rate": current_lr, "epoch_time": eta_min})

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

        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind-V Training (Pretrain/SFT)")
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录 (默认: ../output/vlm_pretrain 或 ../output/vlm_sft)")
    parser.add_argument('--stage', type=str, default='pretrain', choices=['pretrain', 'sft'], 
                        help="训练阶段: pretrain(冻结LLM,只训练projection) 或 sft(全量训练)")
    parser.add_argument('--save_weight', default=None, type=str, help="保存权重的前缀名 (默认: vlm_pretrain 或 vlm_sft)")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热步数比例")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="最大保留的 checkpoint 数量")
    

    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="键值头数")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    

    parser.add_argument("--data_path", type=str, required=True, help="Parquet 数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./vermind_tokenizer", help="tokenizer路径")
    parser.add_argument("--vision_encoder_path", type=str, default="./siglip-base-patch16-224", help="Vision Encoder 路径")
    

    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练（支持目录路径）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument('--freeze_vision', default=1, type=int, choices=[0, 1], help="是否冻结 Vision Encoder（推荐冻结）")
    parser.add_argument('--freeze_llm', default=None, type=int, choices=[0, 1], help="是否冻结 LLM（默认: pretrain阶段冻结, sft阶段解冻）")
    

    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default=None, help="swanlab项目名 (默认: VerMind-V-Pretrain 或 VerMind-V-SFT)")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    args = parser.parse_args()


    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=bool(args.use_moe)
    )
    

    if args.save_dir:
        save_dir = args.save_dir
    elif args.stage == 'sft':
        save_dir = '../output/vlm_sft'
    else:
        save_dir = '../output/vlm_pretrain'
    
    if args.save_weight:
        save_weight = args.save_weight
    elif args.stage == 'sft':
        save_weight = 'vlm_sft'
    else:
        save_weight = 'vlm_pretrain'
    

    training_state = None
    resume_path = None
    if args.from_resume == 1:
        moe_suffix = '_moe' if lm_config.use_moe else ''
        resume_path = f'{save_dir}/{save_weight}_{lm_config.hidden_size}{moe_suffix}'
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

        if args.swanlab_project:
            swanlab_project = args.swanlab_project
        elif args.stage == 'sft':
            swanlab_project = "VerMind-V-SFT"
        else:
            swanlab_project = "VerMind-V-Pretrain"
        swanlab_run_name = f"VerMind-V-{args.stage.upper()}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        swanlab.init(project=swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()
    

    if training_state is not None:

        model, tokenizer, _ = load_checkpoint(resume_path, device=args.device, load_training_state=False)
        Logger('Model and tokenizer loaded from resume checkpoint')
    elif args.from_weight != 'none':

        Logger(f'Initializing VLM and loading LLM weights from: {args.from_weight}')
        

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model = VerMindVLM(lm_config, vision_model_path=args.vision_encoder_path)
        

        load_path = args.from_weight
        if os.path.isdir(args.from_weight):
            import glob
            checkpoint_pattern = os.path.join(args.from_weight, "checkpoint_*")
            checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
            
            if checkpoints:
                checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
                load_path = checkpoints[-1]
                Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(load_path)}')
            else:
                load_path = args.from_weight
        

        Logger(f'Loading LLM weights from: {load_path}')
        try:
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(load_path, "model.safetensors"))
        except Exception as e:
            Logger(f'Failed to load safetensors: {e}, trying torch.load')
            state_dict = torch.load(os.path.join(load_path, "pytorch_model.bin"), map_location='cpu')
        

        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state:
                if value.shape == model_state[key].shape:
                    model_state[key].copy_(value)
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not in VLM)")
        

        llm_keys = [k for k in loaded_keys if k.startswith('model.') or k == 'lm_head.weight']
        vision_proj_keys = [k for k in loaded_keys if k.startswith('vision_proj.')]
        vision_encoder_keys = [k for k in loaded_keys if k.startswith('vision_encoder.')]
        
        Logger(f'Loaded {len(loaded_keys)} keys from checkpoint:')
        Logger(f'  - LLM: {len(llm_keys)} keys')
        if vision_proj_keys:
            Logger(f'  - Vision Proj: {len(vision_proj_keys)} keys (pre-trained)')
        else:
            Logger(f'  - Vision Proj: 0 keys (will be trained from scratch)')
        if vision_encoder_keys:
            Logger(f'  - Vision Encoder: {len(vision_encoder_keys)} keys')
        
        if skipped_keys:
            Logger(f'Skipped {len(skipped_keys)} keys')
        
        model = model.to(args.device)
    else:

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model = VerMindVLM(lm_config, vision_model_path=args.vision_encoder_path)
        model = model.to(args.device)
        Logger('Model initialized from scratch')
    

    if model.vision_encoder is not None and is_main_process():
        has_nan_or_inf = any(torch.isnan(p).any() or torch.isinf(p).any() 
                             for p in model.vision_encoder.parameters())
        if has_nan_or_inf:
            Logger('Vision Encoder has NaN/Inf, re-loading from original...')
            try:
                original_vision = AutoModel.from_pretrained(
                    args.vision_encoder_path, 
                    local_files_only=True,
                    torch_dtype=torch.float32
                )
                model.vision_encoder.load_state_dict(original_vision.state_dict(), strict=False)
                Logger('Vision Encoder weights re-loaded')
            except Exception as e:
                Logger(f'Failed to re-load Vision Encoder: {e}')
    

    if args.freeze_llm is not None:

        freeze_llm = args.freeze_llm
        mode_str = "冻结 LLM" if freeze_llm == 1 else "全量训练"
        Logger(f'[Stage: {args.stage}] 用户指定: {mode_str}')
    elif args.stage == 'pretrain':

        freeze_llm = 1
        Logger(f'[Stage: Pretrain] 默认: 冻结 LLM，只训练 Vision Projection')
    else:  # sft

        freeze_llm = 0
        Logger(f'[Stage: SFT] 默认: 全量训练（包括 LLM 和 Vision Projection）')
    

    if args.freeze_vision == 1 and model.vision_encoder is not None:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        Logger('Vision Encoder frozen')
    

    if freeze_llm == 1:
        for name, param in model.named_parameters():
            if "vision_proj" not in name:
                param.requires_grad = False
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        Logger('LLM frozen (only training Vision Projection)')
    else:

        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in trainable_params_list) / 1e6
    Logger(f'Total params: {total_params:.2f}M, Trainable: {trainable_params:.2f}M')
    
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    

    train_ds = VLMDataset(
        parquet_path=args.data_path,
        tokenizer=tokenizer,
        vision_encoder_path=args.vision_encoder_path,
        max_length=args.max_seq_len
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))
    

    start_epoch, start_step = 0, 0
    if training_state:
        start_epoch = training_state['epoch']
        start_step = training_state.get('step', 0)
    

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    Logger(f'Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, Trainable: {sum(p.numel() for p in trainable_params_list)/1e6:.2f}M')
    
    optimizer = optim.AdamW(trainable_params_list, lr=args.learning_rate)
    

    if training_state:
        try:
            optimizer.load_state_dict(training_state['optimizer'])
            scaler.load_state_dict(training_state['scaler'])
        except Exception as e:
            Logger(f'Failed to restore optimizer state: {e}')
    

    moe_suffix = '_moe' if lm_config.use_moe else ''
    original_save_path = f'{save_dir}/{save_weight}_{lm_config.hidden_size}{moe_suffix}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path determined: {os.path.basename(base_save_path)}')
    

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        

        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=vlm_collate_fn
        )
        
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, tokenizer, lm_config, base_save_path, args)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, tokenizer, lm_config, base_save_path, args)
    

    if dist.is_initialized():
        dist.destroy_process_group()
