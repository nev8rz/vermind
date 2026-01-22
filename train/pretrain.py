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
from data_loader.pretrain_dataset import PretrainDataset
from utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer
from vermind_models import VerMindConfig
from vermind_models.models.modeling_vermind import VerMindForCausalLM

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
            loss = res.loss + res.aux_loss
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
            # current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_aux_loss = 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if swanlab: swanlab.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # 使用在训练开始前确定的基础路径
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
                max_checkpoints=3,  # 默认保留3个checkpoint
                save_interval=args.save_interval
            )
            model.train()

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="预热步数比例（0.0-1.0）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数（query heads）")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="键值头数（key-value heads）")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="../vermind_tokenizer", help="tokenizer路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始（支持目录路径或旧格式文件名）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-Pretrain", help="swanlab项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True) # 创建保存目录
    lm_config = VerMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=bool(args.use_moe)
    ) # 配置模型参数
    
    # 检查 resume checkpoint（使用新的 transformers 接口）
    training_state = None
    resume_path = None
    if args.from_resume == 1:
        # 尝试从标准 transformers 格式目录恢复
        moe_suffix = '_moe' if lm_config.use_moe else ''
        resume_path = f'../checkpoints/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
        if os.path.exists(resume_path) and os.path.isdir(resume_path):
            try:
                _, _, training_state = resume_training(resume_path, device=args.device)
                Logger(f'Resumed from transformers checkpoint: {resume_path}')
            except Exception as e:
                Logger(f'Failed to resume from {resume_path}: {e}')
                training_state = None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16 #
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=dtype) # 混合精度训练上下文
    
    # ========== 4. 配swanlab ==========
    swanlab_run = None
    if args.use_swanlab and is_main_process():
        swanlab_id = training_state.get('swanlab_id') if training_state else None
        resume = 'must' if swanlab_id else None
        swanlab_run_name = f"VerMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 初始化模型和 tokenizer
    if training_state is not None:
        # 从 resume checkpoint 加载
        model, tokenizer, _ = load_checkpoint(resume_path, device=args.device, load_training_state=False)
        Logger('Model and tokenizer loaded from resume checkpoint')
    elif args.from_weight != 'none':
        # 从指定权重加载
        if os.path.isdir(args.from_weight):
            # 如果是目录，使用 load_checkpoint 加载
            model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
            Logger(f'Model and tokenizer loaded from {args.from_weight}')
        else:
            # 兼容旧格式：从 .pth 文件加载
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
        # 从头开始训练
        model = VerMindForCausalLM(lm_config)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model = model.to(args.device)
        Logger('Model initialized from scratch')
    
    if args.use_compile == 1: 
        model = torch.compile(model) # 使用torch.compile加速模型
        Logger('torch.compile enabled')
    
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len) # 初始化数据集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None # 分布式采样器
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16')) 
    # 梯度缩放器，只需要在 dtype 为 float16 时启用，因为float16的数值范围更小，容易下溢为0
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate) # 优化器
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if training_state:
        optimizer.load_state_dict(training_state['optimizer'])
        scaler.load_state_dict(training_state['scaler'])
        start_epoch = training_state['epoch']
        start_step = training_state.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 忽略freqs_cos和freqs_sin，ddp时候，由于每张卡的seq_length不同，会导致freqs_cos和freqs_sin的shape不同，导致ddp失败，所以不要同步freqs_cos和freqs_sin
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 7.5. 确定基础保存路径（在训练开始前确定一次） ==========
    moe_suffix = '_moe' if lm_config.use_moe else ''
    original_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path determined: {os.path.basename(base_save_path)}')
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); 
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip) # 跳过步数采样器
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        # 标准 ddp 多卡数据加载方式，pin_memory=True 把数据提前拷贝到 CUDA 固定内存（pinned memory），提升数据加载速度
        if skip > 0: # 如果跳过步数大于0，则打印日志，一般是resume模式
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, swanlab_run, tokenizer, lm_config, base_save_path)
        else:
            train_epoch(epoch, loader, len(loader), 0, swanlab_run, tokenizer, lm_config, base_save_path)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()