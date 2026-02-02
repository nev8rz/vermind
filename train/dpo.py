import os
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
import swanlab
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from vermind_models.config import VerMindConfig
from data_loader.dpo_dataset import DPODataset
from utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer
from vermind_models.models.modeling_vermind import VerMindForCausalLM

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):

    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    safe_labels = labels.clamp_min(0)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token # (batch_size, seq_len)


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1, aggregate="mean"):
    """
    DPO Loss. 序列级 log prob 支持 sum 或 mean。
    aggregate: "sum" 整句概率不除长度；"mean" 对 mask 位置求平均。
    104m 小模型使用 mean
    """
    policy_raw = (policy_log_probs * mask).sum(dim=1)
    ref_raw = (ref_log_probs * mask).sum(dim=1)
    if aggregate == "mean":
        seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8).squeeze(-1)
        policy_log_probs_sum = policy_raw / seq_lengths
        ref_log_probs_sum = ref_raw / seq_lengths
    else:
        policy_log_probs_sum = policy_raw
        ref_log_probs_sum = ref_raw

    batch_size = ref_log_probs.shape[0] // 2
    policy_chosen_logps = policy_log_probs_sum[:batch_size]
    policy_rejected_logps = policy_log_probs_sum[batch_size:]
    ref_chosen_logps = ref_log_probs_sum[:batch_size]
    ref_rejected_logps = ref_log_probs_sum[batch_size:]

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)

    with torch.no_grad():
        chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, swanlab=None, beta=0.1, aggregate="mean", tokenizer=None, base_save_path=None):
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)
            
            dpo_loss_val, chosen_rewards, rejected_rewards = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta, aggregate=aggregate)
            loss = dpo_loss_val + outputs.aux_loss
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
            current_dpo_loss = dpo_loss_val.item()
            # current_aux_loss = outputs.aux_loss.item()
            current_aux_loss = 0
            cr_val = chosen_rewards.item()
            rr_val = rejected_rewards.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, chosen_reward: {cr_val:.4f}, rejected_reward: {rr_val:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if swanlab:
                swanlab.log({
                    "loss": current_loss,
                    "dpo_loss": current_dpo_loss,
                    "aux_loss": current_aux_loss,
                    "chosen_reward": cr_val,
                    "rejected_reward": rr_val,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })

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
                swanlab=swanlab,
                max_checkpoints=args.max_checkpoints,
                save_interval=args.save_interval,
                steps_per_epoch=iters
            )
            model.train()

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs
        del dpo_loss_val, chosen_rewards, rejected_rewards, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind DPO Training")
    parser.add_argument("--save_dir", type=str, default="./output/dpo", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
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
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="../vermind_tokenizer", help="tokenizer路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始（支持目录路径或旧格式文件名）")
    parser.add_argument('--ref_weight', type=str, required=True, help="参考模型权重路径（用于计算ref_log_probs）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO损失函数中的beta参数")
    parser.add_argument("--dpo_aggregate", type=str, default="mean", choices=["sum", "mean"], help="序列级log prob聚合: sum 或 mean")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-DPO", help="swanlab项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
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
        swanlab_run_name = f"VerMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()
    

    Logger('Loading reference model...')
    ref_tokenizer = None
    if os.path.isdir(args.ref_weight):
        import glob
        checkpoint_pattern = os.path.join(args.ref_weight, "checkpoint_*")
        checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
            latest_checkpoint = checkpoints[-1]
            Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(latest_checkpoint)}')
            ref_model, ref_tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
        else:
            ref_model, ref_tokenizer, _ = load_checkpoint(args.ref_weight, device=args.device, load_training_state=False)
    else:
        ref_model = VerMindForCausalLM(lm_config)
        ref_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{args.save_dir}/{args.ref_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=args.device)
            if isinstance(weights, dict) and 'model' in weights:
                ref_model.load_state_dict(weights['model'], strict=False)
            else:
                ref_model.load_state_dict(weights, strict=False)
        ref_model = ref_model.to(args.device)
    
    if ref_tokenizer is None:
        ref_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    

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
            else:
                model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
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
            model = model.to(args.device)
    else:
        model = VerMindForCausalLM(lm_config)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        model = model.to(args.device)
        Logger('Model initialized from scratch')
    

    if 'tokenizer' not in locals():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    if args.use_compile == 1: 
        model = torch.compile(model)
        Logger('torch.compile enabled')
    

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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

        ref_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        ref_model = DistributedDataParallel(ref_model, device_ids=[local_rank])
    

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
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, swanlab_run, args.beta, args.dpo_aggregate, tokenizer, base_save_path)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, swanlab_run, args.beta, args.dpo_aggregate, tokenizer, base_save_path)
    

    if dist.is_initialized(): dist.destroy_process_group()
