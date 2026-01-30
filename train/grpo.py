"""
GRPO (Group Relative Policy Optimization) 训练脚本
基于组内相对优势的强化学习训练，无需critic模型
"""
import os
import argparse
import re
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
from vermind_models.models.modeling_vermind import VerMindForCausalLM
from data_loader.rlaif_dataset import RLAIFDataset
from train.utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, device, num_generations, reasoning=0):
    """
    整合所有奖励函数计算总奖励

    Args:
        prompts: 提示文本列表，长度 B
        responses: 响应文本列表，长度 B*num_generations
        reward_model: 奖励模型
        reward_tokenizer: 奖励模型的tokenizer
        device: 计算设备
        num_generations: 每个prompt生成的响应数量
        reasoning: 是否为推理模式（增加格式奖励）

    Returns:
        rewards: 奖励张量 [B*num_generations]
    """
    def reasoning_model_reward(rewards):
        """计算推理模型的格式奖励"""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]
        
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards = rewards + torch.tensor(format_rewards, device=device)
        
        # 标记奖励（防止严格奖励稀疏）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward
        
        mark_rewards = [mark_num(response) for response in responses]
        rewards = rewards + torch.tensor(mark_rewards, device=device)
        return rewards
    
    rewards = torch.zeros(len(responses), device=device)
    
    # 格式奖励（仅推理模式）
    if reasoning == 1:
        rewards = reasoning_model_reward(rewards)
    
    # 使用 reward model 计算奖励
    if reward_model is not None:
        with torch.no_grad():
            reward_model_scores = []
            batch_size = len(prompts)
            scale = 3.0
            
            for i in range(batch_size):
                for j in range(num_generations):
                    response_idx = i * num_generations + j
                    response = responses[response_idx]
                    prompt = prompts[i]
                    
                    # 解析 prompt 中的对话历史
                    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                    matches = re.findall(pattern, prompt, re.DOTALL)
                    messages = [{"role": role, "content": content.strip()} for role, content in matches]
                    
                    tmp_chat = messages + [{"role": "assistant", "content": response}]
                    
                    try:
                        score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        score = max(min(score, scale), -scale)
                        
                        # 当 reasoning=1 时，额外计算 <answer> 内容的奖励
                        if reasoning == 1:
                            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                            if answer_match:
                                answer_content = answer_match.group(1).strip()
                                tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                                answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                                answer_score = max(min(answer_score, scale), -scale)
                                score = score * 0.4 + answer_score * 0.6
                    except Exception:
                        score = 0.0
                    
                    reward_model_scores.append(score)
            
            reward_model_scores = torch.tensor(reward_model_scores, device=device)
            rewards = rewards + reward_model_scores
    
    return rewards


def get_per_token_logps(model, input_ids, n_keep):
    """
    获取每个token的log概率
    
    Args:
        model: 模型
        input_ids: 输入token ids [B, L]
        n_keep: 需要保留的最后n_keep个token的log概率
        
    Returns:
        per_token_logps: [B, n_keep]
    """
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    logits = model(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


def grpo_train_epoch(epoch, loader, iters, model, ref_model, reward_model, reward_tokenizer,
                     optimizer, scheduler, scaler, tokenizer, lm_config, args,
                     autocast_ctx, start_step=0, swanlab=None, base_save_path=None):
    """
    GRPO 训练一个 epoch
    """
    model.train()
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        
        # 编码 prompt
        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False,
            padding_side="left", 
            add_special_tokens=False
        ).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
        
        # 使用模型生成多个响应
        with torch.no_grad():
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )  # [B*num_gen, P+R]
        
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
        # 计算当前策略的 per-token log probabilities
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        # 计算参考模型的 per-token log probabilities
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        
        # 解码响应并计算奖励
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(
            prompts, completions, reward_model, reward_tokenizer,
            args.device, args.num_generations, args.reasoning
        )  # [B*num_gen]
        
        # 计算组内相对优势
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]
        
        # 创建 completion mask（处理 eos）
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)
        ).int()  # [B*num_gen, R]
        
        # 计算 KL 散度（相对于参考模型）
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        
        # 计算 GRPO 损失
        # policy gradient with KL penalty
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) 
            - args.beta * per_token_kl
        )  # [B*num_gen, R]
        
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # 日志记录
        if (step % args.log_interval == 0 or step == iters) and is_main_process():
            spend_time = time.time() - start_time
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']
            eta_min = spend_time / step * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Policy Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, '
                   f'Reward: {avg_reward_val:.4f}, Avg Response Len: {avg_len_val:.2f}, '
                   f'LR: {current_lr:.8f}, ETA: {eta_min:.1f}min')
            
            if swanlab:
                swanlab.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr,
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
                swanlab=swanlab,
                max_checkpoints=args.max_checkpoints,
                save_interval=args.save_interval,
                steps_per_epoch=iters,
            )
            model.train()
        
        # 清理内存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages
        del completion_mask, kl_div, per_token_kl, per_token_loss, policy_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind GRPO Training")
    # 基础训练参数
    parser.add_argument("--save_dir", type=str, default="./output/grpo", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size（每个prompt的数量）")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的响应数量（G）")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热步数比例（0.0-1.0）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="最大保留的 checkpoint 数量")

    # 模型参数
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--num_attention_heads', default=8, type=int, help="注意力头数（query heads）")
    parser.add_argument('--num_key_value_heads', default=2, type=int, help="键值头数（key-value heads）")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--max_gen_len', default=256, type=int, help="生成的最大长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")

    # 数据和权重路径
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="../vermind_tokenizer", help="tokenizer路径")
    parser.add_argument('--from_weight', default='none', type=str, help="初始权重路径")
    parser.add_argument('--ref_weight', type=str, required=True, help="参考模型权重路径")
    parser.add_argument('--reward_model_path', type=str, default='', help="奖励模型路径（留空则不使用外部奖励模型）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")

    # GRPO 特定参数
    parser.add_argument("--beta", type=float, default=0.04, help="KL 惩罚系数")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成温度")
    parser.add_argument("--reasoning", type=int, default=0, choices=[0, 1], help="是否训练推理模型（增加格式奖励）")

    # 其他
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-GRPO", help="swanlab项目名")
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
                Logger(f'Resumed from transformers checkpoint: {resume_path}')
            except Exception as e:
                Logger(f'Failed to resume from {resume_path}: {e}')
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
        swanlab_run_name = f"VerMind-GRPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-G-{args.num_generations}-LR-{args.learning_rate}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()

    # ========== 5. 加载参考模型 (ref_model) ==========
    Logger('Loading reference model...')
    if os.path.isdir(args.ref_weight):
        import glob
        checkpoint_pattern = os.path.join(args.ref_weight, "checkpoint_*")
        checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]

        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
            latest_checkpoint = checkpoints[-1]
            Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(latest_checkpoint)}')
            ref_model, ref_tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
            Logger(f'Reference model loaded from {latest_checkpoint}')
        else:
            ref_model, ref_tokenizer, _ = load_checkpoint(args.ref_weight, device=args.device, load_training_state=False)
            Logger(f'Reference model loaded from {args.ref_weight}')
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
            Logger(f'Reference model weights loaded from {weight_path}')
        ref_model = ref_model.to(args.device)

    if ref_tokenizer is None:
        ref_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    Logger('Reference model loaded and frozen')

    # ========== 6. 加载奖励模型（可选） ==========
    reward_model = None
    reward_tokenizer = None
    if args.reward_model_path:
        Logger(f'Loading reward model from {args.reward_model_path}...')
        try:
            reward_model = AutoModel.from_pretrained(
                args.reward_model_path,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(args.device)
            reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
            reward_model.eval()
            for param in reward_model.parameters():
                param.requires_grad = False
            Logger('Reward model loaded')
        except Exception as e:
            Logger(f'Failed to load reward model: {e}')
            Logger('Proceeding without external reward model')
            reward_model = None
            reward_tokenizer = None

    # ========== 7. 定义 Actor 模型 ==========
    if training_state is not None:
        actor_model, tokenizer, _ = load_checkpoint(resume_path, device=args.device, load_training_state=False)
        Logger('Actor model and tokenizer loaded from resume checkpoint')
    elif args.from_weight != 'none':
        if os.path.isdir(args.from_weight):
            import glob
            checkpoint_pattern = os.path.join(args.from_weight, "checkpoint_*")
            checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]

            if checkpoints:
                checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
                latest_checkpoint = checkpoints[-1]
                Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(latest_checkpoint)}')
                actor_model, tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
                Logger(f'Actor model and tokenizer loaded from {latest_checkpoint}')
            else:
                actor_model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
                Logger(f'Actor model and tokenizer loaded from {args.from_weight}')
        else:
            actor_model = VerMindForCausalLM(lm_config)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            moe_suffix = '_moe' if lm_config.use_moe else ''
            weight_path = f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if os.path.exists(weight_path):
                weights = torch.load(weight_path, map_location=args.device)
                if isinstance(weights, dict) and 'model' in weights:
                    actor_model.load_state_dict(weights['model'], strict=False)
                else:
                    actor_model.load_state_dict(weights, strict=False)
                Logger(f'Actor model weights loaded from {weight_path}')
            actor_model = actor_model.to(args.device)
    else:
        actor_model = VerMindForCausalLM(lm_config)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        actor_model = actor_model.to(args.device)
        Logger('Actor model initialized from scratch')

    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled for actor')

    # ========== 8. 数据加载器和优化器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)

    # 学习率调度器
    total_steps = args.epochs * (len(train_ds) // (args.batch_size * args.accumulation_steps * (dist.get_world_size() if dist.is_initialized() else 1)))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, total_steps, 1.0, args.warmup_ratio)
    )

    # ========== 9. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if training_state:
        optimizer.load_state_dict(training_state['optimizer'])
        scaler.load_state_dict(training_state['scaler'])
        start_epoch = training_state['epoch']
        start_step = training_state.get('step', 0)

    # ========== 10. DDP 包装模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])

        ref_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        ref_model = DistributedDataParallel(ref_model, device_ids=[local_rank])

    # ========== 11. 确定基础保存路径 ==========
    moe_suffix = '_moe' if lm_config.use_moe else ''
    original_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path determined: {os.path.basename(base_save_path)}')

    # ========== 12. 开始训练 ==========
    Logger(f'Starting GRPO training: {args.epochs} epochs, batch_size={args.batch_size}, num_generations={args.num_generations}')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        total_iters = len(loader) + skip if skip > 0 else len(loader)

        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前 {start_step} 个 step，从 step {start_step + 1} 开始')

        grpo_train_epoch(
            epoch=epoch,
            loader=loader,
            iters=total_iters,
            model=actor_model,
            ref_model=ref_model,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            tokenizer=tokenizer,
            lm_config=lm_config,
            args=args,
            autocast_ctx=autocast_ctx,
            start_step=skip,
            swanlab=swanlab_run,
            base_save_path=base_save_path
        )

        # 重置 start_step
        start_step = 0

    # ========== 13. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()

    Logger('GRPO training completed!')
