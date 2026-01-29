"""
PPO (Proximal Policy Optimization) 训练脚本
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
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from vermind_models.config import VerMindConfig
from vermind_models.models.modeling_vermind import VerMindForCausalLM, VerMindModel
from data_loader.rlaif_dataset import RLAIFDataset
from utils import (
    get_lr, init_distributed_mode, setup_seed, Logger, is_main_process, SkipBatchSampler,
    save_checkpoint, load_checkpoint, resume_training, get_base_save_path
)
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')


class CriticModel(nn.Module):
    """
    Critic模型，用于估计状态价值 V(s)
    基于VerMindModel的隐藏状态输出一个标量值
    """
    def __init__(self, config: VerMindConfig):
        super().__init__()
        self.config = config
        self.model = VerMindModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 获取隐藏状态
        hidden_states, _, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 输出价值估计 [B, seq_len]
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer, device, reasoning=0):
    """
    计算奖励值

    Args:
        prompts: 提示文本列表
        responses: 响应文本列表
        reward_model: 奖励模型
        reward_tokenizer: 奖励模型的tokenizer
        device: 计算设备
        reasoning: 是否为推理模式（增加格式奖励）

    Returns:
        rewards: 奖励张量 [B]
    """
    def reasoning_model_reward(rewards):
        """计算推理模型的格式奖励"""
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards = rewards + torch.tensor(format_rewards, device=device)

        # 2. 标记奖励（防止严格奖励稀疏）
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
            for prompt, response in zip(prompts, responses):
                # 解析 prompt 中的对话历史
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                tmp_chat = messages + [{"role": "assistant", "content": response}]

                try:
                    score = reward_model.get_score(reward_tokenizer, tmp_chat)
                except Exception:
                    # 如果 reward model 不支持 get_score，尝试其他方式
                    score = 0.0

                scale = 3.0
                score = max(min(score, scale), -scale)

                # 当 reasoning=1 时，额外计算 <answer> 内容的奖励
                if reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        try:
                            answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                            answer_score = max(min(answer_score, scale), -scale)
                            score = score * 0.4 + answer_score * 0.6
                        except Exception:
                            pass

                reward_model_scores.append(score)

            reward_model_scores = torch.tensor(reward_model_scores, device=device)
            rewards = rewards + reward_model_scores

    return rewards


def logits_to_log_probs(logits, labels):
    """将 logits 转换为 log probabilities"""
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp_min(0)
    log_probs_per_token = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_per_token


def train_epoch(epoch, loader, iters, actor_model, critic_model, old_actor_model, ref_model,
                actor_optimizer, critic_optimizer, scaler, tokenizer, lm_config, args,
                autocast_ctx, reward_model=None, reward_tokenizer=None, start_step=0,
                swanlab=None, base_save_path=None):
    """
    PPO 训练一个 epoch
    """
    actor_model.train()
    critic_model.train()
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B

        # 编码 prompt
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=args.max_seq_len, padding_side="left").to(args.device)
        prompt_length = enc.input_ids.shape[1]

        # 使用 actor 生成响应
        with torch.no_grad():
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )  # [B, P+R]

        # 解码响应文本
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

        # 计算奖励
        rewards = calculate_rewards(
            prompts, responses_text, reward_model, reward_tokenizer,
            args.device, args.reasoning
        )  # [B]

        # 计算学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for param_group in actor_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in critic_optimizer.param_groups:
            param_group['lr'] = lr * args.critic_lr_ratio

        # 计算价值估计
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]

        raw_critic = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        values_seq = raw_critic(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]

        # 获取序列最后一个有效位置的价值
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]

        # 计算优势
        advantages = rewards - values.detach()  # [B]
        # 标准化优势
        if args.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with autocast_ctx:
            # 计算当前 actor 的 log prob
            actor_outputs = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = actor_outputs.logits  # [B, P+R, V]
            aux_loss = actor_outputs.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

            labels = gen_out[:, 1:].clone()  # [B, P+R-1]
            logp_tokens = logits_to_log_probs(logits[:, :-1], labels)  # [B, P+R-1]

            # 创建响应部分的 mask
            seq_len = gen_out.size(1) - 1
            resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
            final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]

            actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

            # 计算 old actor 的 log prob
            with torch.no_grad():
                old_actor_outputs = old_actor_model(input_ids=gen_out, attention_mask=full_mask)
                old_logits = old_actor_outputs.logits
                old_logp_tokens = logits_to_log_probs(old_logits[:, :-1], labels)
                old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]

                # 计算 ref model 的 log prob（用于 KL 惩罚）
                ref_outputs = ref_model(input_ids=gen_out, attention_mask=full_mask)
                ref_logits = ref_outputs.logits
                ref_logp_tokens = logits_to_log_probs(ref_logits[:, :-1], labels)
                ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

            # 计算 KL 散度
            kl = (actor_logp - old_logp).mean()  # 与旧策略的 KL
            kl_ref = (actor_logp - ref_logp).mean()  # 与参考模型的 KL

            # PPO 裁剪目标
            ratio = torch.exp(actor_logp - old_logp)  # [B]
            surr1 = ratio * advantages  # [B]
            surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(values, rewards)

            # 总损失
            loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(actor_optimizer)
            scaler.unscale_(critic_optimizer)
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            scaler.step(actor_optimizer)
            scaler.step(critic_optimizer)
            scaler.update()
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)

        # 更新 old actor（定期同步）
        if (step + 1) % args.update_old_actor_freq == 0:
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            state_dict = raw_actor.state_dict()
            old_actor_model.load_state_dict({k: v.clone() for k, v in state_dict.items()})

        # 日志记录
        if (step % args.log_interval == 0 or step == iters - 1) and is_main_process():
            spend_time = time.time() - start_time

            # 计算平均响应长度
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            current_lr = actor_optimizer.param_groups[0]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Aux Loss: {current_aux_loss:.4f}, Reward: {reward_val:.4f}, "
                   f"KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Len: {avg_len_val:.2f}, LR: {current_lr:.2e}, ETA: {eta_min:.1f}min")

            if swanlab:
                swanlab.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr,
                })

        # 保存 checkpoint
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            critic_model.eval()

            # 获取 raw critic model
            raw_critic = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            raw_critic = getattr(raw_critic, '_orig_mod', raw_critic)

            # 使用在训练开始前确定的基础路径
            save_checkpoint(
                model=actor_model,
                tokenizer=tokenizer,
                config=lm_config,
                save_path=base_save_path,
                optimizer=actor_optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                swanlab=swanlab,
                max_checkpoints=args.max_checkpoints,
                save_interval=args.save_interval,
                steps_per_epoch=iters,
                critic_model=raw_critic,
                critic_optimizer=critic_optimizer,
            )

            actor_model.train()
            critic_model.train()

        # 清理内存
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp
        del ref_logits, ref_logp, kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerMind PPO Training")
    # 基础训练参数
    parser.add_argument("--save_dir", type=str, default="./output/ppo", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="actor 初始学习率")
    parser.add_argument("--critic_lr_ratio", type=float, default=1.0, help="critic 学习率与 actor 的比例")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热步数比例（0.0-1.0）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
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
    parser.add_argument('--from_weight', default='none', type=str, help="actor 初始权重路径")
    parser.add_argument('--ref_weight', type=str, required=True, help="参考模型权重路径")
    parser.add_argument('--reward_model_path', type=str, default='', help="奖励模型路径（留空则不使用外部奖励模型）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")

    # PPO 特定参数
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO 裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="价值函数损失系数")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL 惩罚系数")
    parser.add_argument("--update_old_actor_freq", type=int, default=10, help="更新 old actor 的频率（步数）")
    parser.add_argument("--normalize_advantage", type=int, default=1, choices=[0, 1], help="是否标准化优势")
    parser.add_argument("--temperature", type=float, default=0.8, help="生成温度")
    parser.add_argument("--reasoning", type=int, default=0, choices=[0, 1], help="是否训练推理模型（增加格式奖励）")

    # 其他
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用swanlab")
    parser.add_argument("--swanlab_project", type=str, default="VerMind-PPO", help="swanlab项目名")
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
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast('cuda', dtype=dtype)

    # ========== 4. 配置 swanlab ==========
    swanlab_run = None
    if args.use_swanlab and is_main_process():
        swanlab_id = training_state.get('swanlab_id') if training_state else None
        resume = 'must' if swanlab_id else None
        swanlab_run_name = f"VerMind-PPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        swanlab.init(project=args.swanlab_project, name=swanlab_run_name, id=swanlab_id, resume=resume)
        swanlab_run = swanlab.get_run()

    # ========== 5. 加载参考模型 (ref_model) ==========
    Logger('Loading reference model...')
    ref_tokenizer = None
    if os.path.isdir(args.ref_weight):
        # 如果是目录，检查是否是基础路径（包含 checkpoint_* 子目录）还是具体的 checkpoint 路径
        import glob
        checkpoint_pattern = os.path.join(args.ref_weight, "checkpoint_*")
        checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]

        if checkpoints:
            # 如果是基础路径（包含多个 checkpoint），自动选择最新的
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
            latest_checkpoint = checkpoints[-1]
            Logger(f'Found {len(checkpoints)} checkpoints, using latest: {os.path.basename(latest_checkpoint)}')
            ref_model, ref_tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
            Logger(f'Reference model loaded from {latest_checkpoint}')
        else:
            # 如果是具体的 checkpoint 目录，直接加载
            ref_model, ref_tokenizer, _ = load_checkpoint(args.ref_weight, device=args.device, load_training_state=False)
            Logger(f'Reference model loaded from {args.ref_weight}')
    else:
        # 兼容旧格式：从 .pth 文件加载
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

    ref_model.eval()  # 参考模型始终处于eval模式
    for param in ref_model.parameters():
        param.requires_grad = False  # 冻结参考模型参数
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
    # 初始化模型和 tokenizer
    if training_state is not None:
        # 从 resume checkpoint 加载
        actor_model, tokenizer, _ = load_checkpoint(resume_path, device=args.device, load_training_state=False)
        Logger('Actor model and tokenizer loaded from resume checkpoint')
    elif args.from_weight != 'none':
        # 从指定权重加载
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
                actor_model, tokenizer, _ = load_checkpoint(latest_checkpoint, device=args.device, load_training_state=False)
                Logger(f'Actor model and tokenizer loaded from {latest_checkpoint}')
            else:
                # 如果是具体的 checkpoint 目录，直接加载
                actor_model, tokenizer, _ = load_checkpoint(args.from_weight, device=args.device, load_training_state=False)
                Logger(f'Actor model and tokenizer loaded from {args.from_weight}')
        else:
            # 兼容旧格式：从 .pth 文件加载
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
        # 从头开始训练
        actor_model = VerMindForCausalLM(lm_config)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        actor_model = actor_model.to(args.device)
        Logger('Actor model initialized from scratch')

    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled for actor')

    # ========== 8. 定义 Old Actor 模型（用于计算 ratio） ==========
    old_actor_model = VerMindForCausalLM(lm_config)
    raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
    raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
    old_actor_model.load_state_dict(raw_actor.state_dict())
    old_actor_model = old_actor_model.to(args.device)
    old_actor_model.eval()
    for param in old_actor_model.parameters():
        param.requires_grad = False
    Logger('Old actor model initialized')

    # ========== 9. 定义 Critic 模型 ==========
    critic_model = CriticModel(lm_config)
    # 从 actor 的 backbone 权重初始化 critic
    critic_model.model.load_state_dict(raw_actor.model.state_dict())
    critic_model = critic_model.to(args.device)

    # 如果有训练状态，加载 critic 权重
    if training_state and 'critic_model' in training_state:
        critic_model.load_state_dict(training_state['critic_model'])
        Logger('Critic model loaded from checkpoint')
    else:
        Logger('Critic model initialized from actor backbone')

    # ========== 10. 数据加载器和优化器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.learning_rate * args.critic_lr_ratio)

    # ========== 11. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if training_state:
        actor_optimizer.load_state_dict(training_state['optimizer'])
        scaler.load_state_dict(training_state['scaler'])
        if 'critic_optimizer' in training_state:
            critic_optimizer.load_state_dict(training_state['critic_optimizer'])
        start_epoch = training_state['epoch']
        start_step = training_state.get('step', 0)

    # ========== 12. DDP 包装模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])

        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])

        ref_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        ref_model = DistributedDataParallel(ref_model, device_ids=[local_rank])

    # ========== 13. 确定基础保存路径（在训练开始前确定一次） ==========
    moe_suffix = '_moe' if lm_config.use_moe else ''
    original_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}'
    base_save_path = get_base_save_path(original_save_path)
    if is_main_process():
        Logger(f'Base save path determined: {os.path.basename(base_save_path)}')

    # ========== 14. 开始训练 ==========
    Logger(f'Starting PPO training: {args.epochs} epochs, batch_size={args.batch_size}')
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

        train_epoch(
            epoch=epoch,
            loader=loader,
            iters=total_iters,
            actor_model=actor_model,
            critic_model=critic_model,
            old_actor_model=old_actor_model,
            ref_model=ref_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            lm_config=lm_config,
            args=args,
            autocast_ctx=autocast_ctx,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            start_step=skip,
            swanlab=swanlab_run,
            base_save_path=base_save_path
        )

        # 重置 start_step
        start_step = 0

    # ========== 15. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()

    Logger('PPO training completed!')
