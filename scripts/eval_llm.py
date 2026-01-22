#!/usr/bin/env python3
"""
VerMind 模型推理与对话脚本
支持从 checkpoint 目录加载模型，支持 chat template 和流式输出
"""

import time
import argparse
import random
import warnings
import os
import sys
import glob

import torch
from transformers import AutoTokenizer, TextStreamer

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.utils import load_checkpoint, setup_seed

warnings.filterwarnings('ignore')


def find_latest_checkpoint(base_path):
    """
    从基础路径中找到最新的 checkpoint
    
    Args:
        base_path: 基础路径，如 /root/vermind/output/pretrain/pretrain_768
    
    Returns:
        最新的 checkpoint 路径，如果没找到返回 None
    """
    if not os.path.isdir(base_path):
        return None
    
    checkpoint_pattern = os.path.join(base_path, "checkpoint_*")
    checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
    
    if checkpoints:
        checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
        return checkpoints[-1]
    return None


def init_model(args):
    """
    初始化模型和 tokenizer
    
    Args:
        args: 命令行参数
    
    Returns:
        model, tokenizer
    """
    model_path = args.load_from
    
    # 如果路径是基础路径（包含 checkpoint_* 子目录），自动找最新的
    if os.path.isdir(model_path):
        checkpoint_pattern = os.path.join(model_path, "checkpoint_*")
        checkpoints = [p for p in glob.glob(checkpoint_pattern) if os.path.isdir(p)]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("checkpoint_", "")))
            latest_checkpoint = checkpoints[-1]
            print(f"📦 找到 {len(checkpoints)} 个 checkpoint，使用最新的: {os.path.basename(latest_checkpoint)}")
            model_path = latest_checkpoint
    
    # 加载模型和 tokenizer
    print(f"📥 正在加载模型: {model_path}")
    model, tokenizer, _ = load_checkpoint(model_path, device=args.device, load_training_state=False)
    print(f"✅ 模型加载完成")
    
    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总计 {total_params / 1e6:.2f}M, 可训练 {trainable_params / 1e6:.2f}M")
    
    return model.eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description="VerMind 模型推理与对话")
    parser.add_argument(
        '--load_from',
        default='/root/vermind/checkpoint_4000',
        type=str,
        help="模型加载路径（checkpoint 目录或包含 checkpoint_* 的基础路径）"
    )
    parser.add_argument(
        '--max_new_tokens',
        default=2048,
        type=int,
        help="最大生成长度"
    )
    parser.add_argument(
        '--temperature',
        default=0.85,
        type=float,
        help="生成温度，控制随机性（0-1，越大越随机）"
    )
    parser.add_argument(
        '--top_p',
        default=0.85,
        type=float,
        help="nucleus 采样阈值（0-1）"
    )
    parser.add_argument(
        '--repetition_penalty',
        default=1.0,
        type=float,
        help="重复惩罚系数（>1.0 减少重复）"
    )
    parser.add_argument(
        '--historys',
        default=0,
        type=int,
        help="携带历史对话轮数（需为偶数，0表示不携带历史）"
    )
    parser.add_argument(
        '--show_speed',
        default=1,
        type=int,
        choices=[0, 1],
        help="显示 decode 速度（tokens/s）"
    )
    parser.add_argument(
        '--use_chat_template',
        default=1,
        type=int,
        choices=[0, 1],
        help="是否使用 chat template（1=使用，0=不使用，直接拼接 prompt）"
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str,
        help="运行设备"
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help="随机种子（None 表示随机）"
    )
    args = parser.parse_args()
    
    # 预设测试提示词
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的？',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门？',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    # 初始化模型
    conversation = []
    model, tokenizer = init_model(args)
    
    # 选择输入模式
    print("\n" + "=" * 60)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n请选择: '))
    print("=" * 60 + "\n")
    
    # 设置流式输出
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 创建提示词迭代器
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    for prompt in prompt_iter:
        # 设置随机种子
        if args.seed is not None:
            setup_seed(args.seed)
        else:
            setup_seed(random.randint(0, 2048))
        
        if input_mode == 0:
            print(f'💬: {prompt}')
        
        # 管理对话历史
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        
        # 准备输入
        if args.use_chat_template:
            # 使用 chat template
            try:
                inputs_text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"⚠️  Chat template 应用失败，使用简单拼接: {e}")
                inputs_text = tokenizer.bos_token + prompt
        else:
            # 不使用 chat template，直接拼接
            inputs_text = tokenizer.bos_token + prompt if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token else prompt
        
        # Tokenize
        inputs = tokenizer(
            inputs_text,
            return_tensors="pt",
            truncation=True,
            max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 32768
        ).to(args.device)
        
        # 生成回复
        print('🤖: ', end='', flush=True)
        st = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty
            )
        
        # 解码回复
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        
        # 添加到对话历史
        conversation.append({"role": "assistant", "content": response})
        
        # 显示速度
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        elapsed = time.time() - st
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / elapsed:.2f} tokens/s ({gen_tokens} tokens in {elapsed:.2f}s)\n')
        else:
            print('\n')
        
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
