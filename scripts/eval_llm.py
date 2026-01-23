#!/usr/bin/env python3
"""
VerMind 模型推理与对话脚本
使用 OpenAI 兼容接口进行测试（本地 8000 端口）
"""

import time
import argparse
import warnings

from openai import OpenAI

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description="VerMind 模型推理与对话（使用 OpenAI 接口）")
    parser.add_argument(
        '--api_base',
        default='http://localhost:8000/v1',
        type=str,
        help="OpenAI API 基础 URL"
    )
    parser.add_argument(
        '--api_key',
        default='sk-no-key-required',
        type=str,
        help="API Key（本地服务通常不需要）"
    )
    parser.add_argument(
        '--model',
        default='vermind',
        type=str,
        help="模型名称"
    )
    parser.add_argument(
        '--max_tokens',
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
        help="显示生成速度（tokens/s）"
    )
    parser.add_argument(
        '--stream',
        default=1,
        type=int,
        choices=[0, 1],
        help="是否使用流式输出（1=使用，0=不使用）"
    )
    args = parser.parse_args()
    
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )
    print(f"🔗 连接到 API: {args.api_base}")
    print(f"📦 使用模型: {args.model}\n")
    
    # 预设测试提示词
    prompts = [
        '写一个计算斐波那契数列的代码',
        '写一个快速排序的代码',
        '你有什么特长？',
        '为什么天空是蓝色的？',
        '中国有哪些比较好的大学',
        '你知道光速是多少吗?',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门？',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食',
        '你是谁？',
        '你叫什么名字',
        '你是chatgpt吗？',
        '你是谁开发的？'
    ]
    
    # 初始化对话历史
    conversation = []
    
    # 选择输入模式
    print("\n" + "=" * 60)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n请选择: '))
    print("=" * 60 + "\n")
    
    # 创建提示词迭代器
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    
    for prompt in prompt_iter:
        if input_mode == 0:
            print(f'💬: {prompt}')
        
        # 管理对话历史
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        
        # 生成回复
        print('🤖: ', end='', flush=True)
        st = time.time()
        response_text = ""
        gen_tokens = 0
        
        try:
            if args.stream:
                # 流式输出
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=conversation,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        response_text += content
                    # 尝试从 usage 中获取 token 计数（通常在最后一个 chunk 中）
                    if hasattr(chunk, 'usage') and chunk.usage:
                        gen_tokens = chunk.usage.completion_tokens if hasattr(chunk.usage, 'completion_tokens') else 0
            else:
                # 非流式输出
                response = client.chat.completions.create(
                    model=args.model,
                    messages=conversation,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=False
                )
                response_text = response.choices[0].message.content
                print(response_text, end='', flush=True)
                gen_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            continue
        
        # 添加到对话历史
        conversation.append({"role": "assistant", "content": response_text})
        
        # 显示速度
        elapsed = time.time() - st
        if args.show_speed and gen_tokens > 0:
            print(f'\n[Speed]: {gen_tokens / elapsed:.2f} tokens/s ({gen_tokens} tokens in {elapsed:.2f}s)\n')
        elif args.show_speed:
            print(f'\n[Time]: {elapsed:.2f}s\n')
        else:
            print('\n')
        
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
