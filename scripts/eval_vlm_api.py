#!/usr/bin/env python3
"""
VerMind-V 视觉语言模型推理脚本 (vLLM API 版本)
通过 OpenAI 兼容接口调用 vLLM 服务进行图像理解

注意：vLLM 对 VLM 的多模态支持有限，此脚本使用 base64 编码图像
"""

import os
import sys
import time
import argparse
import warnings
import base64
from pathlib import Path
from io import BytesIO

from openai import OpenAI
from PIL import Image

warnings.filterwarnings('ignore')


def encode_image_to_base64(image_path):
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_image_to_base64_pil(image):
    """将 PIL Image 编码为 base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def list_images(image_dir):
    """列出评估图像目录中的所有图片"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ 图片目录不存在: {image_dir}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    images = []
    
    for f in sorted(image_dir.iterdir()):
        if f.suffix.lower() in image_extensions:
            images.append(f)
    
    return images


def generate_response(client, model, image_path, prompt, max_tokens=512, temperature=0.7):
    """通过 API 生成回复"""
    # 将图片编码为 base64
    base64_image = encode_image_to_base64(image_path)
    
    # 构建消息（OpenAI 视觉格式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    # 调用 API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.85
    )
    
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="VerMind-V VLM 推理 (vLLM API)")
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
        help="API Key"
    )
    parser.add_argument(
        '--model',
        default='vermind-v',
        type=str,
        help="模型名称"
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='./dataset/eval_images',
        help="评估图片目录"
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help="指定单张图片路径"
    )
    parser.add_argument(
        '--max_tokens',
        default=512,
        type=int,
        help="最大生成长度"
    )
    parser.add_argument(
        '--temperature',
        default=0.7,
        type=float,
        help="生成温度"
    )
    parser.add_argument(
        '--show_speed',
        default=1,
        type=int,
        choices=[0, 1],
        help="显示生成速度"
    )
    args = parser.parse_args()
    
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )
    print(f"🔗 连接到 API: {args.api_base}")
    print(f"📦 使用模型: {args.model}\n")
    
    # 准备图片列表
    if args.image:
        images = [Path(args.image)]
    else:
        images = list_images(args.image_dir)
    
    if not images:
        print("❌ 没有找到图片")
        return
    
    print(f"🖼️  找到 {len(images)} 张图片\n")
    print("=" * 60)
    
    for i, img_path in enumerate(images):
        print(f"[{i}] {img_path.name}")
    print("=" * 60 + "\n")
    
    # 交互模式
    while True:
        try:
            choice = input(f"选择图片 [0-{len(images)-1}] 或 'q' 退出: ").strip()
            if choice.lower() == 'q':
                break
            
            try:
                img_idx = int(choice)
                if img_idx < 0 or img_idx >= len(images):
                    print(f"❌ 无效选择")
                    continue
            except ValueError:
                print("❌ 无效输入")
                continue
            
            image_path = images[img_idx]
            print(f"\n📷 图片: {image_path.name}")
            
            # 对话循环
            print("\n💡 提示: 输入问题开始对话，输入 'next' 切换图片，输入 'exit' 退出\n")
            
            while True:
                prompt = input('💬: ').strip()
                
                if prompt.lower() == 'exit':
                    return
                if prompt.lower() == 'next':
                    break
                if not prompt:
                    continue
                
                # 添加预设测试
                if prompt == 'test':
                    test_prompts = [
                        '描述这张图片',
                        '这张图片里有什么？',
                        '请详细描述图片中的内容',
                        '这张图片的主要元素是什么？'
                    ]
                    print(f"\n📝 自动测试 {len(test_prompts)} 个提示...\n")
                    for i, test_prompt in enumerate(test_prompts):
                        print(f"[{i+1}/{len(test_prompts)}] 💬: {test_prompt}")
                        print('🤖: ', end='', flush=True)
                        
                        try:
                            st = time.time()
                            response = generate_response(
                                client, args.model, image_path, test_prompt,
                                args.max_tokens, args.temperature
                            )
                            elapsed = time.time() - st
                            
                            print(response)
                            if args.show_speed:
                                print(f'\n[Time]: {elapsed:.2f}s')
                        except Exception as e:
                            print(f"❌ 错误: {e}")
                        
                        print("-" * 40 + "\n")
                    continue
                
                # 生成回复
                print('🤖: ', end='', flush=True)
                st = time.time()
                
                try:
                    response = generate_response(
                        client, args.model, image_path, prompt,
                        args.max_tokens, args.temperature
                    )
                    print(response)
                    
                    elapsed = time.time() - st
                    if args.show_speed:
                        print(f'\n[Time]: {elapsed:.2f}s')
                    
                    print("\n" + "-" * 60 + "\n")
                    
                except Exception as e:
                    print(f"\n❌ 生成错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
