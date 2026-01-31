#!/usr/bin/env python3
"""
VerMind-V 视觉语言模型推理与对话脚本
支持本地推理和 vLLM API 两种模式
"""

import os
import sys
import time
import argparse
import warnings
import base64
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image

warnings.filterwarnings('ignore')


def encode_image_to_base64(image_path):
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


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


def load_model_local(model_path, device='cuda'):
    """本地加载 VerMind-V 模型"""
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from vermind_models import VerMindVLM
    
    print(f"📦 加载本地模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = VerMindVLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = model.to(device).eval()
    
    print(f"✅ 模型加载完成，设备: {device}\n")
    return model, tokenizer


def generate_response_local(model, tokenizer, image, prompt, max_length=512, temperature=0.7, device='cuda'):
    """本地生成回复"""
    # 构建消息
    messages = [
        {"role": "user", "content": f"<image>\n{prompt}"}
    ]
    
    # 应用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # 处理图像
    pixel_values = model.image2tensor(image, model.processor)
    pixel_values = pixel_values.unsqueeze(0).to(device)
    
    # 生成
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 提取助手回复
    if "assistant" in generated_text.lower():
        parts = generated_text.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
        else:
            response = generated_text[len(text):].strip()
    else:
        response = generated_text[len(text):].strip()
    
    return response


def generate_response_api(client, model, image_path, prompt, max_tokens=512, temperature=0.7):
    """通过 API 生成回复"""
    base64_image = encode_image_to_base64(image_path)
    
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
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.85
    )
    
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="VerMind-V VLM 推理与对话")
    
    # 模式选择
    parser.add_argument(
        '--use_api',
        action='store_true',
        help="使用 vLLM API 模式（默认使用本地推理）"
    )
    
    # 本地推理参数
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="本地模型路径（本地推理模式必需）"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="推理设备（本地模式）"
    )
    
    # API 参数
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
        help="API 模型名称"
    )
    
    # 通用参数
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
        '--max_length',
        type=int,
        default=512,
        help="最大生成长度"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
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
    
    # 根据模式初始化
    if args.use_api:
        # API 模式
        from openai import OpenAI
        client = OpenAI(api_key=args.api_key, base_url=args.api_base)
        print(f"🔗 API 模式: {args.api_base}")
        print(f"📦 模型: {args.model}\n")
        generate_fn = lambda img_path, prompt: generate_response_api(
            client, args.model, img_path, prompt, args.max_length, args.temperature
        )
    else:
        # 本地模式
        if not args.model_path:
            print("❌ 本地推理模式需要指定 --model_path")
            return
        
        model, tokenizer = load_model_local(args.model_path, args.device)
        print(f"📍 本地模式: {args.model_path}\n")
        
        def generate_fn(img_path, prompt):
            image = Image.open(img_path).convert('RGB')
            return generate_response_local(
                model, tokenizer, image, prompt,
                args.max_length, args.temperature, args.device
            )
    
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
                
                # 自动测试
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
                            response = generate_fn(image_path, test_prompt)
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
                    response = generate_fn(image_path, prompt)
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
