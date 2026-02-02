#!/usr/bin/env python3
"""
VerMind-V Web Demo
基于 Gradio 的视觉语言模型交互界面
"""

import os
import sys
import argparse
import warnings
import base64
import inspect
from pathlib import Path
from threading import Thread
from queue import Queue

import torch
from PIL import Image
from transformers import AutoTokenizer, TextIteratorStreamer

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vermind_models import VerMindVLM, VLMConfig

warnings.filterwarnings('ignore')

# 全局变量
model = None
tokenizer = None
preprocess = None
vision_model = None
lm_config = None
args = None


def init_model(model_path, device='cuda'):
    """初始化 VerMind-V 模型"""
    global model, tokenizer, preprocess, vision_model, lm_config
    
    print(f"📦 正在加载模型: {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 加载模型
    model = VerMindVLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = model.to(device).eval()
    
    # 获取视觉模型和预处理器
    vision_model = model.vision_encoder
    preprocess = model.processor
    
    # 获取配置
    lm_config = model.params
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ 模型加载完成")
    print(f"📊 可训练参数量: {total_params / 1e6:.2f}M")
    print(f"🖥️  设备: {device}\n")
    
    return model, tokenizer, vision_model, preprocess


def insert_image_tokens(input_ids, image_token_ids, image_ids):
    """将 <image> 的 tokenizer 输出替换为 image_ids"""
    new_input_ids = []
    i = 0
    while i < len(input_ids):
        if input_ids[i:i+len(image_token_ids)] == image_token_ids:
            new_input_ids.extend(image_ids)
            i += len(image_token_ids)
        else:
            new_input_ids.append(input_ids[i])
            i += 1
    return new_input_ids


def generate_response(image, prompt, temperature=0.7, top_p=0.85, max_new_tokens=512):
    """生成回复，支持流式输出"""
    global model, tokenizer, preprocess, vision_model, lm_config, args
    
    device = args.device
    
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
    input_ids_list = inputs.input_ids[0].tolist()
    
    # 获取 <image> 的 token ids 并替换为 image_ids
    image_token_ids = tokenizer("<image>", add_special_tokens=False).input_ids
    image_ids = lm_config.image_ids
    input_ids_list = insert_image_tokens(input_ids_list, image_token_ids, image_ids)
    
    # 转回 tensor
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    
    # 处理图像
    if image is not None:
        image_pil = Image.open(image).convert('RGB')
        pixel_values = VerMindVLM.image2tensor(image_pil, preprocess)
        pixel_values = pixel_values.unsqueeze(0).to(device)
    else:
        pixel_values = None
    
    # 创建流式生成器
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'do_sample': True,
        'top_p': top_p,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'streamer': streamer
    }
    
    # 在新线程中生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 流式输出
    for new_text in streamer:
        yield new_text
    
    thread.join()


def create_demo():
    """创建 Gradio 界面"""
    
    # 读取 SVG logo（优先使用彩色版本）
    logo_path = Path(__file__).parent.parent / "docs" / "assets" / "vermind_logo_color.svg"
    if not logo_path.exists():
        logo_path = Path(__file__).parent.parent / "docs" / "assets" / "vermind_logo.svg"
    logo_html = ''
    if logo_path.exists():
        try:
            with open(logo_path, 'r', encoding='utf-8') as f:
                logo_svg = f.read()
            # 提取 SVG 内容用于内嵌显示
            import re
            svg_match = re.search(r'(<svg.*?</svg>)', logo_svg, re.DOTALL)
            if svg_match:
                logo_html = svg_match.group(1)
                # 调整大小
                logo_html = re.sub(r'width="[^"]*"', 'width="60"', logo_html)
                logo_html = re.sub(r'height="[^"]*"', 'height="60"', logo_html)
        except Exception as e:
            print(f"⚠️  Logo 加载失败: {e}")
            logo_html = ''
    
    import gradio as gr
    
    with gr.Blocks(title="VerMind-V", css="""
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .logo-container { display: flex; align-items: center; justify-content: center; gap: 15px; }
        .chat-container { height: 600px; }
        .input-container { margin-top: 10px; }
    """) as demo:
        
        # 标题区域
        gr.HTML(f"""
            <div class="header">
                <div class="logo-container">
                    {logo_html}
                    <span style="font-size: 36px; font-weight: bold; 
                                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        VerMind-V
                    </span>
                </div>
                <p style="color: #666; margin-top: 10px;">多模态视觉语言模型对话系统</p>
            </div>
        """)
        
        # 状态变量
        current_image = gr.State(value=None)
        
        with gr.Row():
            # 左侧：图片上传和参数设置
            with gr.Column(scale=3):
                with gr.Blocks():
                    # 图片上传
                    image_input = gr.Image(
                        type="filepath",
                        label="📷 上传图片",
                        height=400
                    )
                    
                    # 参数设置
                    with gr.Group():
                        gr.Markdown("### ⚙️ 生成参数")
                        temperature_slider = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=1.5,
                            value=0.7,
                            step=0.1,
                            info="控制生成随机性，越大越随机"
                        )
                        top_p_slider = gr.Slider(
                            label="Top-P",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.85,
                            step=0.05,
                            info="Nucleus 采样阈值"
                        )
                        max_tokens_slider = gr.Slider(
                            label="Max New Tokens",
                            minimum=64,
                            maximum=2048,
                            value=512,
                            step=64,
                            info="最大生成长度"
                        )
                    
                    # 清空按钮
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
            
            # 右侧：对话区域
            with gr.Column(scale=5):
                chatbot_kwargs = {
                    "label": "💬 对话",
                    "height": 550,
                    "avatar_images": (None, None),
                }
                # gradio 版本兼容：旧版不支持 bubble_full_width
                if "bubble_full_width" in inspect.signature(gr.Chatbot.__init__).parameters:
                    chatbot_kwargs["bubble_full_width"] = False
                # gradio 新版默认 messages，显式使用 tuples 兼容旧格式
                if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
                    chatbot_kwargs["type"] = "tuples"
                chatbot = gr.Chatbot(**chatbot_kwargs)
                chatbot_format = getattr(chatbot, "type", None) or "tuples"
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="请输入你的问题...",
                        show_label=False,
                        container=False,
                        scale=8
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                
                # 示例问题
                gr.Examples(
                    examples=[
                        "描述一下这张图片的内容",
                        "这张图片里有什么？",
                        "图片中的主要元素是什么？",
                        "请详细描述图片中的场景",
                        "这张图片传达了什么情感或氛围？"
                    ],
                    inputs=msg_input,
                    label="💡 示例问题"
                )
        
        # 交互逻辑
        def _ensure_history(history):
            return history or []

        def user_message(message, history, image_path):
            """处理用户消息"""
            if not message.strip():
                return "", history, image_path
            
            history = _ensure_history(history)

            if image_path is None:
                if chatbot_format == "messages":
                    history = history + [
                        {"role": "user", "content": "请先上传图片"},
                        {"role": "assistant", "content": "❌ 请先上传一张图片再进行对话"},
                    ]
                else:
                    history = history + [("请先上传图片", "❌ 请先上传一张图片再进行对话")]
                return "", history, image_path
            
            # 显示用户消息（包含图片）
            image_html = f'<img src="file/{image_path}" style="max-width:100px;max-height:100px;border-radius:8px;margin-bottom:5px;"><br>'
            if chatbot_format == "messages":
                history = history + [{"role": "user", "content": f"{image_html}{message}"}]
            else:
                history = history + [(f"{image_html}{message}", None)]
            return "", history, image_path
        
        def bot_response(history, image_path, temperature, top_p, max_tokens):
            """生成机器人回复"""
            history = _ensure_history(history)
            if not history:
                return history

            if chatbot_format == "messages":
                last_msg = history[-1]
                if last_msg.get("role") == "assistant" and last_msg.get("content"):
                    return history
                if last_msg.get("role") == "assistant":
                    user_message_text = history[-2]["content"] if len(history) >= 2 else ""
                else:
                    user_message_text = last_msg.get("content", "")
                    history.append({"role": "assistant", "content": ""})
            else:
                if history[-1][1] is not None:
                    return history
                user_message_text = history[-1][0]

            # 提取纯文本（去掉图片 HTML）
            import re
            text_only = re.sub(r'<img[^>]*>', '', user_message_text).strip()
            
            # 生成回复
            response = ""
            for new_text in generate_response(image_path, text_only, temperature, top_p, max_tokens):
                response += new_text
                if chatbot_format == "messages":
                    history[-1]["content"] = response
                else:
                    history[-1] = (user_message_text, response)
                yield history
        
        def clear_chat():
            """清空对话"""
            return None, []
        
        # 绑定事件
        msg_input.submit(
            user_message,
            [msg_input, chatbot, image_input],
            [msg_input, chatbot, current_image],
            queue=False
        ).then(
            bot_response,
            [chatbot, current_image, temperature_slider, top_p_slider, max_tokens_slider],
            chatbot
        )
        
        submit_btn.click(
            user_message,
            [msg_input, chatbot, image_input],
            [msg_input, chatbot, current_image],
            queue=False
        ).then(
            bot_response,
            [chatbot, current_image, temperature_slider, top_p_slider, max_tokens_slider],
            chatbot
        )
        
        clear_btn.click(
            clear_chat,
            None,
            [image_input, chatbot],
            queue=False
        )
        
        # 图片更新时更新状态
        image_input.change(
            lambda x: x,
            inputs=image_input,
            outputs=current_image
        )
    
    return demo


def main():
    global args
    
    parser = argparse.ArgumentParser(description="VerMind-V Web Demo")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="模型路径（包含 config.json 和模型权重的目录）"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="运行设备 (cuda/cpu)"
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help="服务器主机地址"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help="服务器端口"
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help="创建公开分享链接（Gradio Tunnel）"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 初始化模型
    init_model(args.model_path, args.device)
    
    # 创建并启动界面
    demo = create_demo()
    
    print(f"\n🚀 启动 Web Demo...")
    print(f"🔗 访问地址: http://{args.host}:{args.port}")
    if args.share:
        print("🌐 公开分享链接已启用")
    
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
