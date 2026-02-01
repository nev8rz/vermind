#!/usr/bin/env python3
"""
上传完整的 VerMind-V (VLM) 模型到 HuggingFace

此脚本会收集以下文件上传：
1. 模型权重 (model.safetensors)
2. 配置文件 (config.json)
3. Tokenizer 文件
4. VLM 特有的模型定义文件 (modeling_vermind_v.py, configuration_vermind_v.py)
5. 生成配置和 chat template

用法:
    python scripts/upload_vlm_model.py \
        --checkpoint_path /path/to/checkpoint_10000 \
        --repo_id your_username/vermind-v-base \
        --private

需要安装:
    pip install huggingface_hub

需要设置环境变量:
    export HF_TOKEN=your_huggingface_token
"""

import os
import sys
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_vlm_files(checkpoint_path: str, temp_dir: str):
    """
    复制 VLM 模型所需的所有文件到临时目录
    """
    checkpoint_path = Path(checkpoint_path)
    temp_path = Path(temp_dir)
    
    print(f"📦 正在收集文件从: {checkpoint_path}")
    
    # 0. 验证 model.safetensors 包含 Vision Encoder 权重
    try:
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path / "model.safetensors")
        
        llm_keys = [k for k in state_dict.keys() if k.startswith('model.') or k == 'lm_head.weight']
        vision_proj_keys = [k for k in state_dict.keys() if k.startswith('vision_proj.')]
        vision_encoder_keys = [k for k in state_dict.keys() if k.startswith('vision_encoder.')]
        
        print(f"\n📊 模型权重统计:")
        print(f"  - LLM: {len(llm_keys)} keys")
        print(f"  - Vision Projection: {len(vision_proj_keys)} keys")
        print(f"  - Vision Encoder (SigLIP): {len(vision_encoder_keys)} keys")
        
        if len(vision_encoder_keys) == 0:
            print(f"\n⚠️  警告: model.safetensors 中没有 Vision Encoder 权重!")
            print(f"   模型将无法处理图像。请确保从正确的 VLM checkpoint 上传。")
        else:
            print(f"\n✅ Vision Encoder 权重已包含在 model.safetensors 中")
    except Exception as e:
        print(f"\n⚠️  无法验证权重: {e}")
    
    # 1. 复制 checkpoint 中的文件
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]
    
    for file_name in required_files:
        src = checkpoint_path / file_name
        if src.exists():
            shutil.copy2(src, temp_path / file_name)
            print(f"  ✅ {file_name}")
        else:
            print(f"  ⚠️  跳过 (不存在): {file_name}")
    
    # 2. 复制 VLM 模型定义文件
    vlm_model_files = [
        ("vermind_models/models/modeling_vermind_v.py", "modeling_vermind_v.py"),
        ("vermind_models/config/config_vlm.py", "configuration_vermind_v.py"),
    ]
    
    for src_rel, dst_name in vlm_model_files:
        src = PROJECT_ROOT / src_rel
        if src.exists():
            shutil.copy2(src, temp_path / dst_name)
            print(f"  ✅ {dst_name}")
        else:
            print(f"  ❌ 缺失关键文件: {src_rel}")
            return False
    
    # 3. 复制基础模型定义文件（VLM 继承自这些）
    base_model_files = [
        ("vermind_models/models/modeling_vermind.py", "modeling_vermind.py"),
        ("vermind_models/config/config.py", "configuration_vermind.py"),
    ]
    
    for src_rel, dst_name in base_model_files:
        src = PROJECT_ROOT / src_rel
        if src.exists():
            shutil.copy2(src, temp_path / dst_name)
            print(f"  ✅ {dst_name}")
        else:
            print(f"  ❌ 缺失关键文件: {src_rel}")
            return False
    
    # 4. 复制其他可能需要的模块
    extra_files = [
        ("vermind_models/base_module.py", "base_module.py"),
        ("vermind_models/GQA.py", "GQA.py"),
        ("vermind_models/FFN.py", "FFN.py"),
    ]
    
    for src_rel, dst_name in extra_files:
        src = PROJECT_ROOT / src_rel
        if src.exists():
            shutil.copy2(src, temp_path / dst_name)
            print(f"  ✅ {dst_name}")
    
    print(f"\n📁 临时目录: {temp_dir}")
    return True


def upload_to_hf(
    repo_id: str,
    local_path: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
):
    """上传到 HuggingFace"""
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("错误: 请先安装 huggingface_hub: pip install huggingface_hub")
        return False

    api = HfApi(token=token)
    local_path = Path(local_path)

    # 创建仓库（如果不存在）
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"✅ 仓库就绪: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  创建仓库失败: {e}")
        return False

    # 上传
    try:
        print(f"\n📤 正在上传...")
        upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or "Upload VerMind-V model",
            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store", "training_state.pt"],
        )
        print(f"✅ 上传成功!")
        print(f"🌐 模型地址: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="上传完整的 VerMind-V 模型到 HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 上传 VLM 模型（公开）
  python scripts/upload_vlm_model.py \\
      --checkpoint_path ./output/vlm_sft/vlm_sft_768/checkpoint_29753 \\
      --repo_id your_username/vermind-v-base

  # 上传为私有仓库
  python scripts/upload_vlm_model.py \\
      --checkpoint_path ./output/vlm_pretrain/vlm_pretrain_768/checkpoint_10000 \\
      --repo_id your_username/vermind-v-pretrain \\
      --private

环境变量:
  HF_TOKEN - HuggingFace API Token
        """
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="VLM checkpoint 路径 (例如: ./output/vlm_sft/vlm_sft_768/checkpoint_29753)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace 仓库 ID (格式: namespace/model_name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API Token (默认从 HF_TOKEN 环境变量读取)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="创建私有仓库"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="提交信息"
    )

    args = parser.parse_args()

    # 检查 checkpoint 路径
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ 错误: checkpoint 路径不存在: {checkpoint_path}")
        return 1
    
    if not (checkpoint_path / "model.safetensors").exists():
        print(f"❌ 错误: 未找到 model.safetensors，请确认是有效的 checkpoint 路径")
        return 1

    # 从环境变量获取 token
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ 错误: 请提供 HF Token (使用 --token 或设置 HF_TOKEN 环境变量)")
        print("   获取 Token: https://huggingface.co/settings/tokens")
        return 1

    # 创建临时目录并复制文件
    with tempfile.TemporaryDirectory() as temp_dir:
        print("=" * 60)
        print("📋 步骤 1/2: 收集模型文件")
        print("=" * 60)
        
        if not copy_vlm_files(args.checkpoint_path, temp_dir):
            print("❌ 文件收集失败")
            return 1
        
        print("\n" + "=" * 60)
        print("📤 步骤 2/2: 上传到 HuggingFace")
        print("=" * 60)
        
        success = upload_to_hf(
            repo_id=args.repo_id,
            local_path=temp_dir,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 完成!")
            print("=" * 60)
            print(f"\n你可以使用以下方式加载模型:")
            print(f"```python")
            print(f"from vermind_models import VerMindVLM")
            print(f"model = VerMindVLM.from_pretrained('{args.repo_id}')")
            print(f"```")
            return 0
        else:
            return 1


if __name__ == "__main__":
    exit(main())
