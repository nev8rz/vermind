#!/usr/bin/env python3
"""
上传 VerMind 或 VerMind-V 模型到 HuggingFace
自动从 vllm_adapter 复制完整的模型定义文件
"""

import os
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_model_files(model_path: str, temp_dir: str, model_type: str):
    """复制模型所需的所有文件到临时目录"""
    model_path = Path(model_path)
    temp_path = Path(temp_dir)
    
    print(f"📦 正在收集文件从: {model_path}")
    print(f"   模型类型: {model_type.upper()}")
    

    try:
        from safetensors.torch import load_file
        state_dict = load_file(model_path / "model.safetensors")
        
        llm_keys = [k for k in state_dict.keys() if k.startswith('model.') or k == 'lm_head.weight']
        vision_proj_keys = [k for k in state_dict.keys() if k.startswith('vision_proj.')]
        vision_encoder_keys = [k for k in state_dict.keys() if k.startswith('vision_encoder.')]
        
        print(f"\n📊 模型权重统计:")
        print(f"  - LLM: {len(llm_keys)} keys")
        
        if vision_proj_keys:
            print(f"  - Vision Projection: {len(vision_proj_keys)} keys")
        if vision_encoder_keys:
            print(f"  - Vision Encoder (SigLIP): {len(vision_encoder_keys)} keys")
        
        if model_type == 'vlm' and len(vision_encoder_keys) == 0:
            print(f"\n⚠️  警告: VLM 模型中没有 Vision Encoder 权重!")
            return False
        elif model_type == 'llm' and len(vision_encoder_keys) > 0:
            print(f"\nℹ️  注意: LLM 模型中包含 Vision Encoder 权重 (共 {len(vision_encoder_keys)} keys)")
        else:
            print(f"\n✅ 模型权重验证通过")
    except Exception as e:
        print(f"\n⚠️  无法验证权重: {e}")
    

    print(f"\n📋 复制 checkpoint 文件...")
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
        src = model_path / file_name
        if src.exists():
            shutil.copy2(src, temp_path / file_name)
            print(f"  ✅ {file_name}")
        else:
            print(f"  ⚠️  跳过 (不存在): {file_name}")
    

    config_path = temp_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if model_type == 'vlm':

            print(f"\n📋 注入 VLM auto_map 到 config.json...")
            config["auto_map"] = {
                "AutoConfig": "configuration_vermind_v.VLMConfig",
                "AutoModelForCausalLM": "modeling_vermind_v.VerMindVLM"
            }
            print(f"  ✅ auto_map 已注入")
            

            print(f"\n📋 复制 VLM 模型定义文件...")
            vlm_files = [
                ("vllm_adapter/vlm/configuration_vermind_v.py", "configuration_vermind_v.py"),
                ("vllm_adapter/vlm/modeling_vermind_v.py", "modeling_vermind_v.py"),
            ]
            
            for src_rel, dst_name in vlm_files:
                src = PROJECT_ROOT / src_rel
                if src.exists():
                    shutil.copy2(src, temp_path / dst_name)
                    print(f"  ✅ {dst_name}")
                else:
                    print(f"  ❌ 缺失关键文件: {src_rel}")
                    return False
        else:

            print(f"\n📋 复制 LLM 模型定义文件...")
            base_files = [
                ("vllm_adapter/core/configuration_vermind.py", "configuration_vermind.py"),
                ("vllm_adapter/core/modeling_vermind.py", "modeling_vermind.py"),
            ]
            
            for src_rel, dst_name in base_files:
                src = PROJECT_ROOT / src_rel
                if src.exists():
                    shutil.copy2(src, temp_path / dst_name)
                    print(f"  ✅ {dst_name}")
                else:
                    print(f"  ❌ 缺失关键文件: {src_rel}")
                    return False
        

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  ✅ config.json 已更新")
    
    print(f"\n📁 临时目录准备完成: {temp_dir}")
    return True


def delete_remote_files(repo_id: str, files: list, token: Optional[str] = None, 
                        commit_message: Optional[str] = None):
    """删除 HuggingFace 远程仓库中的文件"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("错误: 请先安装 huggingface_hub: pip install huggingface_hub")
        return False
    
    api = HfApi(token=token)
    
    try:
        print(f"\n🗑️  正在删除远程文件...")
        for file_path in files:
            try:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    token=token,
                    commit_message=commit_message or f"Delete {file_path}"
                )
                print(f"  ✅ 已删除: {file_path}")
            except Exception as e:
                if "404" in str(e):
                    print(f"  ⚠️  文件不存在: {file_path}")
                else:
                    print(f"  ❌ 删除失败 {file_path}: {e}")
        return True
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        return False


def upload_to_hf(repo_id: str, local_path: str, token: Optional[str] = None, 
                 private: bool = False, commit_message: Optional[str] = None):
    """上传到 HuggingFace"""
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("错误: 请先安装 huggingface_hub: pip install huggingface_hub")
        return False


    api = HfApi(token=token)
    local_path = Path(local_path)


    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"✅ 仓库就绪: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  创建仓库失败: {e}")

        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"\n💡 提示: 请确保已登录 huggingface-cli:")
            print(f"   huggingface-cli login")
            print(f"   或提供 --token 参数")
        return False


    try:
        print(f"\n📤 正在上传文件...")
        upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or "Upload VerMind model",
            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store", "training_state.pt"],
        )
        print(f"✅ 上传成功!")
        print(f"🌐 模型地址: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="上传 VerMind/VerMind-V 模型到 HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  python scripts/upload_hf.py \
      --model_path /root/vermind/output/dpo/dpo_768/checkpoint_1610 \
      --repo_id your_username/vermind-dpo \
      --model_type llm


  python scripts/upload_hf.py \
      --model_path /root/vermind/output/vlm_sft/vlm_sft_768/checkpoint_29753 \
      --repo_id your_username/vermind-v-sft \
      --model_type vlm
        """
    )
    
    parser.add_argument("--model_path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace 仓库 ID (格式: namespace/model_name)")
    parser.add_argument("--model_type", type=str, choices=["llm", "vlm"], required=True, help="模型类型")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API Token (默认从 HF_TOKEN 环境变量读取)")
    parser.add_argument("--private", action="store_true", help="创建私有仓库")
    parser.add_argument("--commit_message", type=str, default=None, help="提交信息")

    args = parser.parse_args()


    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 错误: 路径不存在: {model_path}")
        return 1
    
    if not (model_path / "model.safetensors").exists():
        print(f"❌ 错误: 未找到 model.safetensors")
        return 1


    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("ℹ️  未提供 token，尝试使用 huggingface-cli 已登录的凭证..."
    )


    with tempfile.TemporaryDirectory() as temp_dir:
        print("=" * 60)
        print("📋 步骤 1/2: 收集模型文件")
        print("=" * 60)
        
        if not copy_model_files(args.model_path, temp_dir, args.model_type):
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
            print("🎉 上传完成!")
            print("=" * 60)
            print(f"\n加载方式:")
            if args.model_type == 'vlm':
                print(f"```python")
                print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
                print(f"model = AutoModelForCausalLM.from_pretrained('{args.repo_id}', trust_remote_code=True)")
                print(f"tokenizer = AutoTokenizer.from_pretrained('{args.repo_id}', trust_remote_code=True)")
                print(f"```")
            else:
                print(f"```python")
                print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
                print(f"model = AutoModelForCausalLM.from_pretrained('{args.repo_id}', trust_remote_code=True)")
                print(f"tokenizer = AutoTokenizer.from_pretrained('{args.repo_id}', trust_remote_code=True)")
                print(f"```")
            return 0
        else:
            return 1


if __name__ == "__main__":
    exit(main())
