#!/usr/bin/env python3
"""
上传模型到 HuggingFace 或 ModelScope
支持上传单个文件或整个目录
"""

import os
import argparse
from pathlib import Path
from typing import Optional


def upload_to_hf(
    repo_id: str,
    local_path: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
):
    """上传到 HuggingFace"""
    try:
        from huggingface_hub import HfApi, upload_folder, upload_file
    except ImportError:
        print("错误: 请先安装 huggingface_hub: pip install huggingface_hub")
        return False

    api = HfApi(token=token)
    local_path = Path(local_path)

    # 创建仓库（如果不存在）
    try:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"✅ 仓库就绪: {repo_id}")
    except Exception as e:
        print(f"⚠️  创建仓库失败: {e}")

    # 上传
    try:
        if local_path.is_file():
            print(f"正在上传文件: {local_path.name}")
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or f"Upload {local_path.name}",
            )
        else:
            print(f"正在上传目录: {local_path}")
            upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or "Upload model",
                ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"],
            )
        print("✅ 上传成功")
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False


def upload_to_modelscope(
    repo_id: str,
    local_path: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
):
    """上传到 ModelScope"""
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("错误: 请先安装 modelscope: pip install modelscope")
        return False

    api = HubApi(token=token)
    local_path = Path(local_path)

    # 创建仓库（如果不存在）
    try:
        visibility = 1 if private else 5  # 1=私有, 5=公开
        api.create_model(model_id=repo_id, visibility=visibility)
        print(f"✅ 仓库创建成功: {repo_id}")
    except Exception as e:
        if "exists" in str(e).lower() or "已存在" in str(e):
            print(f"ℹ️  仓库已存在: {repo_id}")
        else:
            print(f"⚠️  创建仓库失败: {e}")

    # 上传
    try:
        if local_path.is_file():
            print(f"正在上传文件: {local_path.name}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=repo_id,
                commit_message=commit_message or f"Upload {local_path.name}",
            )
        else:
            print(f"正在上传目录: {local_path}")
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                commit_message=commit_message or "Upload model",
                ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"],
            )
        print("✅ 上传成功")
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="上传模型到 HuggingFace 或 ModelScope")
    parser.add_argument("--platform", choices=["hf", "modelscope"], required=True, help="目标平台")
    parser.add_argument("--repo_id", type=str, required=True, help="仓库ID (格式: namespace/model_name)")
    parser.add_argument("--local_path", type=str, required=True, help="本地模型路径(文件或目录)")
    parser.add_argument("--token", type=str, default=None, help="API Token")
    parser.add_argument("--private", action="store_true", help="创建私有仓库")
    parser.add_argument("--commit_message", type=str, default=None, help="提交信息")

    args = parser.parse_args()

    # 从环境变量获取 token
    token = args.token
    if not token:
        if args.platform == "hf":
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        else:
            token = os.getenv("MODELSCOPE_TOKEN") or os.getenv("MODEL_SCOPE_TOKEN")

    # 上传
    if args.platform == "hf":
        success = upload_to_hf(
            repo_id=args.repo_id,
            local_path=args.local_path,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )
    else:
        success = upload_to_modelscope(
            repo_id=args.repo_id,
            local_path=args.local_path,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
