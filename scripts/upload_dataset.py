#!/usr/bin/env python3
"""
上传数据集到 ModelScope
支持上传单个文件或整个目录
"""

import os
import argparse
from pathlib import Path
from typing import Optional

try:
    from modelscope.hub.api import HubApi
except ImportError:
    print("错误: 请先安装 modelscope: pip install modelscope")
    exit(1)


def upload_file(
    api: HubApi,
    repo_id: str,
    file_path: str,
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    上传单个文件到 ModelScope
    
    Args:
        api: HubApi 实例
        repo_id: 仓库 ID (格式: namespace/dataset_name)
        file_path: 本地文件路径
        path_in_repo: 在仓库中的路径（默认使用文件名）
        commit_message: 提交信息
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if path_in_repo is None:
        path_in_repo = file_path.name
    
    if commit_message is None:
        commit_message = f"Upload {file_path.name}"
    
    print(f"正在上传: {file_path} -> {repo_id}/{path_in_repo}")
    
    try:

        result = api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type='dataset',
            commit_message=commit_message,
            token=api.token if hasattr(api, 'token') else None,
        )
        print(f"✅ 上传成功: {path_in_repo}")
        return True
    except Exception as e:
        error_msg = str(e)
        if 'already exists' in error_msg.lower() or '已存在' in error_msg:
            print(f"ℹ️  文件已存在，跳过: {path_in_repo}")
            return True
        print(f"❌ 上传失败: {e}")
        return False


def upload_directory(
    api: HubApi,
    repo_id: str,
    local_dir: str,
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
    ignore_patterns: Optional[list] = None,
    token: Optional[str] = None,
):
    """
    上传整个目录到 ModelScope
    
    Args:
        api: HubApi 实例
        repo_id: 仓库 ID
        local_dir: 本地目录路径
        path_in_repo: 在仓库中的路径（默认使用目录名）
        commit_message: 提交信息
        ignore_patterns: 忽略的文件模式列表
    """
    local_dir = Path(local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"目录不存在或不是目录: {local_dir}")
    
    if ignore_patterns is None:
        ignore_patterns = ['.git', '__pycache__', '*.pyc', '.DS_Store']
    
    if commit_message is None:
        commit_message = f"Upload directory {local_dir.name}"
    
    print(f"正在上传目录: {local_dir} -> {repo_id}/{path_in_repo or ''}")
    
    try:

        result = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_dir),
            path_in_repo=path_in_repo or '',
            repo_type='dataset',
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
            token=token,
        )
        print(f"✅ 目录上传成功")
        return len(result) if isinstance(result, list) else 1, 0
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 上传失败: {e}")
        return 0, 1


def create_dataset_repo(
    api: HubApi,
    repo_id: str,
    private: bool = False,
):
    """
    创建新的数据集仓库
    
    Args:
        api: HubApi 实例
        repo_id: 仓库 ID (格式: namespace/dataset_name)
        private: 是否私有
    """
    namespace, dataset_name = repo_id.split('/', 1) if '/' in repo_id else (None, repo_id)
    
    if namespace is None:
        raise ValueError("请提供完整的 repo_id (格式: namespace/dataset_name)")
    
    print(f"正在创建数据集: {repo_id} (私有: {private})")
    
    try:


        visibility = 1 if private else 5
        
        api.create_model(
            model_id=repo_id,
            visibility=visibility,
        )
        print(f"✅ 数据集创建成功: {repo_id}")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if 'already exists' in error_msg or '已存在' in error_msg or 'exists' in error_msg:
            print(f"ℹ️  数据集已存在: {repo_id}")
            return True
        else:
            print(f"⚠️  自动创建失败: {e}")
            print(f"提示: 请先在 ModelScope 网页 (https://modelscope.cn) 手动创建数据集: {repo_id}")
            print(f"     然后使用此脚本上传文件（跳过 --create 参数）")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="上传数据集到 ModelScope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  python scripts/upload_dataset.py \\
      --repo_id your_namespace/your_dataset \\
      --file_path /path/to/data.jsonl \\
      --token YOUR_TOKEN


  python scripts/upload_dataset.py \\
      --repo_id your_namespace/your_dataset \\
      --dir_path /path/to/dataset_dir \\
      --token YOUR_TOKEN


  python scripts/upload_dataset.py \\
      --repo_id your_namespace/new_dataset \\
      --file_path /path/to/data.jsonl \\
      --create \\
      --token YOUR_TOKEN
        """
    )
    
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help="数据集仓库 ID (格式: namespace/dataset_name)"
    )
    parser.add_argument(
        '--file_path',
        type=str,
        help="要上传的单个文件路径"
    )
    parser.add_argument(
        '--dir_path',
        type=str,
        help="要上传的目录路径"
    )
    parser.add_argument(
        '--path_in_repo',
        type=str,
        default=None,
        help="在仓库中的路径（默认使用文件名或目录名）"
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help="ModelScope token (也可通过环境变量 MODEL_SCOPE_TOKEN 设置)"
    )
    parser.add_argument(
        '--create',
        action='store_true',
        help="如果数据集不存在则创建"
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help="创建私有数据集（仅在使用 --create 时有效）"
    )
    parser.add_argument(
        '--commit_message',
        type=str,
        default=None,
        help="提交信息"
    )
    parser.add_argument(
        '--ignore',
        type=str,
        nargs='+',
        default=['.git', '__pycache__', '*.pyc', '.DS_Store'],
        help="忽略的文件/目录模式"
    )
    
    args = parser.parse_args()
    

    if not args.file_path and not args.dir_path:
        parser.error("必须指定 --file_path 或 --dir_path")
    
    if args.file_path and args.dir_path:
        parser.error("不能同时指定 --file_path 和 --dir_path")
    

    token = args.token or os.getenv('MODEL_SCOPE_TOKEN')
    if not token:
        print("警告: 未提供 token，将尝试使用已保存的凭证")
        print("提示: 可通过 --token 参数或环境变量 MODEL_SCOPE_TOKEN 设置")
    

    try:
        api = HubApi(token=token)
        if token:
            print("✅ 已使用提供的 token 初始化 API")
        else:
            print("ℹ️  使用默认凭证初始化 API")
    except Exception as e:
        print(f"❌ 初始化 API 失败: {e}")
        print("提示: 请确保已安装 modelscope 并配置了正确的 token")
        return 1
    

    if args.create:
        create_dataset_repo(api, args.repo_id, args.private)
    

    try:
        if args.file_path:
            success = upload_file(
                api,
                args.repo_id,
                args.file_path,
                args.path_in_repo,
                args.commit_message,
                token
            )
            return 0 if success else 1
        else:
            success_count, fail_count = upload_directory(
                api,
                args.repo_id,
                args.dir_path,
                args.path_in_repo,
                args.commit_message,
                args.ignore,
                token
            )
            return 0 if fail_count == 0 else 1
    except Exception as e:
        print(f"❌ 上传过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
