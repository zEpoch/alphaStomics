"""
上传数据集到 HuggingFace Hub

Usage:
    python upload_to_hf.py --data_dir ./dataset --repo_id your-username/alphastomics-dataset
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_to_hub(
    data_dir: str,
    repo_id: str,
    private: bool = False,
    token: str = None,
):
    """
    上传数据集到 HuggingFace Hub
    
    Args:
        data_dir: 本地数据目录（阶段二输出）
        repo_id: HuggingFace 仓库 ID (格式: username/dataset-name)
        private: 是否设为私有仓库
        token: HuggingFace API token
    """
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        raise ImportError("需要安装 huggingface_hub: pip install huggingface_hub")
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError(f"目录不存在: {data_dir}")
    
    logger.info(f"上传 {data_dir} 到 {repo_id}...")
    
    api = HfApi(token=token)
    
    # 创建仓库（如果不存在）
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        logger.info(f"仓库 {repo_id} 已创建/存在")
    except Exception as e:
        logger.warning(f"创建仓库时出错: {e}")
    
    # 上传文件夹
    upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        ignore_patterns=["*.pkl", "_intermediate_slices/**"],  # 不上传中间文件
    )
    
    logger.info(f"上传完成! 访问: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="上传数据集到 HuggingFace Hub")
    parser.add_argument('--data_dir', '-d', required=True, help='数据目录')
    parser.add_argument('--repo_id', '-r', required=True, help='HuggingFace 仓库 ID')
    parser.add_argument('--private', action='store_true', help='设为私有仓库')
    parser.add_argument('--token', '-t', default=None, help='HuggingFace API token')
    
    args = parser.parse_args()
    
    upload_to_hub(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
    )


if __name__ == '__main__':
    main()
