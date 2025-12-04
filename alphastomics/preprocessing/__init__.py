"""
AlphaSTomics 数据预处理模块

两阶段预处理流程:
1. Stage1: 切片级预处理 (log normalize + scale)
2. Stage2: 数据集生成 (无放回采样 + 序列化)

用法:
    # 方法1: 命令行
    python -m alphastomics.preprocessing.preprocess stage1 --input_dir ./raw --output_dir ./processed
    python -m alphastomics.preprocessing.preprocess stage2 --input_dir ./processed --output_dir ./dataset
    
    # 方法2: Python API
    from alphastomics.preprocessing import Stage1Preprocessor, Stage2BatchGenerator
    
    # 阶段一
    preprocessor = Stage1Preprocessor(gene_list_file='genes.txt')
    preprocessor.process('./raw', './processed')
    
    # 阶段二
    generator = Stage2BatchGenerator(train_ratio=0.8, val_ratio=0.1)
    generator.process('./processed', './dataset')
    
    # 加载数据训练
    from alphastomics.preprocessing import create_dataloaders
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        './dataset', batch_size=1024, streaming=True
    )
"""

from .preprocess import Stage1Preprocessor, Stage2BatchGenerator
from .batch_loader import (
    ParquetDataset,
    StreamingParquetDataset,
    HuggingFaceDataset,
    create_dataloaders,
    create_dataloader_from_hub,
    cell_collate_fn,
)

__all__ = [
    'Stage1Preprocessor',
    'Stage2BatchGenerator',
    'ParquetDataset',
    'StreamingParquetDataset',
    'HuggingFaceDataset',
    'create_dataloaders',
    'create_dataloader_from_hub',
    'cell_collate_fn',
]
