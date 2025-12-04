"""
用于加载预处理好的 Parquet/Pickle 数据的 DataLoader

支持：
1. Parquet 格式：HuggingFace 兼容，支持流式加载
2. Pickle 格式：按 batch 保存的文件

训练时的两种模式：
1. 全量加载：适合数据量较小的情况
2. 流式加载：适合数据量大、内存有限的情况
"""

import os
import pickle
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterator, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


# ==================== Parquet 数据集 ====================

class ParquetDataset(Dataset):
    """
    从 Parquet 文件加载数据（全量加载到内存）
    
    适合：数据量较小，可以全部加载到内存
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
    ):
        """
        Args:
            data_dir: 数据目录（包含 data/ 子目录）
            split: 数据集划分 ('train', 'validation', 'test')
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("需要安装 pyarrow: pip install pyarrow")
        
        data_path = Path(data_dir) / 'data'
        
        # 查找 parquet 文件（支持单文件和多分片）
        single_file = data_path / f'{split}.parquet'
        shard_pattern = list(data_path.glob(f'{split}-*.parquet'))
        
        if single_file.exists():
            parquet_files = [single_file]
        elif shard_pattern:
            parquet_files = sorted(shard_pattern)
        else:
            raise ValueError(f"未找到 {split} 数据文件在 {data_path}")
        
        logger.info(f"加载 {split}: {len(parquet_files)} 个文件")
        
        # 加载所有文件
        tables = [pq.read_table(f) for f in parquet_files]
        
        import pyarrow as pa
        table = pa.concat_tables(tables)
        
        # 转换为 numpy
        self.expression = np.array(table['expression'].to_pylist(), dtype=np.float32)
        self.positions = np.array(table['positions'].to_pylist(), dtype=np.float32)
        self.cell_types = np.array(table['cell_types'].to_pylist(), dtype=np.int32)
        self.slice_ids = np.array(table['slice_ids'].to_pylist(), dtype=np.int32)
        
        self.n_cells = len(self.expression)
        logger.info(f"加载完成: {self.n_cells} 细胞")
    
    def __len__(self) -> int:
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'expression': torch.tensor(self.expression[idx], dtype=torch.float32),
            'positions': torch.tensor(self.positions[idx], dtype=torch.float32),
            'cell_types': torch.tensor(self.cell_types[idx], dtype=torch.long),
            'slice_ids': torch.tensor(self.slice_ids[idx], dtype=torch.long),
        }


class StreamingParquetDataset(IterableDataset):
    """
    流式 Parquet 数据集（不全量加载）
    
    适合：数据量大，无法全部加载到内存
    
    特点：
    - 每个 epoch 会重新打乱数据
    - 支持多进程加载 (num_workers > 0)
    - 内存占用恒定，与数据量无关
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        batch_size: int = 1024,
        shuffle: bool = True,
        buffer_size: int = 10000,  # shuffle buffer 大小
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据集划分
            batch_size: 批次大小
            shuffle: 是否打乱
            buffer_size: shuffle buffer 大小
        """
        self.data_dir = Path(data_dir) / 'data'
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        # 查找 parquet 文件
        single_file = self.data_dir / f'{split}.parquet'
        shard_pattern = list(self.data_dir.glob(f'{split}-*.parquet'))
        
        if single_file.exists():
            self.parquet_files = [single_file]
        elif shard_pattern:
            self.parquet_files = sorted(shard_pattern)
        else:
            raise ValueError(f"未找到 {split} 数据文件")
        
        logger.info(f"流式加载 {split}: {len(self.parquet_files)} 个文件")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """流式迭代"""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("需要安装 pyarrow")
        
        # 打乱文件顺序
        files = self.parquet_files.copy()
        if self.shuffle:
            np.random.shuffle(files)
        
        # Shuffle buffer
        buffer = []
        
        for parquet_file in files:
            # 分批读取 parquet 文件
            parquet_reader = pq.ParquetFile(parquet_file)
            
            for batch in parquet_reader.iter_batches(batch_size=self.buffer_size):
                # 转换为 Python 对象
                expr = batch['expression'].to_pylist()
                pos = batch['positions'].to_pylist()
                ct = batch['cell_types'].to_pylist()
                sid = batch['slice_ids'].to_pylist()
                
                for i in range(len(expr)):
                    buffer.append({
                        'expression': expr[i],
                        'positions': pos[i],
                        'cell_types': ct[i],
                        'slice_ids': sid[i],
                    })
                    
                    # Buffer 满了，打乱并输出
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            np.random.shuffle(buffer)
                        
                        for item in buffer:
                            yield {
                                'expression': torch.tensor(item['expression'], dtype=torch.float32),
                                'positions': torch.tensor(item['positions'], dtype=torch.float32),
                                'cell_types': torch.tensor(item['cell_types'], dtype=torch.long),
                                'slice_ids': torch.tensor(item['slice_ids'], dtype=torch.long),
                            }
                        buffer = []
        
        # 输出剩余的数据
        if buffer:
            if self.shuffle:
                np.random.shuffle(buffer)
            for item in buffer:
                yield {
                    'expression': torch.tensor(item['expression'], dtype=torch.float32),
                    'positions': torch.tensor(item['positions'], dtype=torch.float32),
                    'cell_types': torch.tensor(item['cell_types'], dtype=torch.long),
                    'slice_ids': torch.tensor(item['slice_ids'], dtype=torch.long),
                }


# ==================== HuggingFace Datasets 加载 ====================

class HuggingFaceDataset(IterableDataset):
    """
    使用 HuggingFace datasets 库流式加载
    
    支持从 Hub 或本地加载
    """
    
    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = 'train',
        shuffle: bool = True,
        buffer_size: int = 10000,
        streaming: bool = True,
    ):
        """
        Args:
            dataset_name_or_path: HuggingFace Hub 名称或本地路径
            split: 数据集划分
            shuffle: 是否打乱
            buffer_size: shuffle buffer 大小
            streaming: 是否流式加载
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("需要安装 datasets: pip install datasets")
        
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        # 尝试加载
        data_path = Path(dataset_name_or_path)
        if data_path.exists():
            # 本地加载
            parquet_files = list((data_path / 'data').glob(f'{split}*.parquet'))
            if parquet_files:
                self.dataset = load_dataset(
                    'parquet',
                    data_files=[str(f) for f in parquet_files],
                    split='train',
                    streaming=streaming,
                )
            else:
                raise ValueError(f"未找到本地 parquet 文件: {data_path}")
        else:
            # 从 Hub 加载
            self.dataset = load_dataset(
                dataset_name_or_path,
                split=split,
                streaming=streaming,
            )
        
        if shuffle and streaming:
            self.dataset = self.dataset.shuffle(buffer_size=buffer_size)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for sample in self.dataset:
            yield {
                'expression': torch.tensor(sample['expression'], dtype=torch.float32),
                'positions': torch.tensor(sample['positions'], dtype=torch.float32),
                'cell_types': torch.tensor(sample['cell_types'], dtype=torch.long),
                'slice_ids': torch.tensor(sample['slice_ids'], dtype=torch.long),
            }


# ==================== Collate 函数 ====================

def cell_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    将单个细胞样本合并成 batch
    """
    return {
        'expression': torch.stack([b['expression'] for b in batch]),      # (B, G)
        'positions': torch.stack([b['positions'] for b in batch]),        # (B, 3)
        'cell_types': torch.stack([b['cell_types'] for b in batch]),      # (B,)
        'slice_ids': torch.stack([b['slice_ids'] for b in batch]),        # (B,)
        'node_mask': torch.ones(len(batch), dtype=torch.float32),         # (B,)
    }


# ==================== DataLoader 创建函数 ====================

def create_dataloaders(
    data_dir: str,
    batch_size: int = 1024,
    num_workers: int = 4,
    streaming: bool = False,
    buffer_size: int = 10000,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建 DataLoader
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作线程数
        streaming: 是否流式加载（大数据集推荐）
        buffer_size: 流式加载的 shuffle buffer 大小
        pin_memory: 是否固定内存
    
    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    data_dir = Path(data_dir)
    
    # 加载元数据
    metadata = {}
    if (data_dir / 'full_metadata.pkl').exists():
        with open(data_dir / 'full_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    elif (data_dir / 'dataset_info.json').exists():
        with open(data_dir / 'dataset_info.json', 'r') as f:
            metadata = json.load(f)
    
    if streaming:
        # 流式加载
        train_dataset = StreamingParquetDataset(
            data_dir, 'train', batch_size, shuffle=True, buffer_size=buffer_size
        )
        val_dataset = StreamingParquetDataset(
            data_dir, 'validation', batch_size, shuffle=False, buffer_size=buffer_size
        )
        test_dataset = StreamingParquetDataset(
            data_dir, 'test', batch_size, shuffle=False, buffer_size=buffer_size
        )
        
        # IterableDataset 不支持 shuffle 参数
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=cell_collate_fn,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=cell_collate_fn,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=cell_collate_fn,
            pin_memory=pin_memory,
        )
    else:
        # 全量加载
        train_dataset = ParquetDataset(data_dir, 'train')
        val_dataset = ParquetDataset(data_dir, 'validation')
        test_dataset = ParquetDataset(data_dir, 'test')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=cell_collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cell_collate_fn,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cell_collate_fn,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader, test_loader, metadata


def create_dataloader_from_hub(
    dataset_name: str,
    batch_size: int = 1024,
    num_workers: int = 4,
    buffer_size: int = 10000,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    从 HuggingFace Hub 创建 DataLoader
    
    Args:
        dataset_name: HuggingFace Hub 数据集名称
        batch_size: 批次大小
        num_workers: 工作线程数
        buffer_size: shuffle buffer 大小
        pin_memory: 是否固定内存
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = HuggingFaceDataset(
        dataset_name, 'train', shuffle=True, buffer_size=buffer_size
    )
    val_dataset = HuggingFaceDataset(
        dataset_name, 'validation', shuffle=False, buffer_size=buffer_size
    )
    test_dataset = HuggingFaceDataset(
        dataset_name, 'test', shuffle=False, buffer_size=buffer_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=cell_collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=cell_collate_fn,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=cell_collate_fn,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader
