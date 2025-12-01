"""
数据加载和预处理模块
支持从多个 h5ad 文件加载 3D 空间转录组切片数据

两种训练模式:
1. 切片级别 (slice-level): 每次输入一整张切片
2. 细胞级别 (cell-level): 每次采样 batch_size 个细胞

预处理流程:
1. 加载 h5ad 文件
2. 基因筛选（高变异基因）
3. Log normalize + Scale
4. 坐标标准化
5. 保存预处理结果
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import pickle
from dataclasses import dataclass, field
import logging
import h5py

logger = logging.getLogger(__name__)


# ==================== 预处理器 ====================

class SpatialDataPreprocessor:
    """
    空间转录组数据预处理器
    
    支持多平台数据（StereoSeq、MERFISH、MERSCOPE 等）
    
    预处理流程:
    1. 加载 h5ad 文件
    2. 使用固定基因列表（缺失基因补 0）
    3. Log normalize + Scale（每张切片独立处理）
    4. 2D 坐标扩展为 3D
    5. 保存预处理结果
    """
    
    def __init__(
        self,
        gene_list: Optional[List[str]] = None,
        gene_list_file: Optional[str] = None,
        position_key: str = 'ccf',
        position_is_3d: bool = True,
        z_spacing: float = 1.0,
        cell_type_key: Optional[str] = 'cell_type',
        scale: bool = True,
        normalize_position: bool = True,
    ):
        """
        Args:
            gene_list: 固定的基因列表。如果提供，则只使用这些基因
            gene_list_file: 基因列表文件路径（每行一个基因名，或逗号分隔）
            position_key: AnnData.obsm 中坐标的 key（默认 'ccf'）
            position_is_3d: 坐标是否已经是 3D（True=直接使用，False=2D需扩展）
            z_spacing: 切片间的 z 间距（仅当 position_is_3d=False 时使用）
            cell_type_key: 细胞类型的 key（可选）
            scale: 是否进行 scale（z-score 标准化）
            normalize_position: 是否对坐标进行标准化（中心化+缩放到[-1,1]）
        
        注意: gene_list 和 gene_list_file 二选一，优先使用 gene_list
        """
        self.position_key = position_key
        self.position_is_3d = position_is_3d
        self.z_spacing = z_spacing
        self.cell_type_key = cell_type_key
        self.scale = scale
        self.normalize_position = normalize_position
        
        # 加载固定基因列表
        self.fixed_gene_list = self._load_gene_list(gene_list, gene_list_file)
        
        # 预处理后保存的元数据
        self.selected_genes: Optional[List[str]] = None
        self.cell_type_mapping: Optional[Dict[str, int]] = None
        self.n_genes: int = 0
        
        # 全局坐标统计（用于跨切片标准化）
        self.global_position_mean: Optional[np.ndarray] = None
        self.global_position_scale: Optional[float] = None
    
    def _load_gene_list(
        self, 
        gene_list: Optional[List[str]], 
        gene_list_file: Optional[str]
    ) -> Optional[List[str]]:
        """加载基因列表"""
        if gene_list is not None:
            logger.info(f"使用提供的固定基因列表: {len(gene_list)} 个基因")
            return gene_list
        
        if gene_list_file is not None:
            gene_list_path = Path(gene_list_file)
            if not gene_list_path.exists():
                raise FileNotFoundError(f"基因列表文件不存在: {gene_list_file}")
            
            with open(gene_list_path, 'r') as f:
                content = f.read().strip()
            
            # 支持多种格式：每行一个基因，或逗号分隔，或制表符分隔
            if '\n' in content:
                genes = [line.strip() for line in content.split('\n') if line.strip()]
            elif ',' in content:
                genes = [g.strip() for g in content.split(',') if g.strip()]
            elif '\t' in content:
                genes = [g.strip() for g in content.split('\t') if g.strip()]
            else:
                genes = [content]  # 单个基因
            
            logger.info(f"从文件 {gene_list_file} 加载固定基因列表: {len(genes)} 个基因")
            return genes
        
        return None  # 不使用固定列表，将使用共同基因
        
    def preprocess_and_save(
        self,
        h5ad_files: List[str],
        output_dir: str,
        z_coords: Optional[List[float]] = None,
    ) -> Dict:
        """
        预处理 h5ad 文件并保存
        
        Args:
            h5ad_files: h5ad 文件路径列表
            output_dir: 输出目录
            z_coords: 每个切片的 z 坐标（仅当 position_is_3d=False 时使用）
        
        Returns:
            元数据字典
        """
        try:
            import anndata
            import scanpy as sc
        except ImportError:
            raise ImportError("需要安装 anndata 和 scanpy: pip install anndata scanpy")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # z_coords 仅在 2D 扩展模式下使用
        if not self.position_is_3d and z_coords is None:
            z_coords = [i * self.z_spacing for i in range(len(h5ad_files))]
        
        logger.info(f"加载 {len(h5ad_files)} 个 h5ad 文件...")
        
        # 第一遍：加载所有数据
        all_adata = []
        for file_path in h5ad_files:
            logger.info(f"  加载 {file_path}")
            adata = anndata.read_h5ad(file_path)
            adata.var_names_make_unique()
            all_adata.append(adata)
        
        # 确定使用的基因列表
        if self.fixed_gene_list is not None:
            # 使用固定基因列表
            self.selected_genes = self.fixed_gene_list
            logger.info(f"使用固定基因列表: {len(self.selected_genes)} 个基因")
            
            # 统计每个切片的基因覆盖情况
            for i, adata in enumerate(all_adata):
                available = set(adata.var_names)
                covered = sum(1 for g in self.selected_genes if g in available)
                coverage = covered / len(self.selected_genes) * 100
                logger.info(f"  切片 {i}: 覆盖 {covered}/{len(self.selected_genes)} ({coverage:.1f}%) 基因")
        else:
            # 找共同基因（使用所有共同基因，不筛选）
            common_genes = set(all_adata[0].var_names)
            for adata in all_adata[1:]:
                common_genes = common_genes.intersection(set(adata.var_names))
            self.selected_genes = sorted(list(common_genes))
            logger.info(f"使用共同基因: {len(self.selected_genes)} 个基因")
        
        self.n_genes = len(self.selected_genes)
        
        # 收集细胞类型
        if self.cell_type_key:
            all_cell_types = set()
            for adata in all_adata:
                if self.cell_type_key in adata.obs.columns:
                    all_cell_types.update(adata.obs[self.cell_type_key].unique())
            if all_cell_types:
                self.cell_type_mapping = {ct: i for i, ct in enumerate(sorted(all_cell_types))}
                logger.info(f"细胞类型数量: {len(self.cell_type_mapping)}")
        
        # 如果需要全局坐标标准化，先收集所有坐标统计量
        if self.normalize_position and self.position_is_3d:
            all_positions = []
            for adata in all_adata:
                if self.position_key in adata.obsm:
                    pos = np.array(adata.obsm[self.position_key])[:, :3]
                    all_positions.append(pos)
            if all_positions:
                all_positions = np.concatenate(all_positions, axis=0)
                self.global_position_mean = all_positions.mean(axis=0)
                self.global_position_scale = np.abs(all_positions - self.global_position_mean).max()
                logger.info(f"全局坐标统计: mean={self.global_position_mean}, scale={self.global_position_scale}")
        
        # 第二遍：预处理每张切片并保存
        all_cells_info = []  # 记录每个切片的细胞信息
        total_cells = 0
        
        for i, (adata, file_path) in enumerate(zip(all_adata, h5ad_files)):
            slice_id = Path(file_path).stem
            logger.info(f"处理切片 {slice_id}...")
            
            # 选择基因（只选择数据中存在的基因）
            genes_in_data = [g for g in self.selected_genes if g in adata.var_names]
            missing_genes = [g for g in self.selected_genes if g not in adata.var_names]
            
            if missing_genes:
                logger.info(f"  切片 {slice_id}: 缺少 {len(missing_genes)} 个基因，将补 0")
            
            if genes_in_data:
                adata_subset = adata[:, genes_in_data].copy()
            else:
                # 如果没有任何匹配的基因，创建全零表达矩阵
                logger.warning(f"切片 {slice_id} 没有任何匹配的基因！")
                expression = np.zeros((adata.n_obs, len(self.selected_genes)), dtype=np.float32)
                # 跳过后续的基因处理，直接处理坐标
                goto_positions = True
            
            if not (len(genes_in_data) == 0):
                # Log normalize（每张切片独立）
                sc.pp.normalize_total(adata_subset, target_sum=1e4)
                sc.pp.log1p(adata_subset)
                
                # Scale（可选）
                if self.scale:
                    sc.pp.scale(adata_subset, max_value=10)
                
                # 获取表达量
                if hasattr(adata_subset.X, 'toarray'):
                    expression_subset = adata_subset.X.toarray()
                else:
                    expression_subset = np.array(adata_subset.X)
                
                # 创建完整表达矩阵，缺失基因补 0
                expression = np.zeros((adata.n_obs, len(self.selected_genes)), dtype=np.float32)
                for j, gene in enumerate(genes_in_data):
                    gene_idx = self.selected_genes.index(gene)
                    expression[:, gene_idx] = expression_subset[:, j]
            
            # 获取坐标
            if self.position_key in adata.obsm:
                positions_raw = np.array(adata.obsm[self.position_key])
            elif 'X_spatial' in adata.obsm:
                positions_raw = np.array(adata.obsm['X_spatial'])
            elif 'spatial' in adata.obs.columns:
                positions_raw = np.array(adata.obs['spatial'].tolist())
            else:
                raise ValueError(f"未找到空间坐标。可用的 keys: {list(adata.obsm.keys())}")
            
            if self.position_is_3d:
                # 直接使用 3D 坐标（如 CCF 坐标系）
                positions = positions_raw[:, :3].copy()
                
                if self.normalize_position:
                    # 使用全局统计量标准化（保持相对位置关系）
                    positions = positions - self.global_position_mean
                    if self.global_position_scale > 0:
                        positions = positions / self.global_position_scale
                
                z_coord_value = positions[:, 2].mean()  # 记录平均 z 值
            else:
                # 2D 坐标扩展为 3D
                positions_2d = positions_raw[:, :2]
                
                if self.normalize_position:
                    # 每张切片独立标准化
                    positions_2d = positions_2d - positions_2d.mean(axis=0)
                    scale = np.abs(positions_2d).max()
                    if scale > 0:
                        positions_2d = positions_2d / scale
                
                # 扩展为 3D
                z_coord_value = z_coords[i] if z_coords else i * self.z_spacing
                z_col = np.full((positions_2d.shape[0], 1), z_coord_value)
                positions = np.hstack([positions_2d, z_col])
            
            # 细胞类型
            cell_types = None
            if self.cell_type_key and self.cell_type_key in adata.obs.columns:
                cell_types = adata.obs[self.cell_type_key].map(self.cell_type_mapping).values.astype(np.int64)
            
            # 保存切片数据
            slice_data = {
                'expression': expression.astype(np.float32),
                'positions': positions.astype(np.float32),
                'cell_types': cell_types,
                'slice_id': slice_id,
                'n_cells': expression.shape[0],
                'z_coord': float(z_coord_value),
            }
            
            slice_file = output_dir / f"slice_{i:03d}_{slice_id}.pkl"
            with open(slice_file, 'wb') as f:
                pickle.dump(slice_data, f)
            
            all_cells_info.append({
                'slice_idx': i,
                'slice_id': slice_id,
                'slice_file': str(slice_file),
                'n_cells': expression.shape[0],
                'cell_start_idx': total_cells,
            })
            
            total_cells += expression.shape[0]
            logger.info(f"  切片 {slice_id}: {expression.shape[0]} 个细胞")
        
        # 保存元数据
        metadata = {
            'selected_genes': self.selected_genes,
            'n_genes': self.n_genes,
            'cell_type_mapping': self.cell_type_mapping,
            'n_cell_types': len(self.cell_type_mapping) if self.cell_type_mapping else 0,
            'slices_info': all_cells_info,
            'n_slices': len(all_cells_info),
            'total_cells': total_cells,
            # 坐标相关元数据
            'position_is_3d': self.position_is_3d,
            'position_key': self.position_key,
            'normalize_position': self.normalize_position,
            'global_position_mean': self.global_position_mean.tolist() if self.global_position_mean is not None else None,
            'global_position_scale': float(self.global_position_scale) if self.global_position_scale is not None else None,
        }
        
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"预处理完成! 总细胞数: {total_cells}")
        return metadata


# ==================== 切片级别数据集 ====================

class SliceLevelDataset(Dataset):
    """
    切片级别数据集
    每次返回一整张切片的所有细胞
    """
    
    def __init__(
        self,
        data_dir: str,
        slice_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            data_dir: 预处理数据目录
            slice_indices: 使用的切片索引（用于 train/val/test 划分）
        """
        self.data_dir = Path(data_dir)
        
        # 加载元数据
        with open(self.data_dir / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.slices_info = self.metadata['slices_info']
        
        # 筛选切片
        if slice_indices is not None:
            self.slices_info = [self.slices_info[i] for i in slice_indices]
        
        self.n_genes = self.metadata['n_genes']
    
    def __len__(self) -> int:
        return len(self.slices_info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        slice_info = self.slices_info[idx]
        
        # 加载切片数据
        with open(slice_info['slice_file'], 'rb') as f:
            data = pickle.load(f)
        
        result = {
            'expression': torch.tensor(data['expression'], dtype=torch.float32),
            'positions': torch.tensor(data['positions'], dtype=torch.float32),
            'node_mask': torch.ones(data['n_cells'], dtype=torch.float32),
            'slice_id': data['slice_id'],
            'n_cells': data['n_cells'],
        }
        
        if data['cell_types'] is not None:
            result['cell_types'] = torch.tensor(data['cell_types'], dtype=torch.long)
        
        return result


def slice_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    切片级别的 collate 函数
    将不同大小的切片 pad 到相同大小
    """
    max_cells = max(item['n_cells'] for item in batch)
    batch_size = len(batch)
    n_genes = batch[0]['expression'].shape[1]
    
    # 初始化
    expression = torch.zeros(batch_size, max_cells, n_genes)
    positions = torch.zeros(batch_size, max_cells, 3)
    node_mask = torch.zeros(batch_size, max_cells)
    slice_ids = []
    
    has_cell_types = 'cell_types' in batch[0]
    if has_cell_types:
        cell_types = torch.zeros(batch_size, max_cells, dtype=torch.long)
    
    for i, item in enumerate(batch):
        n = item['n_cells']
        expression[i, :n] = item['expression']
        positions[i, :n] = item['positions']
        node_mask[i, :n] = 1.0
        slice_ids.append(item['slice_id'])
        
        if has_cell_types:
            cell_types[i, :n] = item['cell_types']
    
    result = {
        'expression': expression,
        'positions': positions,
        'node_mask': node_mask,
        'slice_ids': slice_ids,
    }
    
    if has_cell_types:
        result['cell_types'] = cell_types
    
    return result


# ==================== 细胞级别数据集 ====================

class CellLevelDataset(Dataset):
    """
    细胞级别数据集
    将所有切片的细胞合并，每次采样 batch_size 个细胞
    
    注意：这个数据集返回单个细胞，使用 DataLoader 的 batch_size 来控制每批细胞数
    """
    
    def __init__(
        self,
        data_dir: str,
        slice_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            data_dir: 预处理数据目录
            slice_indices: 使用的切片索引
        """
        self.data_dir = Path(data_dir)
        
        # 加载元数据
        with open(self.data_dir / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        slices_info = self.metadata['slices_info']
        if slice_indices is not None:
            slices_info = [slices_info[i] for i in slice_indices]
        
        # 加载所有切片数据并合并
        logger.info("加载细胞级别数据...")
        all_expression = []
        all_positions = []
        all_cell_types = []
        all_slice_ids = []
        
        for slice_info in slices_info:
            with open(slice_info['slice_file'], 'rb') as f:
                data = pickle.load(f)
            
            all_expression.append(data['expression'])
            all_positions.append(data['positions'])
            if data['cell_types'] is not None:
                all_cell_types.append(data['cell_types'])
            all_slice_ids.extend([data['slice_id']] * data['n_cells'])
        
        self.expression = np.concatenate(all_expression, axis=0)
        self.positions = np.concatenate(all_positions, axis=0)
        self.cell_types = np.concatenate(all_cell_types, axis=0) if all_cell_types else None
        self.slice_ids = all_slice_ids
        
        self.n_cells = self.expression.shape[0]
        self.n_genes = self.expression.shape[1]
        
        logger.info(f"加载完成: {self.n_cells} 个细胞, {self.n_genes} 个基因")
    
    def __len__(self) -> int:
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {
            'expression': torch.tensor(self.expression[idx], dtype=torch.float32),
            'positions': torch.tensor(self.positions[idx], dtype=torch.float32),
            'slice_id': self.slice_ids[idx],
        }
        
        if self.cell_types is not None:
            result['cell_types'] = torch.tensor(self.cell_types[idx], dtype=torch.long)
        
        return result


def cell_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    细胞级别的 collate 函数
    将单个细胞堆叠成 batch
    """
    batch_size = len(batch)
    
    expression = torch.stack([item['expression'] for item in batch])  # (B, G)
    positions = torch.stack([item['positions'] for item in batch])    # (B, 3)
    
    # 为了与模型兼容，添加一个节点维度 (B, 1, ...)
    result = {
        'expression': expression.unsqueeze(1),  # (B, 1, G)
        'positions': positions.unsqueeze(1),     # (B, 1, 3)
        'node_mask': torch.ones(batch_size, 1),  # (B, 1)
        'slice_ids': [item['slice_id'] for item in batch],
    }
    
    if 'cell_types' in batch[0]:
        cell_types = torch.stack([item['cell_types'] for item in batch])
        result['cell_types'] = cell_types.unsqueeze(1)  # (B, 1)
    
    return result


# ==================== 数据加载器创建函数 ====================

def create_slice_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建切片级别的数据加载器
    
    Args:
        data_dir: 预处理数据目录
        batch_size: 每批切片数量（通常为 1，因为切片大小不同）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 工作线程数
        shuffle_train: 是否打乱训练数据
    
    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    # 加载元数据
    with open(Path(data_dir) / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    n_slices = metadata['n_slices']
    
    # 划分切片索引
    indices = np.random.permutation(n_slices)
    n_train = int(n_slices * train_ratio)
    n_val = int(n_slices * val_ratio)
    
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()
    
    logger.info(f"切片划分: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # 创建数据集
    train_dataset = SliceLevelDataset(data_dir, train_indices)
    val_dataset = SliceLevelDataset(data_dir, val_indices)
    test_dataset = SliceLevelDataset(data_dir, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=slice_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=slice_collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=slice_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, metadata


def create_cell_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建细胞级别的数据加载器
    
    Args:
        data_dir: 预处理数据目录
        batch_size: 每批细胞数量
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 工作线程数
        shuffle_train: 是否打乱训练数据
    
    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    # 加载元数据
    with open(Path(data_dir) / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    n_slices = metadata['n_slices']
    
    # 按切片划分（保证同一切片的细胞在同一集合）
    indices = np.random.permutation(n_slices)
    n_train = int(n_slices * train_ratio)
    n_val = int(n_slices * val_ratio)
    
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()
    
    logger.info(f"切片划分: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # 创建数据集
    train_dataset = CellLevelDataset(data_dir, train_indices)
    val_dataset = CellLevelDataset(data_dir, val_indices)
    test_dataset = CellLevelDataset(data_dir, test_indices)
    
    logger.info(f"细胞划分: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=cell_collate_fn,
        pin_memory=True,
        drop_last=True,  # 丢弃最后不完整的 batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cell_collate_fn,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=cell_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader, metadata


# 保持向后兼容的别名
def create_dataloaders(
    data_dir: str,
    mode: str = 'slice',
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建数据加载器（统一接口）
    
    Args:
        data_dir: 预处理数据目录
        mode: 'slice'（切片级别）或 'cell'（细胞级别）
        **kwargs: 传递给具体创建函数的参数
    
    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    if mode == 'slice':
        return create_slice_dataloaders(data_dir, **kwargs)
    elif mode == 'cell':
        return create_cell_dataloaders(data_dir, **kwargs)
    else:
        raise ValueError(f"未知模式: {mode}，请使用 'slice' 或 'cell'")


# ==================== 命令行接口 ====================

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='预处理空间转录组数据')
    parser.add_argument('--input_dir', type=str, required=True, help='包含 h5ad 文件的目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--gene_list_file', type=str, default=None, 
                        help='固定基因列表文件（每行一个基因名）')
    parser.add_argument('--position_key', type=str, default='ccf', help='空间坐标的 key（默认 ccf）')
    parser.add_argument('--position_is_3d', action='store_true', default=True,
                        help='坐标是否已经是 3D（默认 True）')
    parser.add_argument('--position_is_2d', action='store_true',
                        help='坐标是 2D，需要扩展为 3D')
    parser.add_argument('--z_spacing', type=float, default=1.0, help='切片间 z 间距（仅2D模式）')
    parser.add_argument('--cell_type_key', type=str, default='cell_type', help='细胞类型的 key')
    parser.add_argument('--no_scale', action='store_true', help='不进行表达量 scale')
    parser.add_argument('--no_normalize_position', action='store_true', help='不进行坐标标准化')
    
    args = parser.parse_args()
    
    # 查找 h5ad 文件
    input_path = Path(args.input_dir)
    h5ad_files = sorted(input_path.glob('*.h5ad'))
    
    if not h5ad_files:
        raise ValueError(f"在 {args.input_dir} 中未找到 h5ad 文件")
    
    logger.info(f"找到 {len(h5ad_files)} 个 h5ad 文件")
    
    # 确定坐标维度
    position_is_3d = not args.position_is_2d  # 如果指定了2D，则不是3D
    
    # 预处理
    preprocessor = SpatialDataPreprocessor(
        gene_list_file=args.gene_list_file,
        position_key=args.position_key,
        position_is_3d=position_is_3d,
        z_spacing=args.z_spacing,
        cell_type_key=args.cell_type_key,
        scale=not args.no_scale,
        normalize_position=not args.no_normalize_position,
    )
    
    metadata = preprocessor.preprocess_and_save(
        [str(f) for f in h5ad_files],
        args.output_dir
    )
    
    logger.info("预处理完成!")
    logger.info(f"  基因数: {metadata['n_genes']}")
    logger.info(f"  切片数: {metadata['n_slices']}")
    logger.info(f"  总细胞数: {metadata['total_cells']}")
    logger.info(f"  坐标模式: {'3D (CCF)' if position_is_3d else '2D -> 3D'}")

