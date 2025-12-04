"""
AlphaSTomics 两阶段数据预处理脚本

阶段一：切片级预处理
- 读取原始 h5ad 文件
- Log normalize + Scale（每片独立）
- 基因对齐（固定基因列表或共同基因）
- 坐标标准化
- 保存为 .h5ad 或 .pkl

阶段二：Batch 生成
- 跨切片随机采样细胞
- 生成固定大小的训练 batch
- 序列化保存（支持 Arrow/Parquet 格式，便于上传 HuggingFace）

Usage:
    # 阶段一：预处理切片
    python preprocess.py stage1 --input_dir ./raw --output_dir ./processed_slices
    
    # 阶段二：生成 batch
    python preprocess.py stage2 --input_dir ./processed_slices --output_dir ./batches
    
    # 一步完成两个阶段
    python preprocess.py all --raw_dir ./raw --output_dir ./dataset
"""

import os
import sys
import argparse
import logging
import pickle
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================

@dataclass
class SliceMetadata:
    """单个切片的元数据"""
    slice_id: str
    n_cells: int
    n_genes: int
    z_coord: float
    file_path: str
    has_cell_types: bool = False
    cell_type_counts: Optional[Dict[str, int]] = None


@dataclass 
class DatasetMetadata:
    """整个数据集的元数据"""
    n_slices: int
    total_cells: int
    n_genes: int
    gene_names: List[str]
    n_cell_types: int
    cell_type_mapping: Optional[Dict[str, int]]
    slices: List[SliceMetadata]
    # 坐标相关
    position_key: str
    position_is_3d: bool
    global_position_mean: Optional[List[float]] = None
    global_position_scale: Optional[float] = None
    # 处理参数
    preprocessing_params: Optional[Dict] = None


# ==================== 阶段一：切片预处理 ====================

class Stage1Preprocessor:
    """
    阶段一：切片级预处理
    
    - 读取原始 h5ad
    - Log normalize + Scale
    - 基因对齐
    - 坐标处理
    - 保存预处理后的切片
    """
    
    def __init__(
        self,
        gene_list_file: Optional[str] = None,
        position_key: str = 'spatial',
        position_is_3d: bool = False,
        z_spacing: float = 10.0,
        cell_type_key: Optional[str] = 'cell_type',
        scale: bool = True,
        normalize_position: bool = True,
        target_sum: float = 1e4,
        max_scale_value: float = 10.0,
    ):
        """
        Args:
            gene_list_file: 固定基因列表文件（每行一个基因）
            position_key: obsm 中坐标的 key
            position_is_3d: 坐标是否已经是 3D
            z_spacing: 切片间 z 间距（仅 2D 时使用）
            cell_type_key: obs 中细胞类型的 key
            scale: 是否做 z-score 标准化
            normalize_position: 是否标准化坐标
            target_sum: normalize_total 的目标值
            max_scale_value: scale 后的最大值截断
        """
        self.gene_list_file = gene_list_file
        self.position_key = position_key
        self.position_is_3d = position_is_3d
        self.z_spacing = z_spacing
        self.cell_type_key = cell_type_key
        self.scale = scale
        self.normalize_position = normalize_position
        self.target_sum = target_sum
        self.max_scale_value = max_scale_value
        
        # 加载固定基因列表
        self.fixed_gene_list = self._load_gene_list()
        
        # 处理过程中收集的信息
        self.selected_genes: Optional[List[str]] = None
        self.cell_type_mapping: Optional[Dict[str, int]] = None
        self.global_position_stats: Dict = {}
    
    def _load_gene_list(self) -> Optional[List[str]]:
        """加载基因列表文件"""
        if self.gene_list_file is None:
            return None
        
        path = Path(self.gene_list_file)
        if not path.exists():
            raise FileNotFoundError(f"基因列表文件不存在: {self.gene_list_file}")
        
        with open(path, 'r') as f:
            content = f.read().strip()
        
        # 支持多种分隔符
        if '\n' in content:
            genes = [line.strip() for line in content.split('\n') if line.strip()]
        elif ',' in content:
            genes = [g.strip() for g in content.split(',') if g.strip()]
        elif '\t' in content:
            genes = [g.strip() for g in content.split('\t') if g.strip()]
        else:
            genes = [content]
        
        logger.info(f"从 {self.gene_list_file} 加载 {len(genes)} 个基因")
        return genes
    
    def process(
        self,
        input_dir: str,
        output_dir: str,
        z_coords: Optional[List[float]] = None,
        file_pattern: str = "*.h5ad",
    ) -> DatasetMetadata:
        """
        处理所有切片
        
        Args:
            input_dir: 输入目录（包含 h5ad 文件）
            output_dir: 输出目录
            z_coords: 每个切片的 z 坐标（可选）
            file_pattern: 文件匹配模式
        
        Returns:
            数据集元数据
        """
        try:
            import anndata
            import scanpy as sc
        except ImportError:
            raise ImportError("需要安装 anndata 和 scanpy: pip install anndata scanpy")
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有 h5ad 文件
        h5ad_files = sorted(list(input_dir.glob(file_pattern)))
        if not h5ad_files:
            raise ValueError(f"在 {input_dir} 中未找到 {file_pattern} 文件")
        
        logger.info(f"找到 {len(h5ad_files)} 个 h5ad 文件")
        
        # 如果是 2D 数据且没有指定 z_coords，自动生成
        if not self.position_is_3d and z_coords is None:
            z_coords = [i * self.z_spacing for i in range(len(h5ad_files))]
        
        # ===== 第一遍扫描：确定基因列表和细胞类型 =====
        logger.info("第一遍扫描：确定基因列表和细胞类型...")
        
        all_adata = []
        for fpath in tqdm(h5ad_files, desc="加载文件"):
            adata = anndata.read_h5ad(fpath)
            adata.var_names_make_unique()
            all_adata.append(adata)
        
        # 确定基因列表
        if self.fixed_gene_list is not None:
            self.selected_genes = self.fixed_gene_list
            logger.info(f"使用固定基因列表: {len(self.selected_genes)} 个基因")
        else:
            # 找共同基因
            common_genes = set(all_adata[0].var_names)
            for adata in all_adata[1:]:
                common_genes = common_genes.intersection(set(adata.var_names))
            self.selected_genes = sorted(list(common_genes))
            logger.info(f"使用共同基因: {len(self.selected_genes)} 个基因")
        
        # 收集细胞类型
        all_cell_types = set()
        if self.cell_type_key:
            for adata in all_adata:
                if self.cell_type_key in adata.obs.columns:
                    all_cell_types.update(adata.obs[self.cell_type_key].dropna().unique())
        
        if all_cell_types:
            self.cell_type_mapping = {ct: i for i, ct in enumerate(sorted(all_cell_types))}
            logger.info(f"细胞类型数量: {len(self.cell_type_mapping)}")
        
        # 如果需要全局坐标标准化，收集统计量
        if self.normalize_position and self.position_is_3d:
            all_positions = []
            for adata in all_adata:
                pos = self._get_positions(adata)
                if pos is not None:
                    all_positions.append(pos[:, :3])
            
            if all_positions:
                all_positions = np.concatenate(all_positions, axis=0)
                self.global_position_stats = {
                    'mean': all_positions.mean(axis=0),
                    'scale': np.abs(all_positions - all_positions.mean(axis=0)).max()
                }
                logger.info(f"全局坐标统计: mean={self.global_position_stats['mean']}, scale={self.global_position_stats['scale']}")
        
        # ===== 第二遍处理：预处理每张切片并保存 =====
        logger.info("第二遍处理：预处理并保存...")
        
        slices_metadata = []
        total_cells = 0
        
        for i, (adata, fpath) in enumerate(tqdm(zip(all_adata, h5ad_files), total=len(all_adata), desc="处理切片")):
            slice_id = fpath.stem
            z_coord = z_coords[i] if z_coords else 0.0
            
            # 处理单个切片
            slice_data, slice_meta = self._process_single_slice(
                adata, slice_id, i, z_coord
            )
            
            # 保存
            output_file = output_dir / f"slice_{i:04d}_{slice_id}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(slice_data, f)
            
            slice_meta.file_path = str(output_file)
            slices_metadata.append(slice_meta)
            total_cells += slice_meta.n_cells
        
        # 创建数据集元数据
        dataset_metadata = DatasetMetadata(
            n_slices=len(slices_metadata),
            total_cells=total_cells,
            n_genes=len(self.selected_genes),
            gene_names=self.selected_genes,
            n_cell_types=len(self.cell_type_mapping) if self.cell_type_mapping else 0,
            cell_type_mapping=self.cell_type_mapping,
            slices=[asdict(s) for s in slices_metadata],
            position_key=self.position_key,
            position_is_3d=self.position_is_3d,
            global_position_mean=self.global_position_stats.get('mean', None),
            global_position_scale=self.global_position_stats.get('scale', None),
            preprocessing_params={
                'scale': self.scale,
                'normalize_position': self.normalize_position,
                'target_sum': self.target_sum,
                'max_scale_value': self.max_scale_value,
                'z_spacing': self.z_spacing,
            }
        )
        
        # 保存元数据
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(asdict(dataset_metadata), f, indent=2, default=_json_serializer)
        
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(asdict(dataset_metadata), f)
        
        # 保存基因列表
        with open(output_dir / 'genes.txt', 'w') as f:
            f.write('\n'.join(self.selected_genes))
        
        logger.info(f"阶段一完成！共 {total_cells} 个细胞，{len(self.selected_genes)} 个基因")
        logger.info(f"输出目录: {output_dir}")
        
        return dataset_metadata
    
    def _get_positions(self, adata) -> Optional[np.ndarray]:
        """获取坐标"""
        # 尝试多个可能的 key
        possible_keys = [self.position_key, 'spatial', 'X_spatial', 'X_umap']
        
        for key in possible_keys:
            if key in adata.obsm:
                return np.array(adata.obsm[key])
        
        # 尝试从 obs 获取
        if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            x = adata.obs['x'].values
            y = adata.obs['y'].values
            if 'z' in adata.obs.columns:
                z = adata.obs['z'].values
                return np.stack([x, y, z], axis=1)
            return np.stack([x, y], axis=1)
        
        return None
    
    def _process_single_slice(
        self,
        adata,
        slice_id: str,
        slice_idx: int,
        z_coord: float,
    ) -> Tuple[Dict, SliceMetadata]:
        """处理单个切片"""
        import scanpy as sc
        
        n_cells = adata.n_obs
        
        # ===== 基因选择和表达量处理 =====
        genes_in_data = [g for g in self.selected_genes if g in adata.var_names]
        missing_genes = set(self.selected_genes) - set(genes_in_data)
        
        if missing_genes:
            logger.debug(f"切片 {slice_id}: 缺少 {len(missing_genes)} 个基因")
        
        if genes_in_data:
            adata_subset = adata[:, genes_in_data].copy()
            
            # Log normalize（每片独立）
            sc.pp.normalize_total(adata_subset, target_sum=self.target_sum)
            sc.pp.log1p(adata_subset)
            
            # Scale（可选）
            if self.scale:
                sc.pp.scale(adata_subset, max_value=self.max_scale_value)
            
            # 获取表达矩阵
            if hasattr(adata_subset.X, 'toarray'):
                expr_subset = adata_subset.X.toarray()
            else:
                expr_subset = np.array(adata_subset.X)
            
            # 创建完整表达矩阵，缺失基因补 0
            expression = np.zeros((n_cells, len(self.selected_genes)), dtype=np.float32)
            for j, gene in enumerate(genes_in_data):
                gene_idx = self.selected_genes.index(gene)
                expression[:, gene_idx] = expr_subset[:, j]
        else:
            # 没有匹配的基因
            logger.warning(f"切片 {slice_id}: 没有匹配的基因！")
            expression = np.zeros((n_cells, len(self.selected_genes)), dtype=np.float32)
        
        # ===== 坐标处理 =====
        positions_raw = self._get_positions(adata)
        if positions_raw is None:
            raise ValueError(f"切片 {slice_id}: 未找到空间坐标")
        
        if self.position_is_3d:
            positions = positions_raw[:, :3].copy().astype(np.float32)
            
            if self.normalize_position and self.global_position_stats:
                positions = positions - self.global_position_stats['mean']
                if self.global_position_stats['scale'] > 0:
                    positions = positions / self.global_position_stats['scale']
            
            z_coord_value = float(positions[:, 2].mean())
        else:
            positions_2d = positions_raw[:, :2].copy().astype(np.float32)
            
            if self.normalize_position:
                positions_2d = positions_2d - positions_2d.mean(axis=0)
                scale = np.abs(positions_2d).max()
                if scale > 0:
                    positions_2d = positions_2d / scale
            
            z_col = np.full((n_cells, 1), z_coord, dtype=np.float32)
            positions = np.hstack([positions_2d, z_col])
            z_coord_value = z_coord
        
        # ===== 细胞类型 =====
        cell_types = None
        cell_type_counts = None
        has_cell_types = False
        
        if self.cell_type_key and self.cell_type_key in adata.obs.columns:
            ct_series = adata.obs[self.cell_type_key]
            if self.cell_type_mapping:
                cell_types = ct_series.map(self.cell_type_mapping).values.astype(np.int32)
                # 处理未知类型
                cell_types = np.nan_to_num(cell_types, nan=-1).astype(np.int32)
                has_cell_types = True
                
                # 统计
                cell_type_counts = ct_series.value_counts().to_dict()
        
        # 构建输出数据
        slice_data = {
            'expression': expression,           # (n_cells, n_genes), float32
            'positions': positions,             # (n_cells, 3), float32
            'cell_types': cell_types,           # (n_cells,), int32 or None
            'slice_id': slice_id,
            'slice_idx': slice_idx,
            'n_cells': n_cells,
            'z_coord': z_coord_value,
        }
        
        slice_meta = SliceMetadata(
            slice_id=slice_id,
            n_cells=n_cells,
            n_genes=len(self.selected_genes),
            z_coord=z_coord_value,
            file_path="",  # 稍后填充
            has_cell_types=has_cell_types,
            cell_type_counts=cell_type_counts,
        )
        
        return slice_data, slice_meta


# ==================== 阶段二：Batch 生成（内存友好版）====================

class Stage2BatchGenerator:
    """
    阶段二：Batch 生成（内存友好版）
    
    特点：
    - 流式处理：不一次性加载所有切片到内存
    - 无放回采样：每个细胞只出现一次
    - 按细胞划分：同一切片的细胞可以分到不同的 train/val/test
    
    流程：
    1. 第一遍扫描：统计总细胞数，生成全局细胞索引
    2. 打乱索引并划分 train/val/test
    3. 流式读取切片，按需提取对应细胞
    4. 保存为 batch 文件
    """
    
    def __init__(
        self,
        batch_size: int = 1024,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        max_shard_size: int = 100_000,  # 每个 shard 文件最大细胞数
    ):
        """
        Args:
            batch_size: 每个 batch 的细胞数（训练时用）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
            max_shard_size: 每个 shard 文件的最大细胞数（控制文件大小）
        """
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_shard_size = max_shard_size
        
        np.random.seed(seed)
    
    def process(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = "parquet",  # "parquet", "pickle"
    ) -> Dict:
        """
        生成数据集
        
        Args:
            input_dir: 阶段一输出目录
            output_dir: 输出目录
            output_format: 输出格式 ("parquet" 推荐用于 HuggingFace)
        
        Returns:
            生成统计信息
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载元数据
        with open(input_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"元数据: {metadata['n_slices']} 切片, {metadata['total_cells']} 细胞")
        
        # ===== 第一步：构建全局细胞索引 =====
        logger.info("第一步：构建全局细胞索引...")
        cell_index = self._build_cell_index(metadata)
        
        total_cells = len(cell_index)
        logger.info(f"总细胞数: {total_cells}")
        
        # ===== 第二步：打乱并划分 train/val/test =====
        logger.info("第二步：划分数据集...")
        np.random.shuffle(cell_index)
        
        n_train = int(total_cells * self.train_ratio)
        n_val = int(total_cells * self.val_ratio)
        
        train_index = cell_index[:n_train]
        val_index = cell_index[n_train:n_train + n_val]
        test_index = cell_index[n_train + n_val:]
        
        logger.info(f"划分: train={len(train_index)}, val={len(val_index)}, test={len(test_index)}")
        
        # ===== 第三步：流式生成各 split =====
        logger.info("第三步：生成数据文件...")
        
        stats = {}
        
        if output_format == "parquet":
            stats['train'] = self._generate_parquet_split(
                train_index, metadata, output_dir / 'data', 'train'
            )
            stats['validation'] = self._generate_parquet_split(
                val_index, metadata, output_dir / 'data', 'validation'
            )
            stats['test'] = self._generate_parquet_split(
                test_index, metadata, output_dir / 'data', 'test'
            )
        else:
            stats['train'] = self._generate_pickle_split(
                train_index, metadata, output_dir / 'data', 'train'
            )
            stats['validation'] = self._generate_pickle_split(
                val_index, metadata, output_dir / 'data', 'validation'
            )
            stats['test'] = self._generate_pickle_split(
                test_index, metadata, output_dir / 'data', 'test'
            )
        
        # 保存数据集信息
        self._save_dataset_info(output_dir, metadata, stats)
        
        logger.info(f"阶段二完成！输出目录: {output_dir}")
        return stats
    
    def _build_cell_index(self, metadata: Dict) -> np.ndarray:
        """
        构建全局细胞索引
        
        返回: shape (total_cells, 2) 的数组
              每行是 [slice_idx, cell_idx_in_slice]
        """
        index_list = []
        
        for slice_info in metadata['slices']:
            slice_idx = slice_info['slice_idx'] if 'slice_idx' in slice_info else metadata['slices'].index(slice_info)
            n_cells = slice_info['n_cells']
            
            # 生成该切片所有细胞的索引
            slice_indices = np.column_stack([
                np.full(n_cells, slice_idx, dtype=np.int32),
                np.arange(n_cells, dtype=np.int32)
            ])
            index_list.append(slice_indices)
        
        return np.concatenate(index_list, axis=0)
    
    def _generate_parquet_split(
        self,
        cell_index: np.ndarray,
        metadata: Dict,
        output_dir: Path,
        split_name: str,
    ) -> Dict:
        """
        流式生成 parquet 格式的 split
        
        策略：
        - 按 slice_idx 排序，这样可以顺序读取切片
        - 每次只加载一个切片，提取需要的细胞
        - 分 shard 写入，控制文件大小
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow 未安装，使用 pickle 格式")
            return self._generate_pickle_split(cell_index, metadata, output_dir, split_name)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_cells = len(cell_index)
        if n_cells == 0:
            logger.warning(f"{split_name} 没有数据")
            return {'n_cells': 0, 'n_shards': 0}
        
        # 按 slice_idx 排序，方便顺序读取
        sort_order = np.lexsort((cell_index[:, 1], cell_index[:, 0]))
        sorted_index = cell_index[sort_order]
        
        # 记录每个切片需要哪些细胞
        slice_to_cells = {}
        for global_idx, (slice_idx, cell_idx) in enumerate(sorted_index):
            slice_idx = int(slice_idx)
            if slice_idx not in slice_to_cells:
                slice_to_cells[slice_idx] = []
            slice_to_cells[slice_idx].append((global_idx, int(cell_idx)))
        
        # 流式处理
        all_expression = []
        all_positions = []
        all_cell_types = []
        all_slice_ids = []
        
        shard_idx = 0
        cells_in_current_shard = 0
        
        def write_shard():
            nonlocal shard_idx, cells_in_current_shard
            nonlocal all_expression, all_positions, all_cell_types, all_slice_ids
            
            if not all_expression:
                return
            
            table = pa.table({
                'expression': pa.array(all_expression),
                'positions': pa.array(all_positions),
                'cell_types': pa.array(all_cell_types),
                'slice_ids': pa.array(all_slice_ids),
            })
            
            # 单文件或多文件命名
            if n_cells <= self.max_shard_size:
                output_file = output_dir / f"{split_name}.parquet"
            else:
                output_file = output_dir / f"{split_name}-{shard_idx:05d}.parquet"
            
            pq.write_table(table, output_file, compression='snappy')
            logger.info(f"写入 {output_file}: {len(all_expression)} 细胞")
            
            # 清空缓存
            all_expression = []
            all_positions = []
            all_cell_types = []
            all_slice_ids = []
            shard_idx += 1
            cells_in_current_shard = 0
        
        # 按切片顺序处理
        for slice_info in tqdm(metadata['slices'], desc=f"生成 {split_name}"):
            slice_idx = slice_info['slice_idx'] if 'slice_idx' in slice_info else metadata['slices'].index(slice_info)
            
            if slice_idx not in slice_to_cells:
                continue
            
            # 加载该切片
            with open(slice_info['file_path'], 'rb') as f:
                slice_data = pickle.load(f)
            
            # 提取需要的细胞
            cells_needed = slice_to_cells[slice_idx]
            cell_indices = [c[1] for c in cells_needed]
            
            expr = slice_data['expression'][cell_indices]
            pos = slice_data['positions'][cell_indices]
            ct = slice_data['cell_types']
            if ct is not None:
                ct = ct[cell_indices]
            else:
                ct = np.full(len(cell_indices), -1, dtype=np.int32)
            
            # 添加到缓存
            for i in range(len(cell_indices)):
                all_expression.append(expr[i].tolist())
                all_positions.append(pos[i].tolist())
                all_cell_types.append(int(ct[i]))
                all_slice_ids.append(slice_idx)
                cells_in_current_shard += 1
                
                # 达到 shard 大小，写入文件
                if cells_in_current_shard >= self.max_shard_size:
                    write_shard()
        
        # 写入剩余数据
        write_shard()
        
        return {
            'n_cells': n_cells,
            'n_shards': shard_idx,
            'format': 'parquet',
        }
    
    def _generate_pickle_split(
        self,
        cell_index: np.ndarray,
        metadata: Dict,
        output_dir: Path,
        split_name: str,
    ) -> Dict:
        """流式生成 pickle 格式的 split（按 batch 保存）"""
        output_dir = output_dir / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_cells = len(cell_index)
        if n_cells == 0:
            return {'n_cells': 0, 'n_batches': 0}
        
        # 按 slice_idx 排序
        sort_order = np.lexsort((cell_index[:, 1], cell_index[:, 0]))
        sorted_index = cell_index[sort_order]
        
        # 构建切片到细胞的映射
        slice_to_cells = {}
        for global_idx, (slice_idx, cell_idx) in enumerate(sorted_index):
            slice_idx = int(slice_idx)
            if slice_idx not in slice_to_cells:
                slice_to_cells[slice_idx] = []
            slice_to_cells[slice_idx].append((global_idx, int(cell_idx)))
        
        # 流式处理，按 batch 保存
        buffer_expr = []
        buffer_pos = []
        buffer_ct = []
        buffer_slice = []
        
        batch_idx = 0
        
        def write_batch():
            nonlocal batch_idx, buffer_expr, buffer_pos, buffer_ct, buffer_slice
            
            if not buffer_expr:
                return
            
            batch_data = {
                'expression': np.array(buffer_expr, dtype=np.float32),
                'positions': np.array(buffer_pos, dtype=np.float32),
                'cell_types': np.array(buffer_ct, dtype=np.int32),
                'slice_ids': np.array(buffer_slice, dtype=np.int32),
            }
            
            batch_file = output_dir / f"batch_{batch_idx:06d}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
            
            buffer_expr = []
            buffer_pos = []
            buffer_ct = []
            buffer_slice = []
            batch_idx += 1
        
        for slice_info in tqdm(metadata['slices'], desc=f"生成 {split_name}"):
            slice_idx = slice_info['slice_idx'] if 'slice_idx' in slice_info else metadata['slices'].index(slice_info)
            
            if slice_idx not in slice_to_cells:
                continue
            
            with open(slice_info['file_path'], 'rb') as f:
                slice_data = pickle.load(f)
            
            cells_needed = slice_to_cells[slice_idx]
            cell_indices = [c[1] for c in cells_needed]
            
            expr = slice_data['expression'][cell_indices]
            pos = slice_data['positions'][cell_indices]
            ct = slice_data['cell_types']
            if ct is not None:
                ct = ct[cell_indices]
            else:
                ct = np.full(len(cell_indices), -1, dtype=np.int32)
            
            for i in range(len(cell_indices)):
                buffer_expr.append(expr[i])
                buffer_pos.append(pos[i])
                buffer_ct.append(ct[i])
                buffer_slice.append(slice_idx)
                
                if len(buffer_expr) >= self.batch_size:
                    write_batch()
        
        # 写入剩余数据
        if buffer_expr:
            write_batch()
        
        return {
            'n_cells': n_cells,
            'n_batches': batch_idx,
            'batch_size': self.batch_size,
            'format': 'pickle',
        }
    
    def _save_dataset_info(self, output_dir, metadata, stats):
        """保存数据集信息（HuggingFace 格式）"""
        
        # 计算总数
        total_train = stats.get('train', {}).get('n_cells', 0)
        total_val = stats.get('validation', {}).get('n_cells', 0)
        total_test = stats.get('test', {}).get('n_cells', 0)
        
        # dataset_info.json（HuggingFace 格式）
        dataset_info = {
            "description": "AlphaSTomics spatial transcriptomics dataset",
            "citation": "",
            "homepage": "",
            "license": "MIT",
            "features": {
                "expression": {
                    "dtype": "float32",
                    "shape": [metadata['n_genes']],
                    "description": "Gene expression values (log-normalized, scaled)"
                },
                "positions": {
                    "dtype": "float32", 
                    "shape": [3],
                    "description": "3D spatial coordinates (x, y, z)"
                },
                "cell_types": {
                    "dtype": "int32",
                    "description": "Cell type label (-1 if unknown)"
                },
                "slice_ids": {
                    "dtype": "int32",
                    "description": "Source slice index"
                }
            },
            "splits": {
                "train": {"n_cells": total_train},
                "validation": {"n_cells": total_val},
                "test": {"n_cells": total_test},
            },
            "n_genes": metadata['n_genes'],
            "n_cell_types": metadata['n_cell_types'],
            "total_cells": total_train + total_val + total_test,
            "gene_names": metadata['gene_names'][:100],  # 只存前100个基因名
        }
        
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # README.md（HuggingFace Dataset Card）
        readme = f"""# AlphaSTomics Spatial Transcriptomics Dataset

## Description
Preprocessed spatial transcriptomics data for training AlphaSTomics model.

## Dataset Statistics
- **Number of genes**: {metadata['n_genes']}
- **Number of cell types**: {metadata['n_cell_types']}
- **Number of slices**: {metadata['n_slices']}
- **Total cells**: {total_train + total_val + total_test}

## Splits
| Split | Cells |
|-------|-------|
| Train | {total_train:,} |
| Validation | {total_val:,} |
| Test | {total_test:,} |

## Features
- `expression`: Gene expression values (log-normalized, z-score scaled), shape `({metadata['n_genes']},)`
- `positions`: 3D spatial coordinates `(x, y, z)`, shape `(3,)`
- `cell_types`: Cell type label (int, -1 if unknown)
- `slice_ids`: Source slice index (int)

## Usage
```python
from datasets import load_dataset

# 从 HuggingFace Hub 加载
dataset = load_dataset("your-username/alphastomics-dataset")

# 或从本地加载
dataset = load_dataset("parquet", data_files="./data/*.parquet")

# 遍历数据
for sample in dataset['train']:
    expression = sample['expression']  # list of float32
    positions = sample['positions']    # [x, y, z]
    cell_type = sample['cell_types']
```

## Preprocessing
- Log-normalization: `target_sum={metadata.get('preprocessing_params', {}).get('target_sum', 10000)}`
- Z-score scaling: `max_value={metadata.get('preprocessing_params', {}).get('max_scale_value', 10)}`
- Position normalization: `{metadata.get('preprocessing_params', {}).get('normalize_position', True)}`

## Sampling Strategy
- **No replacement**: Each cell appears exactly once
- **Cell-level split**: Cells from the same slice can be in different splits
- **Random shuffle**: Cells are randomly shuffled before splitting
"""
        
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme)
        
        # 保存完整元数据
        with open(output_dir / 'full_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # 保存完整基因列表
        with open(output_dir / 'genes.txt', 'w') as f:
            f.write('\n'.join(metadata['gene_names']))
        
        # 保存详细统计
        stats_info = {
            'train': stats.get('train', {}),
            'validation': stats.get('validation', {}),
            'test': stats.get('test', {}),
            'batch_size': self.batch_size,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'seed': self.seed,
        }
        with open(output_dir / 'stats.json', 'w') as f:
            json.dump(stats_info, f, indent=2)


def _json_serializer(obj):
    """JSON 序列化器，处理 numpy 类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="AlphaSTomics 两阶段数据预处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 阶段一：预处理切片
  python preprocess.py stage1 --input_dir ./raw --output_dir ./processed
  
  # 阶段二：生成数据集（无放回采样，按细胞划分）
  python preprocess.py stage2 --input_dir ./processed --output_dir ./dataset
  
  # 一步完成
  python preprocess.py all --raw_dir ./raw --output_dir ./dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='stage', help='处理阶段')
    
    # ===== 阶段一参数 =====
    stage1_parser = subparsers.add_parser('stage1', help='切片预处理')
    stage1_parser.add_argument('--input_dir', '-i', required=True, help='输入目录（包含 h5ad 文件）')
    stage1_parser.add_argument('--output_dir', '-o', required=True, help='输出目录')
    stage1_parser.add_argument('--gene_list', '-g', default=None, help='固定基因列表文件')
    stage1_parser.add_argument('--position_key', default='spatial', help='坐标的 obsm key')
    stage1_parser.add_argument('--position_is_3d', action='store_true', help='坐标是否为 3D')
    stage1_parser.add_argument('--z_spacing', type=float, default=10.0, help='切片间 z 间距')
    stage1_parser.add_argument('--cell_type_key', default='cell_type', help='细胞类型的 obs key')
    stage1_parser.add_argument('--no_scale', action='store_true', help='不做 z-score 标准化')
    stage1_parser.add_argument('--no_normalize_position', action='store_true', help='不标准化坐标')
    
    # ===== 阶段二参数 =====
    stage2_parser = subparsers.add_parser('stage2', help='数据集生成（内存友好，无放回采样）')
    stage2_parser.add_argument('--input_dir', '-i', required=True, help='阶段一输出目录')
    stage2_parser.add_argument('--output_dir', '-o', required=True, help='输出目录')
    stage2_parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    stage2_parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    stage2_parser.add_argument('--format', choices=['parquet', 'pickle'], default='parquet', help='输出格式')
    stage2_parser.add_argument('--batch_size', type=int, default=1024, help='pickle 格式的 batch 大小')
    stage2_parser.add_argument('--max_shard_size', type=int, default=100000, help='parquet 每个分片最大细胞数')
    stage2_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # ===== 一步完成 =====
    all_parser = subparsers.add_parser('all', help='一步完成两个阶段')
    all_parser.add_argument('--raw_dir', '-r', required=True, help='原始 h5ad 目录')
    all_parser.add_argument('--output_dir', '-o', required=True, help='最终输出目录')
    all_parser.add_argument('--gene_list', '-g', default=None, help='固定基因列表文件')
    all_parser.add_argument('--position_key', default='spatial', help='坐标的 obsm key')
    all_parser.add_argument('--position_is_3d', action='store_true', help='坐标是否为 3D')
    all_parser.add_argument('--z_spacing', type=float, default=10.0, help='切片间 z 间距')
    all_parser.add_argument('--cell_type_key', default='cell_type', help='细胞类型的 obs key')
    all_parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    all_parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    all_parser.add_argument('--format', choices=['parquet', 'pickle'], default='parquet', help='输出格式')
    all_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    if args.stage is None:
        parser.print_help()
        return
    
    if args.stage == 'stage1':
        preprocessor = Stage1Preprocessor(
            gene_list_file=args.gene_list,
            position_key=args.position_key,
            position_is_3d=args.position_is_3d,
            z_spacing=args.z_spacing,
            cell_type_key=args.cell_type_key,
            scale=not args.no_scale,
            normalize_position=not args.no_normalize_position,
        )
        preprocessor.process(args.input_dir, args.output_dir)
    
    elif args.stage == 'stage2':
        generator = Stage2BatchGenerator(
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_shard_size=args.max_shard_size,
        )
        generator.process(args.input_dir, args.output_dir, output_format=args.format)
    
    elif args.stage == 'all':
        # 中间目录
        intermediate_dir = Path(args.output_dir) / '_intermediate_slices'
        
        # 阶段一
        logger.info("=" * 50)
        logger.info("阶段一：切片预处理")
        logger.info("=" * 50)
        
        preprocessor = Stage1Preprocessor(
            gene_list_file=args.gene_list,
            position_key=args.position_key,
            position_is_3d=args.position_is_3d,
            z_spacing=args.z_spacing,
            cell_type_key=args.cell_type_key,
        )
        preprocessor.process(args.raw_dir, str(intermediate_dir))
        
        # 阶段二
        logger.info("")
        logger.info("=" * 50)
        logger.info("阶段二：数据集生成（无放回采样）")
        logger.info("=" * 50)
        
        generator = Stage2BatchGenerator(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        generator.process(str(intermediate_dir), args.output_dir, output_format=args.format)
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("全部完成！")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info("=" * 50)


if __name__ == '__main__':
    main()
