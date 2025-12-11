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
    file_path: str  # 主文件路径（单文件或第一个分块）
    file_paths: Optional[List[str]] = None  # 多文件切片的所有文件路径
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
        log_normalize: bool = True,
        scale: bool = True,
        normalize_position: bool = False,
        target_sum: float = 1e4,
        max_scale_value: float = 10.0,
        max_cells_per_file: int = 50000,
    ):
        """
        Args:
            gene_list_file: 固定基因列表文件（每行一个基因）
            position_key: obsm 中坐标的 key
            position_is_3d: 坐标是否已经是 3D
            z_spacing: 切片间 z 间距（仅 2D 时使用）
            cell_type_key: obs 中细胞类型的 key
            log_normalize: 是否做 log normalize（如果数据已经做过可设为 False）
            scale: 是否做 z-score 标准化
            normalize_position: 是否标准化坐标
            target_sum: normalize_total 的目标值
            max_scale_value: scale 后的最大值截断
            max_cells_per_file: 每个 parquet 文件最大细胞数（防止 List index overflow）
        """
        self.gene_list_file = gene_list_file
        self.position_key = position_key
        self.position_is_3d = position_is_3d
        self.z_spacing = z_spacing
        self.cell_type_key = cell_type_key
        self.log_normalize = log_normalize
        self.scale = scale
        self.normalize_position = normalize_position
        self.target_sum = target_sum
        self.max_scale_value = max_scale_value
        self.max_cells_per_file = max_cells_per_file
        
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
        file_pattern: str = "*.h5ad",
    ) -> DatasetMetadata:
        """
        处理所有切片
        
        Args:
            input_dir: 输入目录（包含 h5ad 文件）
            output_dir: 输出目录
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
        
        # ===== 第一遍扫描：确定基因列表和细胞类型（仅读取元数据）=====
        logger.info("第一遍扫描：确定基因列表和细胞类型...")
        
        # 确定基因列表
        if self.fixed_gene_list is not None:
            self.selected_genes = self.fixed_gene_list
            logger.info(f"使用固定基因列表: {len(self.selected_genes)} 个基因")
        else:
            # 流式扫描，找共同基因
            common_genes = None
            for fpath in tqdm(h5ad_files, desc="扫描基因"):
                adata = anndata.read_h5ad(fpath)
                adata.var_names_make_unique()
                current_genes = set(adata.var_names)
                
                if common_genes is None:
                    common_genes = current_genes
                else:
                    common_genes = common_genes.intersection(current_genes)
                
                del adata  # 立即释放内存
            
            self.selected_genes = sorted(list(common_genes))
            logger.info(f"使用共同基因: {len(self.selected_genes)} 个基因")
        
        # 收集细胞类型（流式扫描）
        all_cell_types = set()
        if self.cell_type_key:
            for fpath in tqdm(h5ad_files, desc="扫描细胞类型"):
                adata = anndata.read_h5ad(fpath)
                if self.cell_type_key in adata.obs.columns:
                    all_cell_types.update(adata.obs[self.cell_type_key].dropna().unique())
                del adata  # 立即释放内存
        
        if all_cell_types:
            self.cell_type_mapping = {ct: i for i, ct in enumerate(sorted(all_cell_types))}
            logger.info(f"细胞类型数量: {len(self.cell_type_mapping)}")
        
        # ===== 第二遍处理：逐片处理并保存（流式）=====
        logger.info("第二遍处理：逐片预处理并保存...")
        
        slices_metadata = []
        total_cells = 0
        skipped_count = 0
        
        for i, fpath in enumerate(tqdm(h5ad_files, desc="处理切片")):
            slice_id = fpath.stem
            
            # 检查是否已处理过（断点续传支持）
            output_file = output_dir / f"slice_{i:04d}_{slice_id}.parquet"
            base_name = f"slice_{i:04d}_{slice_id}"
            
            # 检查单文件或分块文件是否存在
            part_files = sorted(output_dir.glob(f"{base_name}_part*.parquet"))
            existing_files = part_files if part_files else ([output_file] if output_file.exists() else [])
            
            if existing_files:
                # 加载已有元数据（需要统计 total_cells）
                try:
                    import pyarrow.parquet as pq
                    
                    n_cells = 0
                    schema_names = None
                    total_file_size = 0
                    
                    # 遍历所有文件统计细胞数
                    for ef in existing_files:
                        file_size = ef.stat().st_size
                        if file_size == 0:
                            raise ValueError(f"文件大小为 0 字节: {ef}")
                        total_file_size += file_size
                        
                        pf = pq.ParquetFile(ef)
                        n_cells += pf.metadata.num_rows
                        if schema_names is None:
                            schema_names = pf.schema.names
                        del pf
                    
                    if n_cells == 0:
                        raise ValueError(f"文件中细胞数为 0")
                    
                    slice_meta = SliceMetadata(
                        slice_id=slice_id,
                        n_cells=n_cells,
                        n_genes=len(self.selected_genes),
                        file_path=str(existing_files[0]),
                        file_paths=[str(f) for f in existing_files] if len(existing_files) > 1 else None,
                        has_cell_types='cell_type' in schema_names,
                    )
                    slices_metadata.append(slice_meta)
                    total_cells += n_cells
                    skipped_count += 1
                    
                    files_desc = f"{len(existing_files)} 个文件" if len(existing_files) > 1 else existing_files[0].name
                    logger.info(f"跳过已处理的切片: {slice_id} ({files_desc}, {n_cells} 细胞, {total_file_size/1024/1024:.1f}MB)")
                    continue  # 跳过已处理的切片
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"无法读取已有文件，将重新处理。\n"
                        f"  错误类型: {type(e).__name__}\n"
                        f"  错误信息: {e}\n"
                        f"  详细堆栈:\n{traceback.format_exc()}"
                    )
                    # 删除损坏的文件
                    for ef in existing_files:
                        try:
                            ef.unlink()
                            logger.info(f"  删除损坏文件: {ef.name}")
                        except:
                            pass
            
            # 加载单个切片
            adata = anndata.read_h5ad(fpath)
            adata.var_names_make_unique()
            
            # 处理单个切片
            slice_data, slice_meta = self._process_single_slice(
                adata, slice_id, i
            )
            
            # 立即释放内存
            del adata
            
            # 保存为 parquet 格式（高压缩，支持分块）
            saved_files = self._save_slice_parquet(slice_data, output_file)
            
            slice_meta.file_path = saved_files[0]
            slice_meta.file_paths = saved_files if len(saved_files) > 1 else None
            slices_metadata.append(slice_meta)
            total_cells += slice_meta.n_cells
        
        if skipped_count > 0:
            logger.info(f"断点续传：跳过了 {skipped_count} 个已处理的切片")
        
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
    
    def _save_slice_parquet(self, slice_data: Dict, output_file: Path) -> List[str]:
        """
        将切片数据保存为 parquet 格式（高压缩）
        
        如果细胞数超过 max_cells_per_file，会自动分块保存为多个文件：
        - slice_0000_xxx.parquet（小切片，单文件）
        - slice_0000_xxx_part000.parquet, _part001.parquet, ...（大切片，多文件）
        
        Returns:
            保存的文件路径列表
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("需要安装 pyarrow: pip install pyarrow")
        
        n_cells = slice_data['n_cells']
        max_cells = self.max_cells_per_file
        
        # 判断是否需要分块
        if n_cells <= max_cells:
            # 小切片：单文件保存
            self._write_parquet_chunk(slice_data, 0, n_cells, output_file)
            return [str(output_file)]
        
        # 大切片：分块保存
        n_parts = (n_cells + max_cells - 1) // max_cells
        saved_files = []
        
        # 修改文件名格式：slice_0000_xxx_part000.parquet
        base_name = output_file.stem  # slice_0000_xxx
        parent_dir = output_file.parent
        
        for part_idx in range(n_parts):
            start_idx = part_idx * max_cells
            end_idx = min((part_idx + 1) * max_cells, n_cells)
            
            part_file = parent_dir / f"{base_name}_part{part_idx:03d}.parquet"
            self._write_parquet_chunk(slice_data, start_idx, end_idx, part_file)
            saved_files.append(str(part_file))
            
            logger.debug(f"  保存分块 {part_idx+1}/{n_parts}: {end_idx - start_idx} 细胞 -> {part_file.name}")
        
        logger.info(f"  大切片分块保存: {n_cells} 细胞 -> {n_parts} 个文件")
        return saved_files
    
    def _write_parquet_chunk(self, slice_data: Dict, start_idx: int, end_idx: int, output_file: Path):
        """写入单个 parquet 文件块"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        rows = []
        for i in range(start_idx, end_idx):
            row = {
                'expression': slice_data['expression'][i].tolist(),
                'positions': slice_data['positions'][i].tolist(),
                'slice_id': slice_data['slice_id'],
                'slice_idx': slice_data['slice_idx'],
            }
            
            # 细胞类型（可选）
            if slice_data['cell_types'] is not None:
                row['cell_type'] = int(slice_data['cell_types'][i])
            else:
                row['cell_type'] = -1
            
            rows.append(row)
        
        # 保存为 parquet（zstd 最高压缩）
        table = pa.Table.from_pylist(rows)
        pq.write_table(
            table, 
            output_file, 
            compression='zstd',
            compression_level=9  # 最高压缩级别
        )
    
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
            
            # Log normalize（每片独立，可选）
            if self.log_normalize:
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
        else:
            positions_2d = positions_raw[:, :2].copy().astype(np.float32)
            
            if self.normalize_position:
                positions_2d = positions_2d - positions_2d.mean(axis=0)
                scale = np.abs(positions_2d).max()
                if scale > 0:
                    positions_2d = positions_2d / scale
            
            # 2D 数据：添加切片索引作为 z 坐标
            z_col = np.full((n_cells, 1), slice_idx * self.z_spacing, dtype=np.float32)
            positions = np.hstack([positions_2d, z_col])
        
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
        }
        
        slice_meta = SliceMetadata(
            slice_id=slice_id,
            n_cells=n_cells,
            n_genes=len(self.selected_genes),
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
        input_dir: Union[str, List[str]],
        output_dir: str,
        output_format: str = "parquet",  # "parquet", "pickle"
    ) -> Dict:
        """
        生成数据集（支持多目录合并）
        
        Args:
            input_dir: 阶段一输出目录，可以是单个路径或多个路径的列表
            output_dir: 输出目录
            output_format: 输出格式 ("parquet" 推荐用于 HuggingFace)
        
        Returns:
            生成统计信息
        """
        # 支持多目录输入
        if isinstance(input_dir, str):
            input_dirs = [Path(input_dir)]
        else:
            input_dirs = [Path(d) for d in input_dir]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载并合并元数据
        metadata = self._load_and_merge_metadata(input_dirs)
        
        logger.info(f"合并后元数据: {metadata['n_slices']} 切片, {metadata['total_cells']} 细胞")
        
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
    
    def _load_and_merge_metadata(self, input_dirs: List[Path]) -> Dict:
        """
        加载并合并多个目录的元数据
        
        Args:
            input_dirs: Stage1 输出目录列表
            
        Returns:
            合并后的元数据
        """
        all_metadata = []
        for input_dir in input_dirs:
            meta_path = input_dir / 'metadata.pkl'
            if not meta_path.exists():
                raise FileNotFoundError(f"未找到元数据文件: {meta_path}")
            
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            all_metadata.append(meta)
            logger.info(f"加载 {input_dir}: {meta['n_slices']} 切片, {meta['total_cells']} 细胞")
        
        if len(all_metadata) == 1:
            return all_metadata[0]
        
        # 合并多个元数据
        merged = {
            'n_slices': 0,
            'total_cells': 0,
            'n_genes': all_metadata[0]['n_genes'],
            'gene_names': all_metadata[0]['gene_names'],
            'n_cell_types': 0,
            'cell_type_mapping': {},
            'slices': [],
            'position_key': all_metadata[0].get('position_key', 'spatial'),
            'position_is_3d': all_metadata[0].get('position_is_3d', False),
        }
        
        # 验证基因列表一致性
        ref_genes = set(all_metadata[0]['gene_names'])
        for i, meta in enumerate(all_metadata[1:], 1):
            if set(meta['gene_names']) != ref_genes:
                logger.warning(f"警告: 目录 {i} 的基因列表与目录 0 不一致！")
                logger.warning(f"  目录 0: {len(ref_genes)} 个基因")
                logger.warning(f"  目录 {i}: {len(meta['gene_names'])} 个基因")
                # 检查是否顺序也一致
                if meta['gene_names'] != all_metadata[0]['gene_names']:
                    raise ValueError(
                        f"基因列表不一致！请确保所有 Stage1 使用相同的 gene_list_file。"
                        f"\n目录 0 基因数: {len(all_metadata[0]['gene_names'])}"
                        f"\n目录 {i} 基因数: {len(meta['gene_names'])}"
                    )
        
        # 合并细胞类型映射
        all_cell_types = set()
        for meta in all_metadata:
            if meta.get('cell_type_mapping'):
                all_cell_types.update(meta['cell_type_mapping'].keys())
        
        if all_cell_types:
            merged['cell_type_mapping'] = {ct: i for i, ct in enumerate(sorted(all_cell_types))}
            merged['n_cell_types'] = len(merged['cell_type_mapping'])
        
        # 合并切片，重新编号 slice_idx
        global_slice_idx = 0
        for meta in all_metadata:
            for slice_info in meta['slices']:
                new_slice_info = slice_info.copy()
                new_slice_info['slice_idx'] = global_slice_idx
                # 如果细胞类型映射有变化，需要更新
                if meta.get('cell_type_mapping') and merged['cell_type_mapping']:
                    new_slice_info['_old_cell_type_mapping'] = meta['cell_type_mapping']
                merged['slices'].append(new_slice_info)
                merged['total_cells'] += slice_info['n_cells']
                global_slice_idx += 1
        
        merged['n_slices'] = global_slice_idx
        
        logger.info(f"合并完成: {merged['n_slices']} 切片, {merged['total_cells']} 细胞, "
                   f"{merged['n_genes']} 基因, {merged['n_cell_types']} 细胞类型")
        
        return merged
    
    def _load_slice_data(self, slice_info: Dict, row_indices: List[int]) -> Dict:
        """
        加载切片数据的指定行（按索引）
        
        支持单文件和多文件（分块）切片。
        
        Args:
            slice_info: 切片元数据字典，包含 file_path 和可选的 file_paths
            row_indices: 需要读取的行索引列表（相对于整个切片的全局索引）
        
        Returns:
            切片数据字典（仅包含指定的行）
        """
        file_paths = slice_info.get('file_paths')
        
        if file_paths and len(file_paths) > 1:
            # 多文件切片：需要计算每个索引落在哪个文件
            return self._load_multi_file_slice(file_paths, row_indices)
        
        # 单文件切片
        file_path = Path(slice_info['file_path'])
        
        if file_path.suffix == '.parquet':
            # 从 parquet 加载
            try:
                import pyarrow.parquet as pq
                import pyarrow.compute as pc
                import pyarrow as pa
            except ImportError:
                raise ImportError("需要安装 pyarrow: pip install pyarrow")
            
            # 使用 memory_map=True 让操作系统按需加载数据页面，减少实际内存占用
            # 只读取需要的列，进一步减少内存
            table = pq.read_table(
                file_path, 
                columns=['expression', 'positions', 'cell_type', 'slice_id', 'slice_idx'],
                memory_map=True  # 内存映射，按需加载
            )
            
            # 提取指定行
            indices = pa.array(row_indices, type=pa.int64())
            table = pc.take(table, indices)
            
            # 直接从 Arrow 转 NumPy，不经过 Pandas
            n_cells = len(table)
            
            # 表达量：List<float> → NumPy array
            expression = np.array(table['expression'].to_pylist(), dtype=np.float32)
            
            # 坐标：List<float> → NumPy array
            positions = np.array(table['positions'].to_pylist(), dtype=np.float32)
            
            # 细胞类型：int → NumPy array
            cell_types = table['cell_type'].to_numpy().astype(np.int32)
            
            # 元数据（从第一行获取）
            slice_id = table['slice_id'][0].as_py()
            slice_idx = table['slice_idx'][0].as_py()
            
            return {
                'expression': expression,
                'positions': positions,
                'cell_types': cell_types,
                'slice_id': slice_id,
                'slice_idx': slice_idx,
                'n_cells': n_cells,
            }
        else:
            # 兼容旧的 pickle 格式
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 提取指定行的子集
            return {
                'expression': data['expression'][row_indices],
                'positions': data['positions'][row_indices],
                'cell_types': data['cell_types'][row_indices] if data['cell_types'] is not None else None,
                'slice_id': data['slice_id'],
                'slice_idx': data['slice_idx'],
                'n_cells': len(row_indices),
            }
    
    def _load_multi_file_slice(self, file_paths: List[str], row_indices: List[int]) -> Dict:
        """
        加载多文件（分块）切片的指定行
        
        Args:
            file_paths: 所有分块文件的路径列表
            row_indices: 需要读取的行索引（相对于整个切片的全局索引）
        
        Returns:
            切片数据字典
        """
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pyarrow as pa
        
        # 1. 先获取每个文件的行数范围
        file_row_ranges = []  # [(start, end, file_path), ...]
        current_row = 0
        
        for fp in file_paths:
            pf = pq.ParquetFile(fp)
            n_rows = pf.metadata.num_rows
            file_row_ranges.append((current_row, current_row + n_rows, fp))
            current_row += n_rows
            del pf
        
        # 2. 确定每个索引落在哪个文件
        file_to_local_indices = {}  # {file_path: [(global_idx, local_idx), ...]}
        
        for global_idx in row_indices:
            for start, end, fp in file_row_ranges:
                if start <= global_idx < end:
                    local_idx = global_idx - start
                    if fp not in file_to_local_indices:
                        file_to_local_indices[fp] = []
                    file_to_local_indices[fp].append((global_idx, local_idx))
                    break
        
        # 3. 从每个相关文件读取数据
        all_expression = []
        all_positions = []
        all_cell_types = []
        slice_id = None
        slice_idx = None
        
        # 按原始顺序收集结果
        results_by_global_idx = {}
        
        for fp, indices_info in file_to_local_indices.items():
            local_indices = [li for _, li in indices_info]
            
            # 读取文件
            table = pq.read_table(
                fp,
                columns=['expression', 'positions', 'cell_type', 'slice_id', 'slice_idx'],
                memory_map=True
            )
            
            # 提取行
            indices_arr = pa.array(local_indices, type=pa.int64())
            subtable = pc.take(table, indices_arr)
            
            # 获取元数据
            if slice_id is None:
                slice_id = subtable['slice_id'][0].as_py()
                slice_idx = subtable['slice_idx'][0].as_py()
            
            # 保存结果
            for i, (global_idx, _) in enumerate(indices_info):
                results_by_global_idx[global_idx] = {
                    'expression': subtable['expression'][i].as_py(),
                    'positions': subtable['positions'][i].as_py(),
                    'cell_type': subtable['cell_type'][i].as_py(),
                }
        
        # 4. 按原始顺序组装结果
        for global_idx in row_indices:
            r = results_by_global_idx[global_idx]
            all_expression.append(r['expression'])
            all_positions.append(r['positions'])
            all_cell_types.append(r['cell_type'])
        
        return {
            'expression': np.array(all_expression, dtype=np.float32),
            'positions': np.array(all_positions, dtype=np.float32),
            'cell_types': np.array(all_cell_types, dtype=np.int32),
            'slice_id': slice_id,
            'slice_idx': slice_idx,
            'n_cells': len(row_indices),
        }
    
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
            
            # ===== 优化：只读取需要的行 =====
            cells_needed = slice_to_cells[slice_idx]
            cell_indices = [c[1] for c in cells_needed]
            
            # 直接按索引读取（节省内存，支持多文件切片）
            slice_data = self._load_slice_data(slice_info, row_indices=cell_indices)
            
            expr = slice_data['expression']  # 已经是子集
            pos = slice_data['positions']    # 已经是子集
            ct = slice_data['cell_types']    # 已经是子集
            if ct is None:
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
            
            # ===== 优化：只读取需要的行 =====
            cells_needed = slice_to_cells[slice_idx]
            cell_indices = [c[1] for c in cells_needed]
            
            # 直接按索引读取（支持多文件切片）
            slice_data = self._load_slice_data(slice_info, row_indices=cell_indices)
            
            expr = slice_data['expression']  # 已经是子集
            pos = slice_data['positions']    # 已经是子集
            ct = slice_data['cell_types']    # 已经是子集
            if ct is None:
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
    stage1_parser.add_argument('--max_cells_per_file', type=int, default=50000, 
                               help='每个 parquet 文件最大细胞数（防止大文件 List index overflow 错误）')
    
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
            max_cells_per_file=args.max_cells_per_file,
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
