"""
Embedding 提取器
用于从训练好的 AlphaSTomics 模型中提取表达量和位置的 embedding

支持以下提取模式:
1. encoder: 仅通过输入编码器提取
2. transformer: 通过完整 Transformer 处理后提取
3. all_layers: 提取所有 Transformer 层的中间表示

用法示例:
    ```python
    from alphastomics.utils.embedding_extractor import EmbeddingExtractor
    
    # 初始化提取器
    extractor = EmbeddingExtractor(model, device='cuda')
    
    # 提取 embedding
    result = extractor.extract(
        expression=expression_tensor,  # (B, N, G)
        positions=position_tensor,      # (B, N, 3)
        node_mask=mask_tensor,          # (B, N)
        diffusion_time=0.0,             # 使用 t=0 表示原始数据
        mode='transformer'
    )
    
    # 获取结果
    expr_embedding = result['expression_embedding']  # (B, N, D)
    pos_embedding = result['position_embedding']     # (B, N, 3)
    ```
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
import numpy as np
from pathlib import Path


class EmbeddingExtractor:
    """
    从 AlphaSTomics 模型提取 embedding 的工具类
    
    支持三种提取模式:
    - 'encoder': 仅使用输入编码器，得到初始 embedding
    - 'transformer': 通过所有 Transformer 层后的最终 embedding
    - 'all_layers': 返回每一层的 embedding（用于分析）
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ):
        """
        初始化 embedding 提取器
        
        Args:
            model: 训练好的 AlphaSTomics 模型
            device: 计算设备，默认自动检测
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def extract(
        self,
        expression: torch.Tensor,
        positions: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        diffusion_time: float = 0.0,
        mode: str = 'transformer',
        return_predictions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        提取 expression 和 position 的 embedding
        
        Args:
            expression: (B, N, G) 基因表达量
            positions: (B, N, 3) 3D 空间坐标
            node_mask: (B, N) 有效节点掩码，可选
            diffusion_time: 扩散时间，默认 0.0（原始数据）
            mode: 提取模式
                - 'encoder': 仅通过输入编码器
                - 'transformer': 通过完整 Transformer（推荐）
                - 'all_layers': 返回所有层的 embedding
            return_predictions: 是否同时返回模型预测
        
        Returns:
            Dict 包含:
                - 'expression_embedding': (B, N, D) 表达量 embedding
                - 'position_embedding': (B, N, 3) 位置 embedding
                - 'time_embedding': (B, D_t) 时间 embedding
                - 'layer_embeddings' (仅 all_layers 模式): List of embeddings
                - 'pred_expression' (仅 return_predictions=True): 预测的表达量
                - 'pred_positions' (仅 return_predictions=True): 预测的位置
        """
        # 移动到设备
        expression = expression.to(self.device)
        positions = positions.to(self.device)
        
        batch_size, num_nodes, num_genes = expression.shape
        
        # 处理掩码
        if node_mask is None:
            node_mask = torch.ones(batch_size, num_nodes, device=self.device)
        else:
            node_mask = node_mask.to(self.device)
        
        # 创建扩散时间张量
        t = torch.full((batch_size, 1), diffusion_time, device=self.device)
        
        # 通过输入编码器
        x = self.model.mlp_in_expression(expression)   # (B, N, hidden_dim)
        y = self.model.mlp_in_diffusion_time(t)        # (B, 1, time_dim) or (B, time_dim)
        pos = self.model.position_mlp(positions)       # (B, N, 3)
        
        result = {
            'time_embedding': y.squeeze(1) if y.dim() == 3 else y,
        }
        
        if mode == 'encoder':
            # 仅返回编码器输出
            result['expression_embedding'] = x
            result['position_embedding'] = pos
            
        elif mode == 'transformer':
            # 通过所有 Transformer 层
            for layer in self.model.transformer_layers:
                x, pos, y = layer(
                    expression_features=x,
                    diffusion_time=y,
                    position_features=pos
                )
            result['expression_embedding'] = x
            result['position_embedding'] = pos
            
        elif mode == 'all_layers':
            # 收集所有层的 embedding
            expr_layers = [x.clone()]
            pos_layers = [pos.clone()]
            
            for layer in self.model.transformer_layers:
                x, pos, y = layer(
                    expression_features=x,
                    diffusion_time=y,
                    position_features=pos
                )
                expr_layers.append(x.clone())
                pos_layers.append(pos.clone())
            
            result['expression_embedding'] = x  # 最终层
            result['position_embedding'] = pos
            result['expression_layer_embeddings'] = expr_layers  # 所有层
            result['position_layer_embeddings'] = pos_layers
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'encoder', 'transformer', or 'all_layers'")
        
        # 可选：返回模型预测
        if return_predictions:
            # 输出表达量
            out_expression = self.model.mlp_out_expression(x)
            
            # 输出位置
            pos_features = self.model.mlp_out_pos_features(x)
            norm = torch.norm(pos, dim=-1, keepdim=True)
            new_norm = self.model.mlp_out_position_norm(
                torch.cat([pos_features, pos, norm], dim=-1)
            )
            out_position = pos * new_norm / (norm + self.model.positionMLP_eps)
            
            # 应用掩码
            out_expression = out_expression * node_mask.unsqueeze(-1)
            out_position = out_position * node_mask.unsqueeze(-1)
            
            result['pred_expression'] = out_expression
            result['pred_positions'] = out_position
        
        # 应用掩码到 embedding
        result['expression_embedding'] = result['expression_embedding'] * node_mask.unsqueeze(-1)
        result['position_embedding'] = result['position_embedding'] * node_mask.unsqueeze(-1)
        
        return result
    
    @torch.no_grad()
    def extract_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        mode: str = 'transformer',
        diffusion_time: float = 0.0,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        批量提取 embedding，适用于大规模数据
        
        Args:
            dataloader: PyTorch DataLoader
            mode: 提取模式 ('encoder', 'transformer', 'all_layers')
            diffusion_time: 扩散时间
            max_samples: 最大样本数，None 表示全部
            verbose: 是否显示进度
        
        Returns:
            Dict 包含:
                - 'expression_embeddings': (N_total, D) numpy array
                - 'position_embeddings': (N_total, 3) numpy array
                - 'sample_indices': 样本索引
        """
        all_expr_embeddings = []
        all_pos_embeddings = []
        all_sample_indices = []
        
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 从 batch 中提取数据（适配你的 DataHolder 格式）
            if hasattr(batch, 'expression') and hasattr(batch, 'positions'):
                expression = batch.expression
                positions = batch.positions
                node_mask = getattr(batch, 'node_mask', None)
            elif isinstance(batch, dict):
                expression = batch['expression']
                positions = batch['positions']
                node_mask = batch.get('node_mask', None)
            elif isinstance(batch, (list, tuple)):
                expression, positions = batch[0], batch[1]
                node_mask = batch[2] if len(batch) > 2 else None
            else:
                raise ValueError(f"Unknown batch format: {type(batch)}")
            
            # 提取 embedding
            result = self.extract(
                expression=expression,
                positions=positions,
                node_mask=node_mask,
                diffusion_time=diffusion_time,
                mode=mode if mode != 'all_layers' else 'transformer',
            )
            
            # 收集结果
            batch_size, num_nodes, _ = expression.shape
            expr_emb = result['expression_embedding'].cpu().numpy()
            pos_emb = result['position_embedding'].cpu().numpy()
            
            # 如果有掩码，只保留有效节点
            if node_mask is not None:
                mask = node_mask.cpu().numpy().astype(bool)
                for b in range(batch_size):
                    valid_expr = expr_emb[b][mask[b]]
                    valid_pos = pos_emb[b][mask[b]]
                    all_expr_embeddings.append(valid_expr)
                    all_pos_embeddings.append(valid_pos)
                    all_sample_indices.extend([total_samples + b] * valid_expr.shape[0])
            else:
                all_expr_embeddings.append(expr_emb.reshape(-1, expr_emb.shape[-1]))
                all_pos_embeddings.append(pos_emb.reshape(-1, pos_emb.shape[-1]))
                all_sample_indices.extend(
                    [i for b in range(batch_size) for i in [total_samples + b] * num_nodes]
                )
            
            total_samples += batch_size
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"Processed {total_samples} samples...")
            
            if max_samples is not None and total_samples >= max_samples:
                break
        
        return {
            'expression_embeddings': np.concatenate(all_expr_embeddings, axis=0),
            'position_embeddings': np.concatenate(all_pos_embeddings, axis=0),
            'sample_indices': np.array(all_sample_indices),
        }
    
    def save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        save_path: Union[str, Path],
        format: str = 'npz',
    ):
        """
        保存提取的 embedding 到文件
        
        Args:
            embeddings: extract_batch 返回的 embedding 字典
            save_path: 保存路径
            format: 保存格式 ('npz', 'pt', 'h5')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            np.savez_compressed(save_path.with_suffix('.npz'), **embeddings)
        elif format == 'pt':
            torch.save(
                {k: torch.from_numpy(v) for k, v in embeddings.items()},
                save_path.with_suffix('.pt')
            )
        elif format == 'h5':
            try:
                import h5py
                with h5py.File(save_path.with_suffix('.h5'), 'w') as f:
                    for key, value in embeddings.items():
                        f.create_dataset(key, data=value, compression='gzip')
            except ImportError:
                raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'pt', or 'h5'")
        
        print(f"Embeddings saved to {save_path}")
    
    @staticmethod
    def load_embeddings(
        load_path: Union[str, Path],
        format: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        从文件加载 embedding
        
        Args:
            load_path: 文件路径
            format: 文件格式，可自动检测
        
        Returns:
            embedding 字典
        """
        load_path = Path(load_path)
        
        if format is None:
            format = load_path.suffix.lstrip('.')
        
        if format == 'npz':
            data = np.load(load_path)
            return {key: data[key] for key in data.files}
        elif format == 'pt':
            data = torch.load(load_path)
            return {k: v.numpy() for k, v in data.items()}
        elif format in ('h5', 'hdf5'):
            import h5py
            with h5py.File(load_path, 'r') as f:
                return {key: f[key][:] for key in f.keys()}
        else:
            raise ValueError(f"Unknown format: {format}")


class EmbeddingAnalyzer:
    """
    Embedding 分析工具
    提供 UMAP/t-SNE 可视化、聚类等分析功能
    """
    
    def __init__(self, embeddings: Dict[str, np.ndarray]):
        """
        初始化分析器
        
        Args:
            embeddings: EmbeddingExtractor 提取的 embedding 字典
        """
        self.expr_embeddings = embeddings['expression_embeddings']
        self.pos_embeddings = embeddings['position_embeddings']
        self.sample_indices = embeddings.get('sample_indices', None)
        
    def reduce_dimensions(
        self,
        embedding_type: str = 'expression',
        method: str = 'umap',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        降维可视化
        
        Args:
            embedding_type: 'expression' 或 'position'
            method: 降维方法 ('umap', 'tsne', 'pca')
            n_components: 目标维度
            **kwargs: 传递给降维方法的额外参数
        
        Returns:
            降维后的坐标 (N, n_components)
        """
        if embedding_type == 'expression':
            data = self.expr_embeddings
        elif embedding_type == 'position':
            data = self.pos_embeddings
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        if method == 'umap':
            try:
                import umap
            except ImportError:
                raise ImportError("umap-learn is required. Install with: pip install umap-learn")
            
            reducer = umap.UMAP(n_components=n_components, **kwargs)
            return reducer.fit_transform(data)
        
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, **kwargs)
            return reducer.fit_transform(data)
        
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, **kwargs)
            return reducer.fit_transform(data)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap', 'tsne', or 'pca'")
    
    def cluster(
        self,
        embedding_type: str = 'expression',
        method: str = 'leiden',
        resolution: float = 1.0,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        聚类分析
        
        Args:
            embedding_type: 'expression' 或 'position'
            method: 聚类方法 ('leiden', 'louvain', 'kmeans')
            resolution: 分辨率参数（leiden/louvain）
            n_clusters: 聚类数量（kmeans）
            **kwargs: 额外参数
        
        Returns:
            聚类标签 (N,)
        """
        if embedding_type == 'expression':
            data = self.expr_embeddings
        elif embedding_type == 'position':
            data = self.pos_embeddings
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        if method in ('leiden', 'louvain'):
            try:
                import scanpy as sc
                import anndata
            except ImportError:
                raise ImportError("scanpy is required. Install with: pip install scanpy")
            
            # 创建 AnnData 对象
            adata = anndata.AnnData(data)
            
            # 计算邻居图
            sc.pp.neighbors(adata, **kwargs)
            
            # 聚类
            if method == 'leiden':
                sc.tl.leiden(adata, resolution=resolution)
                return adata.obs['leiden'].values.astype(int)
            else:
                sc.tl.louvain(adata, resolution=resolution)
                return adata.obs['louvain'].values.astype(int)
        
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            if n_clusters is None:
                n_clusters = 10  # 默认值
            kmeans = KMeans(n_clusters=n_clusters, **kwargs)
            return kmeans.fit_predict(data)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_similarity(
        self,
        embedding_type: str = 'expression',
        metric: str = 'cosine',
    ) -> np.ndarray:
        """
        计算 embedding 之间的相似度矩阵
        
        Args:
            embedding_type: 'expression' 或 'position'
            metric: 相似度度量 ('cosine', 'euclidean', 'correlation')
        
        Returns:
            相似度矩阵 (N, N)
        """
        from sklearn.metrics import pairwise_distances
        
        if embedding_type == 'expression':
            data = self.expr_embeddings
        elif embedding_type == 'position':
            data = self.pos_embeddings
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(data)
        else:
            distances = pairwise_distances(data, metric=metric)
            # 转换为相似度
            return 1 / (1 + distances)


def extract_embeddings_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config_path: Union[str, Path],
    expression: torch.Tensor,
    positions: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    mode: str = 'transformer',
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """
    从保存的 checkpoint 加载模型并提取 embedding 的便捷函数
    
    Args:
        checkpoint_path: 模型 checkpoint 路径
        config_path: 配置文件路径
        expression: (B, N, G) 表达量
        positions: (B, N, 3) 位置
        node_mask: (B, N) 掩码
        mode: 提取模式
        device: 设备
    
    Returns:
        embedding 字典
    
    Example:
        ```python
        result = extract_embeddings_from_checkpoint(
            checkpoint_path='checkpoints/model.ckpt',
            config_path='config.yaml',
            expression=expr_tensor,
            positions=pos_tensor,
        )
        ```
    """
    import yaml
    from alphastomics.attn_model.model import Model
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 构建模型
    model = Model(
        input_dims=config['model']['input_dims'],
        mlp_in_expression_setting=config['model']['mlp_in_expression_setting'],
        mlp_in_diffusion_time_setting=config['model']['mlp_in_diffusion_time_setting'],
        PositionMLP_setting=config['model']['PositionMLP_setting'],
        TransformerLayer_setting=config['model']['TransformerLayer_setting'],
        mlp_out_expression_setting=config['model']['mlp_out_expression_setting'],
        mlp_out_position_norm_setting=config['model']['mlp_out_position_norm_setting'],
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        # PyTorch Lightning checkpoint
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    # 提取 embedding
    extractor = EmbeddingExtractor(model, device=torch.device(device))
    return extractor.extract(
        expression=expression,
        positions=positions,
        node_mask=node_mask,
        mode=mode,
    )


# 命令行接口
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract embeddings from AlphaSTomics model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data', type=str, required=True, help='Input data path (.pt or .npz)')
    parser.add_argument('--output', type=str, required=True, help='Output path for embeddings')
    parser.add_argument('--mode', type=str, default='transformer', 
                        choices=['encoder', 'transformer', 'all_layers'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--format', type=str, default='npz', choices=['npz', 'pt', 'h5'])
    
    args = parser.parse_args()
    
    # 加载数据
    if args.data.endswith('.pt'):
        data = torch.load(args.data)
    else:
        data = np.load(args.data)
        data = {k: torch.from_numpy(v) for k, v in data.items()}
    
    # 提取 embedding
    result = extract_embeddings_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        expression=data['expression'],
        positions=data['positions'],
        node_mask=data.get('node_mask'),
        mode=args.mode,
        device=args.device,
    )
    
    # 转换为 numpy 并保存
    embeddings = {
        'expression_embeddings': result['expression_embedding'].cpu().numpy(),
        'position_embeddings': result['position_embedding'].cpu().numpy(),
    }
    
    extractor = EmbeddingExtractor.__new__(EmbeddingExtractor)
    extractor.save_embeddings(embeddings, args.output, format=args.format)
