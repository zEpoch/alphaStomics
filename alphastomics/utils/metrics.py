"""
评价指标模块
用于评估 AlphaSTomics 模型的性能

主要指标:
1. 表达量重建指标: MSE, PCC, Cosine Similarity
2. 位置重建指标: Distance Matrix MSE, Procrustes, kNN 保持率
3. 细胞类型分类指标: Accuracy, ARI, NMI
4. 空间结构指标: Moran's I, 空间自相关
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)


class ExpressionMetrics:
    """表达量评价指标"""
    
    @staticmethod
    def mse(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        均方误差
        
        Args:
            pred: 预测表达量 (N, G) 或 (B, N, G)
            target: 真实表达量
            mask: 有效掩码
        """
        if mask is not None:
            pred = pred[mask.astype(bool)]
            target = target[mask.astype(bool)]
        return float(mean_squared_error(target.flatten(), pred.flatten()))
    
    @staticmethod
    def mae(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """平均绝对误差"""
        if mask is not None:
            pred = pred[mask.astype(bool)]
            target = target[mask.astype(bool)]
        return float(mean_absolute_error(target.flatten(), pred.flatten()))
    
    @staticmethod
    def pcc_per_gene(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        每个基因的皮尔逊相关系数
        
        Returns:
            (平均 PCC, 每个基因的 PCC 数组)
        """
        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        
        if mask is not None:
            mask = mask.flatten().astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        n_genes = pred.shape[1]
        pccs = []
        
        for g in range(n_genes):
            if np.std(target[:, g]) > 1e-8 and np.std(pred[:, g]) > 1e-8:
                pcc, _ = pearsonr(target[:, g], pred[:, g])
                pccs.append(pcc)
            else:
                pccs.append(0.0)
        
        pccs = np.array(pccs)
        return float(np.nanmean(pccs)), pccs
    
    @staticmethod
    def pcc_per_cell(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """
        每个细胞的皮尔逊相关系数
        
        Returns:
            (平均 PCC, 每个细胞的 PCC 数组)
        """
        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        
        if mask is not None:
            mask = mask.flatten().astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        n_cells = pred.shape[0]
        pccs = []
        
        for c in range(n_cells):
            if np.std(target[c]) > 1e-8 and np.std(pred[c]) > 1e-8:
                pcc, _ = pearsonr(target[c], pred[c])
                pccs.append(pcc)
            else:
                pccs.append(0.0)
        
        pccs = np.array(pccs)
        return float(np.nanmean(pccs)), pccs
    
    @staticmethod
    def cosine_similarity(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """平均余弦相似度"""
        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        
        if mask is not None:
            mask = mask.flatten().astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        # 归一化
        pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
        
        cos_sim = (pred_norm * target_norm).sum(axis=1)
        return float(np.mean(cos_sim))
    
    @staticmethod
    def spearman_per_gene(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """每个基因的 Spearman 相关系数"""
        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        
        if mask is not None:
            mask = mask.flatten().astype(bool)
            pred = pred[mask]
            target = target[mask]
        
        n_genes = pred.shape[1]
        spcs = []
        
        for g in range(n_genes):
            if np.std(target[:, g]) > 1e-8:
                spc, _ = spearmanr(target[:, g], pred[:, g])
                spcs.append(spc)
            else:
                spcs.append(0.0)
        
        spcs = np.array(spcs)
        return float(np.nanmean(spcs)), spcs


class PositionMetrics:
    """位置评价指标"""
    
    @staticmethod
    def distance_matrix_mse(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        距离矩阵 MSE（旋转平移不变）
        
        Args:
            pred: 预测位置 (N, 3) 或 (B, N, 3)
            target: 真实位置
            mask: 有效掩码 (N,) 或 (B, N)
        """
        if pred.ndim == 3:
            # 批处理模式
            total_mse = 0.0
            count = 0
            for b in range(pred.shape[0]):
                if mask is not None:
                    m = mask[b].astype(bool)
                    p = pred[b][m]
                    t = target[b][m]
                else:
                    p = pred[b]
                    t = target[b]
                
                if len(p) > 1:
                    D_pred = cdist(p, p)
                    D_target = cdist(t, t)
                    total_mse += np.mean((D_pred - D_target) ** 2)
                    count += 1
            
            return total_mse / max(count, 1)
        else:
            if mask is not None:
                pred = pred[mask.astype(bool)]
                target = target[mask.astype(bool)]
            
            D_pred = cdist(pred, pred)
            D_target = cdist(target, target)
            return float(np.mean((D_pred - D_target) ** 2))
    
    @staticmethod
    def procrustes_distance(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Procrustes 距离（对齐后的误差）
        考虑旋转、平移和缩放
        """
        if pred.ndim == 3:
            distances = []
            for b in range(pred.shape[0]):
                if mask is not None:
                    m = mask[b].astype(bool)
                    p = pred[b][m]
                    t = target[b][m]
                else:
                    p = pred[b]
                    t = target[b]
                
                if len(p) > 2:
                    _, _, disparity = procrustes(t, p)
                    distances.append(disparity)
            
            return float(np.mean(distances)) if distances else 0.0
        else:
            if mask is not None:
                pred = pred[mask.astype(bool)]
                target = target[mask.astype(bool)]
            
            _, _, disparity = procrustes(target, pred)
            return float(disparity)
    
    @staticmethod
    def knn_preservation(
        pred: np.ndarray,
        target: np.ndarray,
        k: int = 10,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        k 近邻保持率
        衡量局部结构保持程度
        
        Returns:
            k 近邻重叠率 (0-1)
        """
        if pred.ndim == 3:
            preservations = []
            for b in range(pred.shape[0]):
                if mask is not None:
                    m = mask[b].astype(bool)
                    p = pred[b][m]
                    t = target[b][m]
                else:
                    p = pred[b]
                    t = target[b]
                
                if len(p) > k:
                    # 计算 k 近邻
                    k_actual = min(k, len(p) - 1)
                    knn_pred = NearestNeighbors(n_neighbors=k_actual + 1).fit(p)
                    knn_target = NearestNeighbors(n_neighbors=k_actual + 1).fit(t)
                    
                    _, indices_pred = knn_pred.kneighbors(p)
                    _, indices_target = knn_target.kneighbors(t)
                    
                    # 计算重叠率（排除自身）
                    overlaps = []
                    for i in range(len(p)):
                        neighbors_pred = set(indices_pred[i, 1:])
                        neighbors_target = set(indices_target[i, 1:])
                        overlap = len(neighbors_pred & neighbors_target) / k_actual
                        overlaps.append(overlap)
                    
                    preservations.append(np.mean(overlaps))
            
            return float(np.mean(preservations)) if preservations else 0.0
        else:
            if mask is not None:
                pred = pred[mask.astype(bool)]
                target = target[mask.astype(bool)]
            
            k_actual = min(k, len(pred) - 1)
            knn_pred = NearestNeighbors(n_neighbors=k_actual + 1).fit(pred)
            knn_target = NearestNeighbors(n_neighbors=k_actual + 1).fit(target)
            
            _, indices_pred = knn_pred.kneighbors(pred)
            _, indices_target = knn_target.kneighbors(target)
            
            overlaps = []
            for i in range(len(pred)):
                neighbors_pred = set(indices_pred[i, 1:])
                neighbors_target = set(indices_target[i, 1:])
                overlap = len(neighbors_pred & neighbors_target) / k_actual
                overlaps.append(overlap)
            
            return float(np.mean(overlaps))
    
    @staticmethod
    def centroid_distance(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """质心距离"""
        if pred.ndim == 3:
            distances = []
            for b in range(pred.shape[0]):
                if mask is not None:
                    m = mask[b].astype(bool)
                    p = pred[b][m]
                    t = target[b][m]
                else:
                    p = pred[b]
                    t = target[b]
                
                centroid_pred = p.mean(axis=0)
                centroid_target = t.mean(axis=0)
                distances.append(np.linalg.norm(centroid_pred - centroid_target))
            
            return float(np.mean(distances))
        else:
            if mask is not None:
                pred = pred[mask.astype(bool)]
                target = target[mask.astype(bool)]
            
            centroid_pred = pred.mean(axis=0)
            centroid_target = target.mean(axis=0)
            return float(np.linalg.norm(centroid_pred - centroid_target))


class ClusteringMetrics:
    """聚类和分类评价指标"""
    
    @staticmethod
    def ari(
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """调整兰德指数 (Adjusted Rand Index)"""
        if mask is not None:
            pred_labels = pred_labels[mask.astype(bool)]
            true_labels = true_labels[mask.astype(bool)]
        return float(adjusted_rand_score(true_labels, pred_labels))
    
    @staticmethod
    def nmi(
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """归一化互信息 (Normalized Mutual Information)"""
        if mask is not None:
            pred_labels = pred_labels[mask.astype(bool)]
            true_labels = true_labels[mask.astype(bool)]
        return float(normalized_mutual_info_score(true_labels, pred_labels))
    
    @staticmethod
    def accuracy(
        pred_labels: np.ndarray,
        true_labels: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """分类准确率"""
        if mask is not None:
            pred_labels = pred_labels[mask.astype(bool)]
            true_labels = true_labels[mask.astype(bool)]
        return float(accuracy_score(true_labels, pred_labels))
    
    @staticmethod
    def silhouette(
        embeddings: np.ndarray,
        labels: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """轮廓系数"""
        if mask is not None:
            embeddings = embeddings[mask.astype(bool)]
            labels = labels[mask.astype(bool)]
        
        # 需要至少 2 个类别
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0
        
        return float(silhouette_score(embeddings, labels))


class SpatialMetrics:
    """空间结构评价指标"""
    
    @staticmethod
    def morans_i(
        values: np.ndarray,
        positions: np.ndarray,
        mask: Optional[np.ndarray] = None,
        bandwidth: Optional[float] = None
    ) -> float:
        """
        Moran's I 空间自相关指数
        
        Args:
            values: 表达量或其他特征 (N, ) 或 (N, G)
            positions: 空间坐标 (N, D)
            mask: 有效掩码
            bandwidth: 空间权重的带宽
        
        Returns:
            Moran's I 值 (-1 到 1，正值表示正空间自相关)
        """
        if mask is not None:
            values = values[mask.astype(bool)]
            positions = positions[mask.astype(bool)]
        
        n = len(values)
        if n < 3:
            return 0.0
        
        # 如果是多维特征，取平均
        if values.ndim > 1:
            values = values.mean(axis=1)
        
        # 计算空间权重矩阵
        distances = cdist(positions, positions)
        if bandwidth is None:
            bandwidth = np.percentile(distances[distances > 0], 25)
        
        W = np.exp(-distances ** 2 / (2 * bandwidth ** 2))
        np.fill_diagonal(W, 0)
        W = W / (W.sum() + 1e-8)
        
        # 计算 Moran's I
        y = values - values.mean()
        numerator = n * np.sum(W * np.outer(y, y))
        denominator = np.sum(W) * np.sum(y ** 2)
        
        if denominator < 1e-8:
            return 0.0
        
        return float(numerator / denominator)
    
    @staticmethod
    def spatial_coherence(
        embeddings: np.ndarray,
        positions: np.ndarray,
        k: int = 10,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        空间一致性
        衡量空间邻居在 embedding 空间中的相似性
        
        Returns:
            平均邻居相似度 (0-1)
        """
        if mask is not None:
            embeddings = embeddings[mask.astype(bool)]
            positions = positions[mask.astype(bool)]
        
        n = len(embeddings)
        k_actual = min(k, n - 1)
        
        # 找到空间 k 近邻
        knn = NearestNeighbors(n_neighbors=k_actual + 1).fit(positions)
        _, indices = knn.kneighbors(positions)
        
        # 计算 embedding 余弦相似度
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = []
        for i in range(n):
            neighbors = indices[i, 1:]  # 排除自身
            neighbor_emb = emb_norm[neighbors]
            sim = (emb_norm[i:i+1] @ neighbor_emb.T).mean()
            similarities.append(sim)
        
        return float(np.mean(similarities))


class MetricsCalculator:
    """
    综合指标计算器
    统一计算所有评价指标
    """
    
    def __init__(self, k_neighbors: int = 10):
        """
        Args:
            k_neighbors: k 近邻相关指标使用的 k 值
        """
        self.k_neighbors = k_neighbors
        self.expr_metrics = ExpressionMetrics()
        self.pos_metrics = PositionMetrics()
        self.cluster_metrics = ClusteringMetrics()
        self.spatial_metrics = SpatialMetrics()
    
    def compute_all(
        self,
        pred_expression: np.ndarray,
        pred_positions: np.ndarray,
        target_expression: np.ndarray,
        target_positions: np.ndarray,
        mask: Optional[np.ndarray] = None,
        pred_labels: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            pred_expression: 预测表达量
            pred_positions: 预测位置
            target_expression: 真实表达量
            target_positions: 真实位置
            mask: 有效掩码
            pred_labels: 预测的聚类标签（可选）
            true_labels: 真实的细胞类型标签（可选）
            embeddings: embedding 向量（可选，用于空间一致性）
        
        Returns:
            指标字典
        """
        results = {}
        
        # 表达量指标
        results['expr_mse'] = self.expr_metrics.mse(pred_expression, target_expression, mask)
        results['expr_mae'] = self.expr_metrics.mae(pred_expression, target_expression, mask)
        results['expr_pcc_gene'], _ = self.expr_metrics.pcc_per_gene(pred_expression, target_expression, mask)
        results['expr_pcc_cell'], _ = self.expr_metrics.pcc_per_cell(pred_expression, target_expression, mask)
        results['expr_cosine'] = self.expr_metrics.cosine_similarity(pred_expression, target_expression, mask)
        results['expr_spearman'], _ = self.expr_metrics.spearman_per_gene(pred_expression, target_expression, mask)
        
        # 位置指标
        results['pos_dist_mse'] = self.pos_metrics.distance_matrix_mse(pred_positions, target_positions, mask)
        results['pos_procrustes'] = self.pos_metrics.procrustes_distance(pred_positions, target_positions, mask)
        results['pos_knn_preservation'] = self.pos_metrics.knn_preservation(
            pred_positions, target_positions, self.k_neighbors, mask
        )
        
        # 聚类指标（如果提供了标签）
        if pred_labels is not None and true_labels is not None:
            flat_pred = pred_labels.flatten() if pred_labels.ndim > 1 else pred_labels
            flat_true = true_labels.flatten() if true_labels.ndim > 1 else true_labels
            flat_mask = mask.flatten() if mask is not None else None
            
            results['cluster_ari'] = self.cluster_metrics.ari(flat_pred, flat_true, flat_mask)
            results['cluster_nmi'] = self.cluster_metrics.nmi(flat_pred, flat_true, flat_mask)
            results['cluster_accuracy'] = self.cluster_metrics.accuracy(flat_pred, flat_true, flat_mask)
        
        # 空间指标（如果提供了 embedding）
        if embeddings is not None:
            flat_emb = embeddings.reshape(-1, embeddings.shape[-1]) if embeddings.ndim > 2 else embeddings
            flat_pos = pred_positions.reshape(-1, 3) if pred_positions.ndim > 2 else pred_positions
            flat_mask = mask.flatten() if mask is not None else None
            
            results['spatial_coherence'] = self.spatial_metrics.spatial_coherence(
                flat_emb, flat_pos, self.k_neighbors, flat_mask
            )
        
        return results
    
    def compute_expression_only(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """仅计算表达量指标"""
        results = {}
        results['mse'] = self.expr_metrics.mse(pred, target, mask)
        results['mae'] = self.expr_metrics.mae(pred, target, mask)
        results['pcc_gene'], _ = self.expr_metrics.pcc_per_gene(pred, target, mask)
        results['pcc_cell'], _ = self.expr_metrics.pcc_per_cell(pred, target, mask)
        results['cosine'] = self.expr_metrics.cosine_similarity(pred, target, mask)
        return results
    
    def compute_position_only(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """仅计算位置指标"""
        results = {}
        results['dist_mse'] = self.pos_metrics.distance_matrix_mse(pred, target, mask)
        results['procrustes'] = self.pos_metrics.procrustes_distance(pred, target, mask)
        results['knn_preservation'] = self.pos_metrics.knn_preservation(pred, target, self.k_neighbors, mask)
        return results


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sampler,  # DiffusionSampler
    device: torch.device,
    mode: str = 'joint',
    num_steps: int = 100,
) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: AlphaSTomics 模型
        dataloader: 测试数据加载器
        sampler: DiffusionSampler 实例
        device: 计算设备
        mode: 采样模式 ('expr_to_pos', 'pos_to_expr', 'joint')
        num_steps: 采样步数
    
    Returns:
        所有指标的字典
    """
    model.eval()
    calculator = MetricsCalculator()
    
    all_pred_expr = []
    all_pred_pos = []
    all_target_expr = []
    all_target_pos = []
    all_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            expression = batch['expression'].to(device)
            positions = batch['positions'].to(device)
            node_mask = batch['node_mask'].to(device)
            
            # 根据模式采样
            if mode == 'expr_to_pos':
                # 从表达量预测位置
                _, pred_pos = sampler.sample(
                    expression=expression,
                    positions=None,  # 需要预测
                    node_mask=node_mask,
                    num_steps=num_steps,
                    mode=mode,
                )
                pred_expr = expression  # 表达量不变
            elif mode == 'pos_to_expr':
                # 从位置预测表达量
                pred_expr, _ = sampler.sample(
                    expression=None,  # 需要预测
                    positions=positions,
                    node_mask=node_mask,
                    num_steps=num_steps,
                    mode=mode,
                )
                pred_pos = positions  # 位置不变
            else:  # joint
                pred_expr, pred_pos = sampler.sample(
                    expression=None,
                    positions=None,
                    node_mask=node_mask,
                    num_steps=num_steps,
                    mode=mode,
                )
            
            all_pred_expr.append(pred_expr.cpu().numpy())
            all_pred_pos.append(pred_pos.cpu().numpy())
            all_target_expr.append(expression.cpu().numpy())
            all_target_pos.append(positions.cpu().numpy())
            all_masks.append(node_mask.cpu().numpy())
    
    # 合并所有批次
    pred_expr = np.concatenate(all_pred_expr, axis=0)
    pred_pos = np.concatenate(all_pred_pos, axis=0)
    target_expr = np.concatenate(all_target_expr, axis=0)
    target_pos = np.concatenate(all_target_pos, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    # 计算指标
    if mode == 'expr_to_pos':
        results = calculator.compute_position_only(pred_pos, target_pos, masks)
    elif mode == 'pos_to_expr':
        results = calculator.compute_expression_only(pred_expr, target_expr, masks)
    else:
        results = calculator.compute_all(
            pred_expr, pred_pos, target_expr, target_pos, masks
        )
    
    return results


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """格式化打印指标"""
    print(f"\n{'=' * 50}")
    print(f"{prefix} Evaluation Results")
    print(f"{'=' * 50}")
    
    # 分组显示
    expr_metrics = {k: v for k, v in metrics.items() if k.startswith('expr_')}
    pos_metrics = {k: v for k, v in metrics.items() if k.startswith('pos_')}
    cluster_metrics = {k: v for k, v in metrics.items() if k.startswith('cluster_')}
    other_metrics = {k: v for k, v in metrics.items() 
                     if not any(k.startswith(p) for p in ['expr_', 'pos_', 'cluster_'])}
    
    if expr_metrics:
        print("\n📊 Expression Metrics:")
        for k, v in expr_metrics.items():
            print(f"  {k}: {v:.6f}")
    
    if pos_metrics:
        print("\n📍 Position Metrics:")
        for k, v in pos_metrics.items():
            print(f"  {k}: {v:.6f}")
    
    if cluster_metrics:
        print("\n🔮 Clustering Metrics:")
        for k, v in cluster_metrics.items():
            print(f"  {k}: {v:.6f}")
    
    if other_metrics:
        print("\n📈 Other Metrics:")
        for k, v in other_metrics.items():
            print(f"  {k}: {v:.6f}")
    
    print(f"\n{'=' * 50}\n")
