"""
DataHolder: 统一的数据容器，用于管理表达量、位置和扩散时间信息
支持双模态扩散（表达量 + 坐标）
"""
import torch
from typing import Optional


def to_device(tensor: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    """将张量移动到指定设备"""
    return tensor.to(device) if tensor is not None else None


def apply_mask(tensor: Optional[torch.Tensor], mask: torch.Tensor, mask_dim: int = -1) -> Optional[torch.Tensor]:
    """对张量应用掩码"""
    return tensor * mask.unsqueeze(mask_dim) if tensor is not None else None


def center_positions(positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    中心化位置坐标（减去均值）
    Args:
        positions: (B, N, 3) 位置张量
        mask: (B, N) 有效节点掩码
    """
    for i in range(positions.shape[0]):
        masked_positions = positions[i][mask[i].bool()]
        if masked_positions.shape[0] > 0:
            mean_pos = masked_positions.mean(dim=0)
            positions[i][mask[i].bool()] = masked_positions - mean_pos
    return positions


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    移除位置的均值（考虑掩码）
    Args:
        x: (B, N, D) 位置或其他需要中心化的张量
        node_mask: (B, N) 有效节点掩码
    """
    node_mask_expanded = node_mask.unsqueeze(-1)  # (B, N, 1)
    masked_x = x * node_mask_expanded
    
    # 计算有效节点数量
    num_nodes = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    num_nodes = torch.clamp(num_nodes, min=1)  # 避免除以0
    
    # 计算均值并移除
    mean = masked_x.sum(dim=1, keepdim=True) / num_nodes
    centered_x = (x - mean) * node_mask_expanded
    
    return centered_x


class DataHolder:
    """
    数据持有类，统一管理扩散模型所需的所有数据
    
    支持双模态扩散：
    - 表达量 (expression): 基因表达矩阵
    - 位置 (positions): 3D 空间坐标
    """
    
    def __init__(
        self,
        expression: torch.Tensor,                    # (B, N, G) 表达量
        positions: torch.Tensor,                      # (B, N, 3) 位置
        node_mask: torch.Tensor,                      # (B, N) 有效节点掩码
        diffusion_time: Optional[torch.Tensor] = None, # (B, 1) 扩散时间
        t_int: Optional[torch.Tensor] = None,         # (B, 1) 整数时间步
        t: Optional[torch.Tensor] = None,             # (B, 1) 归一化时间
        cell_class: Optional[torch.Tensor] = None,    # (B, N) 细胞类型
        cell_ID: Optional[torch.Tensor] = None,       # (B, N) 细胞ID
        # 噪声版本（用于加噪后的数据）
        noisy_expression: Optional[torch.Tensor] = None,
        noisy_positions: Optional[torch.Tensor] = None,
    ) -> None:
        self.expression = expression
        self.positions = positions
        self.node_mask = node_mask
        self.t_int = t_int
        self.t = t
        self.diffusion_time = diffusion_time if diffusion_time is not None else t
        self.cell_class = cell_class
        self.cell_ID = cell_ID
        
        # 噪声版本
        self.noisy_expression = noisy_expression
        self.noisy_positions = noisy_positions

    def device_as(self, tensor: torch.Tensor) -> "DataHolder":
        """将所有张量移动到与给定张量相同的设备"""
        device = tensor.device
        self.expression = to_device(self.expression, device)
        self.positions = to_device(self.positions, device)
        self.node_mask = to_device(self.node_mask, device)
        self.diffusion_time = to_device(self.diffusion_time, device)
        self.t_int = to_device(self.t_int, device)
        self.t = to_device(self.t, device)
        self.cell_class = to_device(self.cell_class, device)
        self.cell_ID = to_device(self.cell_ID, device)
        self.noisy_expression = to_device(self.noisy_expression, device)
        self.noisy_positions = to_device(self.noisy_positions, device)
        return self

    def mask(self, node_mask: Optional[torch.Tensor] = None) -> "DataHolder":
        """
        对数据应用掩码
        Args:
            node_mask: (B, N) 掩码，如果为None则使用self.node_mask
        """
        if node_mask is None:
            assert self.node_mask is not None, "node_mask must be provided"
            node_mask = self.node_mask

        # 应用掩码到各个张量
        self.expression = apply_mask(self.expression, node_mask)
        self.positions = apply_mask(self.positions, node_mask)
        self.noisy_expression = apply_mask(self.noisy_expression, node_mask)
        self.noisy_positions = apply_mask(self.noisy_positions, node_mask)
        
        # 中心化位置
        if self.positions is not None:
            self.positions = center_positions(self.positions, node_mask)
        if self.noisy_positions is not None:
            self.noisy_positions = center_positions(self.noisy_positions, node_mask)
            
        self.cell_class = apply_mask(self.cell_class, node_mask)
        self.cell_ID = apply_mask(self.cell_ID, node_mask)

        return self

    def copy(self) -> "DataHolder":
        """创建数据的深拷贝"""
        return DataHolder(
            expression=self.expression.clone() if self.expression is not None else None,
            positions=self.positions.clone() if self.positions is not None else None,
            node_mask=self.node_mask.clone() if self.node_mask is not None else None,
            diffusion_time=self.diffusion_time.clone() if self.diffusion_time is not None else None,
            t_int=self.t_int.clone() if self.t_int is not None else None,
            t=self.t.clone() if self.t is not None else None,
            cell_class=self.cell_class.clone() if self.cell_class is not None else None,
            cell_ID=self.cell_ID.clone() if self.cell_ID is not None else None,
            noisy_expression=self.noisy_expression.clone() if self.noisy_expression is not None else None,
            noisy_positions=self.noisy_positions.clone() if self.noisy_positions is not None else None,
        )

    @staticmethod
    def get_batch(batches: "DataHolder", index: int, batch_size: int = 1) -> "DataHolder":
        """从批次中提取单个样本"""
        extract = lambda x: x[index].unsqueeze(0) if x is not None else None

        return DataHolder(
            expression=extract(batches.expression),
            positions=extract(batches.positions),
            node_mask=extract(batches.node_mask),
            diffusion_time=extract(batches.diffusion_time) if batches.diffusion_time is not None else None,
            t_int=extract(batches.t_int) if batches.t_int is not None else None,
            t=extract(batches.t) if batches.t is not None else None,
            cell_class=extract(batches.cell_class),
            cell_ID=extract(batches.cell_ID),
            noisy_expression=extract(batches.noisy_expression),
            noisy_positions=extract(batches.noisy_positions),
        )

    @property
    def batch_size(self) -> int:
        """返回批次大小"""
        return self.expression.shape[0] if self.expression is not None else 0

    @property
    def num_nodes(self) -> int:
        """返回节点数量"""
        return self.expression.shape[1] if self.expression is not None else 0

    @property
    def num_genes(self) -> int:
        """返回基因数量"""
        return self.expression.shape[2] if self.expression is not None else 0

    def __repr__(self) -> str:
        def shape_str(x):
            return str(x.shape) if isinstance(x, torch.Tensor) else str(x)
        
        return (
            f"DataHolder(\n"
            f"  expression: {shape_str(self.expression)}\n"
            f"  positions: {shape_str(self.positions)}\n"
            f"  node_mask: {shape_str(self.node_mask)}\n"
            f"  t_int: {self.t_int}\n"
            f"  t: {self.t}\n"
            f"  noisy_expression: {shape_str(self.noisy_expression)}\n"
            f"  noisy_positions: {shape_str(self.noisy_positions)}\n"
            f")"
        )
