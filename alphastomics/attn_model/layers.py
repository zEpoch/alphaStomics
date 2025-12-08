"""
AlphaSTomics 基础层
包含 MLP、PositionMLP、PositionNorm 等基础组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Optional


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim_1: int, 
        hidden_dim_2: int,
        out_dim: int, 
        dropout: float = 0.0
    ):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(in_dim, hidden_dim_1)
        self.hidden_layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.output_layer = nn.Linear(hidden_dim_2, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.hidden_layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x


class PositionMLP(nn.Module):
    """
    位置 MLP
    
    通过学习范数变换来调整位置向量
    保持方向不变，只改变长度
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super(PositionMLP, self).__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 3) 位置坐标
        
        Returns:
            new_positions: (B, N, 3) 范数调整后的位置
        """
        norm = torch.norm(positions, dim=-1, keepdim=True)  # (B, N, 1)
        new_norm = self.mlp(norm)  # (B, N, 1)
        new_positions = positions * (new_norm / (norm + self.eps))
        return new_positions


class PositionNorm(nn.Module):
    """
    位置归一化
    
    根据所有节点的平均范数进行归一化
    """
    
    def __init__(
        self, 
        eps: float = 1e-6, 
        device: Optional[torch.device] = None,
        **kwargs  # 忽略额外的关键字参数
    ):
        super(PositionNorm, self).__init__()
        self.normalized_shape = (1,)
        self.eps = eps
        kw = {"device": device} if device is not None else {}
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **kw))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.ones_(self.weight)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 3) 位置坐标
        
        Returns:
            normalized_positions: (B, N, 3) 归一化后的位置
        """
        norm = torch.norm(positions, dim=-1, keepdim=True)  # (B, N, 1)
        mean_norm = torch.mean(norm, dim=1, keepdim=True)   # (B, 1, 1)
        # 确保 eps 是浮点数
        eps_val = float(self.eps) if isinstance(self.eps, str) else self.eps
        new_positions = self.weight * positions / (mean_norm + eps_val)
        return new_positions


class ResidualMLP(nn.Module):
    """带残差连接的 MLP"""
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0
    ):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # 如果输入输出维度不同，需要投影
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual