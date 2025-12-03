"""
SelfAttentionModel: 自注意力模型
使用 Linear Attention 实现高效的大规模注意力计算
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from linear_attention_transformer import LinearAttentionTransformer


class SelfAttentionModel(nn.Module):
    """
    自注意力模型
    
    将表达量、位置、时间特征拼接后进行注意力计算
    使用 Linear Attention 以支持长序列
    """
    
    def __init__(
        self,
        expression_features_dim: int,
        diffusion_features_dim: int,
        position_features_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        last_layer: bool = False
    ):
        """
        初始化自注意力模型
        
        Args:
            expression_features_dim: 表达量特征维度
            diffusion_features_dim: 扩散时间特征维度
            position_features_dim: 位置特征维度
            num_heads: 注意力头数
            dropout: Dropout 比率
            last_layer: 是否为最后一层
        """
        super(SelfAttentionModel, self).__init__()
        self.expression_features_dim = expression_features_dim
        self.diffusion_features_dim = diffusion_features_dim
        self.position_features_dim = position_features_dim

        # 位置编码器（同时保留方向和距离信息）
        self.transform_positions_for_attn_mlp = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),  # 3 (方向) + 1 (范数)
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.position_features_dim)
        )

        total_features_in_dim = (
            self.expression_features_dim +
            self.diffusion_features_dim +
            self.position_features_dim
        )
        total_features_out_dim = self.expression_features_dim

        # 特征融合层
        self.concatenated_features = nn.Linear(
            in_features=total_features_in_dim,
            out_features=total_features_out_dim
        )
        
        assert total_features_out_dim % num_heads == 0, \
            f"Embedding dimension ({total_features_out_dim}) must be divisible by number of heads ({num_heads})."

        self.head_dim = total_features_out_dim // num_heads
        
        # 表达量特征变换
        self.transform_expression_for_attn_mlp = nn.Sequential(
            nn.Linear(in_features=expression_features_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.expression_features_dim)
        )
        
        # Linear Attention（高效处理长序列）
        self.attention = LinearAttentionTransformer(
            dim=total_features_out_dim,
            heads=num_heads, 
            depth=1,
            max_seq_len=50000
        )
        
        # 输出投影
        self.head_features_to_position = nn.Linear(
            in_features=num_heads * self.head_dim,
            out_features=3
        )
        self.head_features_to_expression = nn.Linear(
            in_features=num_heads * self.head_dim,
            out_features=expression_features_dim
        )
        
        self.last_layer = last_layer
        
        if not self.last_layer:
            self.y_y = Linear(
                in_features=diffusion_features_dim,
                out_features=diffusion_features_dim
            )
    
    def transform_positions_for_attention(
        self,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        将位置编码为注意力特征（保留方向和距离信息）
        
        Args:
            positions: (B, N, 3) 位置坐标
        
        Returns:
            transformed: (B, N, position_features_dim) 位置特征
        """
        norm = torch.norm(positions, dim=-1, keepdim=True)  # (B, N, 1)
        direction = positions / (norm + 1e-7)  # (B, N, 3) 归一化方向
        
        # 同时保留方向和距离信息
        features = torch.cat([direction, norm], dim=-1)  # (B, N, 4)
        transformed_positions = self.transform_positions_for_attn_mlp(features)
        
        return transformed_positions
    
    def transform_expression_features(
        self,
        expression_features: torch.Tensor
    ) -> torch.Tensor:
        """变换表达量特征"""
        return self.transform_expression_for_attn_mlp(expression_features)
    
    def transform_diffusion_time(
        self,
        diffusion_time: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """变换扩散时间特征"""
        if self.last_layer:
            return None
        return self.y_y(diffusion_time)
    
    def forward(
        self,
        expression_features: torch.Tensor,
        diffusion_time: torch.Tensor,
        position_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            expression_features: (B, N, expr_dim) 表达量特征
            diffusion_time: (B, time_dim) 或 (B, 1, time_dim) 扩散时间
            position_features: (B, N, 3) 位置特征
        
        Returns:
            transformed_expression: (B, N, expr_dim) 变换后的表达量
            transformed_positions: (B, N, 3) 变换后的位置
            transformed_diffusion_time: (B, time_dim) 变换后的时间
        """
        bs, n, _ = expression_features.size()
        
        # 特征变换
        transformed_expression = self.transform_expression_features(expression_features)
        transformed_positions = self.transform_positions_for_attention(position_features)
        transformed_diffusion_time = self.transform_diffusion_time(diffusion_time)
        
        # 处理时间维度 - 使用转换后的时间特征（具有正确的维度）
        # 注意：last_layer 时 transformed_diffusion_time 为 None，需要直接使用原始 diffusion_time
        if transformed_diffusion_time is not None:
            time_for_concat = transformed_diffusion_time
        else:
            # last_layer 情况：使用原始 diffusion_time
            time_for_concat = diffusion_time
        
        # 确保时间维度正确扩展到 (B, N, dim)
        if len(time_for_concat.shape) == 2:
            # (B, dim) -> (B, 1, dim) -> (B, N, dim)
            time_expanded = time_for_concat.unsqueeze(1).expand(-1, n, -1)
        elif len(time_for_concat.shape) == 3:
            # (B, 1, dim) -> (B, N, dim)
            time_expanded = time_for_concat.expand(-1, n, -1)
        else:
            raise ValueError(f"Unexpected time_for_concat shape: {time_for_concat.shape}")
        
        # 拼接所有特征
        concatenated_features = torch.cat(
            [transformed_expression, time_expanded, transformed_positions],
            dim=-1
        )
        
        # 特征融合
        concatenated_features = self.concatenated_features(concatenated_features)
        
        # 注意力计算
        head_outputs = self.attention(concatenated_features)
        
        # 输出投影
        output_positions = self.head_features_to_position(head_outputs)
        output_expression = self.head_features_to_expression(head_outputs)
        
        return output_expression, output_positions, transformed_diffusion_time