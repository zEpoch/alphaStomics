"""
TransformerLayer: Transformer 层实现
包含自注意力、前馈网络、残差连接和层归一化
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from alphastomics.attn_model.self_attention import SelfAttentionModel
from alphastomics.attn_model.layers import PositionNorm


class TransformerLayer(nn.Module):
    """
    Transformer 层
    
    包含:
    - 自注意力（表达量 + 位置 + 时间）
    - 前馈网络
    - 残差连接
    - 层归一化
    """
    
    def __init__(
        self,
        expression_dim: int,
        position_dim: int,
        diffusion_time_dim: int,
        num_heads: int,
        dim_ff_expression: int,
        dim_ff_diffusion_time: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        last_layer: bool = False
    ):
        """
        初始化 Transformer 层
        
        Args:
            expression_dim: 表达量特征维度
            position_dim: 位置特征维度
            diffusion_time_dim: 扩散时间维度
            num_heads: 注意力头数
            dim_ff_expression: 表达量前馈网络隐藏维度
            dim_ff_diffusion_time: 时间前馈网络隐藏维度
            dropout: Dropout 比率
            layer_norm_eps: LayerNorm 的 epsilon
            device: 设备
            last_layer: 是否为最后一层
        """
        kw = {"device": device} if device is not None else {}
        super().__init__()
        
        # 自注意力模块
        self.attn_model = SelfAttentionModel(
            expression_features_dim=expression_dim,
            diffusion_features_dim=diffusion_time_dim,
            position_features_dim=position_dim,
            num_heads=num_heads,
            dropout=dropout,
            last_layer=last_layer
        )
        
        # 表达量的前馈网络
        self.lin_expression_features_1 = Linear(
            in_features=expression_dim,
            out_features=dim_ff_expression,
            **kw
        )
        self.lin_expression_features_2 = Linear(
            in_features=dim_ff_expression,
            out_features=expression_dim,
            **kw
        )
        
        # 表达量的层归一化
        self.norm_node_features_1 = LayerNorm(
            normalized_shape=expression_dim,
            eps=layer_norm_eps,
            **kw
        )
        self.norm_node_features_2 = LayerNorm(
            normalized_shape=expression_dim,
            eps=layer_norm_eps,
            **kw
        )
        
        # 表达量的 Dropout
        self.dropout_expression_features_1 = Dropout(p=dropout)
        self.dropout_expression_features_2 = Dropout(p=dropout)
        self.dropout_expression_features_3 = Dropout(p=dropout)
        
        # 位置的归一化
        self.norm_positions_1 = PositionNorm(eps=layer_norm_eps, **kw)
        
        self.last_layer = last_layer
        
        # 时间的前馈网络和归一化（非最后一层）
        if not last_layer:
            self.lin_diffusion_time_1 = Linear(
                in_features=diffusion_time_dim,
                out_features=dim_ff_diffusion_time,
                **kw
            )
            self.lin_diffusion_time_2 = Linear(
                in_features=dim_ff_diffusion_time,
                out_features=diffusion_time_dim,
                **kw
            )
            self.norm_diffusion_time_1 = LayerNorm(
                normalized_shape=diffusion_time_dim,
                eps=layer_norm_eps,
                **kw
            )
            self.norm_diffusion_time_2 = LayerNorm(
                normalized_shape=diffusion_time_dim,
                eps=layer_norm_eps,
                **kw
            )
            self.dropout_diffusion_time_1 = Dropout(p=dropout)
            self.dropout_diffusion_time_2 = Dropout(p=dropout)
            self.dropout_diffusion_time_3 = Dropout(p=dropout)
        
        self.activation = F.gelu
        
    def forward(
        self,
        expression_features: torch.Tensor,
        diffusion_time: torch.Tensor,
        position_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            expression_features: (B, N, expr_dim) 表达量特征
            diffusion_time: (B, time_dim) 扩散时间
            position_features: (B, N, 3) 位置特征
        
        Returns:
            output_expression: (B, N, expr_dim) 输出表达量
            output_positions: (B, N, 3) 输出位置
            output_diffusion_time: (B, time_dim) 输出时间
        """
        # 自注意力
        attn_expression, attn_positions, attn_time = self.attn_model(
            expression_features=expression_features,
            diffusion_time=diffusion_time,
            position_features=position_features
        )

        # 位置归一化
        output_positions = self.norm_positions_1(attn_positions)

        # 表达量：残差连接 + LayerNorm
        output_expression = self.norm_node_features_1(
            expression_features + self.dropout_expression_features_1(attn_expression)
        )
        
        # 表达量：前馈网络
        ff_output_expression = self.lin_expression_features_2(
            self.dropout_expression_features_2(
                self.activation(
                    self.lin_expression_features_1(output_expression)
                )
            )
        )
        ff_output_expression = self.dropout_expression_features_3(ff_output_expression)
        
        # 表达量：残差连接 + LayerNorm
        output_expression = self.norm_node_features_2(
            output_expression + ff_output_expression
        )
        
        # 时间：残差连接 + LayerNorm + 前馈网络
        if not self.last_layer and attn_time is not None:
            output_time = self.norm_diffusion_time_1(
                diffusion_time + self.dropout_diffusion_time_1(attn_time)
            )
            
            ff_output_time = self.lin_diffusion_time_2(
                self.dropout_diffusion_time_2(
                    self.activation(
                        self.lin_diffusion_time_1(output_time)
                    )
                )
            )
            ff_output_time = self.dropout_diffusion_time_3(ff_output_time)
            
            output_time = self.norm_diffusion_time_2(
                output_time + ff_output_time
            )
        else:
            output_time = diffusion_time
            
        return output_expression, output_positions, output_time