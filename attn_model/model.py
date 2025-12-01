"""
AlphaSTomics 主模型
支持双模态扩散：表达量 + 3D坐标
"""
import torch
import torch.nn as nn
from typing import Optional
from alphastomics.attn_model.layers import PositionMLP
from alphastomics.attn_model.transformer import TransformerLayer


class Model(nn.Module):
    """
    AlphaSTomics 双模态扩散模型
    
    输入:
        - expression_features: (B, N, G) 加噪后的表达量
        - diffusion_time: (B, 1) 扩散时间
        - position_features: (B, N, 3) 加噪后的3D位置
        - node_mask: (B, N) 有效节点掩码
    
    输出:
        - pred_expression: (B, N, G) 预测的原始表达量
        - pred_positions: (B, N, 3) 预测的原始位置
    """
    
    def __init__(
        self,
        input_dims: int,
        mlp_in_expression_setting: dict,
        mlp_in_diffusion_time_setting: dict,
        PositionMLP_setting: dict,
        TransformerLayer_setting: dict,
        mlp_out_expression_setting: dict,
        mlp_out_position_norm_setting: dict,
        positionMLP_eps: float = 1e-6,
    ):
        """
        初始化模型
        
        Args:
            input_dims: 输入表达量维度（基因数量）
            mlp_in_expression_setting: 表达量输入MLP配置
            mlp_in_diffusion_time_setting: 时间输入MLP配置
            PositionMLP_setting: 位置MLP配置
            TransformerLayer_setting: Transformer层配置
            mlp_out_expression_setting: 表达量输出MLP配置
            mlp_out_position_norm_setting: 位置范数输出MLP配置
            positionMLP_eps: 数值稳定性参数
        """
        super().__init__()
        self.input_expression_dims = self.output_expression_dims = input_dims
        self.input_diffusion_dims = 1
        self.positionMLP_eps = positionMLP_eps
        self.hidden_expression_dims = mlp_in_expression_setting['mlp_out_expression_dims']

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        # 输入表达量编码器
        self.mlp_in_expression = nn.Sequential(
            nn.Linear(
                in_features=self.input_expression_dims,
                out_features=mlp_in_expression_setting['mlp_in_expression_dims']
            ),
            act_fn_in,
            nn.Linear(
                in_features=mlp_in_expression_setting['mlp_in_expression_dims'],
                out_features=mlp_in_expression_setting['mlp_out_expression_dims']
            ),
            act_fn_in
        )

        # 输入时间编码器
        self.mlp_in_diffusion_time = nn.Sequential(
            nn.Linear(
                in_features=self.input_diffusion_dims,
                out_features=mlp_in_diffusion_time_setting['mlp_in_diffusion_time_dims']
            ),
            act_fn_in,
            nn.Linear(
                in_features=mlp_in_diffusion_time_setting['mlp_in_diffusion_time_dims'],
                out_features=mlp_in_diffusion_time_setting['mlp_out_diffusion_time_dims']
            ),
            act_fn_in
        )
        
        # 输入位置编码器
        self.position_mlp = PositionMLP(
            hidden_dim=PositionMLP_setting['hidden_dims'],
            eps=self.positionMLP_eps
        )
        
        # Transformer 层
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                expression_dim=mlp_in_expression_setting['mlp_out_expression_dims'],
                position_dim=3,
                diffusion_time_dim=mlp_in_diffusion_time_setting['mlp_out_diffusion_time_dims'],
                num_heads=TransformerLayer_setting['num_heads'],
                dim_ff_expression=TransformerLayer_setting['dim_ff_expression'],
                dim_ff_diffusion_time=TransformerLayer_setting['dim_ff_diffusion_time'],
                dropout=TransformerLayer_setting['dropout'],
                layer_norm_eps=TransformerLayer_setting['layer_norm_eps'],
                last_layer=False
            )
            for _ in range(TransformerLayer_setting['num_layers'])
        ])
        
        # 输出表达量解码器
        self.mlp_out_expression = nn.Sequential(
            nn.Linear(
                mlp_in_expression_setting['mlp_out_expression_dims'],
                mlp_out_expression_setting['hidden_dims']
            ),
            act_fn_out,
            nn.Linear(
                mlp_out_expression_setting['hidden_dims'],
                self.output_expression_dims
            )
        )
        
        # 输出位置范数预测器（关键修复：输出1维范数标量，而非3维向量）
        # 输入: 表达量特征 + 位置 + 范数 = hidden_dims + 3 + 1
        self.mlp_out_pos_features = nn.Sequential(
            nn.Linear(
                mlp_in_expression_setting['mlp_out_expression_dims'],
                mlp_out_position_norm_setting['hidden_dims']
            ),
            act_fn_out,
        )
        
        self.mlp_out_position_norm = nn.Sequential(
            nn.Linear(
                mlp_out_position_norm_setting['hidden_dims'] + 3 + 1,  # features + pos + norm
                mlp_out_position_norm_setting['hidden_dims']
            ),
            act_fn_out,
            nn.Linear(
                mlp_out_position_norm_setting['hidden_dims'],
                1  # 输出 1 维范数标量（关键修复）
            )
        )
    
    def forward(
        self,
        expression_features: torch.Tensor,
        diffusion_time: torch.Tensor,
        position_features: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ):
        """
        前向传播
        
        Args:
            expression_features: (B, N, G) 加噪后的表达量
            diffusion_time: (B, 1) 扩散时间
            position_features: (B, N, 3) 加噪后的位置
            node_mask: (B, N) 有效节点掩码
        
        Returns:
            out_expression: (B, N, G) 预测的原始表达量
            out_position: (B, N, 3) 预测的原始位置
        """
        batch_size, num_nodes, _ = expression_features.shape
        
        # 如果没有提供 mask，假设所有节点都有效
        if node_mask is None:
            node_mask = torch.ones(batch_size, num_nodes, device=expression_features.device)
        
        # 输入编码
        x = self.mlp_in_expression(expression_features)  # (B, N, hidden_dim)
        y = self.mlp_in_diffusion_time(diffusion_time)   # (B, 1, hidden_dim) or (B, hidden_dim)
        pos = self.position_mlp(position_features)        # (B, N, 3)

        # Transformer 处理
        for layer in self.transformer_layers:
            x, pos, y = layer(
                expression_features=x,
                diffusion_time=y,
                position_features=pos
            )

        # 输出表达量
        out_expression = self.mlp_out_expression(x)  # (B, N, G)
        
        # 输出位置（使用范数调制，保持方向）
        pos_features = self.mlp_out_pos_features(x)  # (B, N, hidden_dim)
        norm = torch.norm(pos, dim=-1, keepdim=True)  # (B, N, 1)
        
        # 预测新的范数标量
        new_norm = self.mlp_out_position_norm(
            torch.cat([pos_features, pos, norm], dim=-1)
        )  # (B, N, 1)
        
        # 用新范数缩放方向向量
        out_position = pos * new_norm / (norm + self.positionMLP_eps)  # (B, N, 3)
        
        # 应用掩码
        out_expression = out_expression * node_mask.unsqueeze(-1)
        out_position = out_position * node_mask.unsqueeze(-1)
        
        # 中心化位置
        masked_sum = (out_position * node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
        num_valid = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        mean_pos = masked_sum / num_valid
        out_position = (out_position - mean_pos) * node_mask.unsqueeze(-1)
        
        return out_expression, out_position
