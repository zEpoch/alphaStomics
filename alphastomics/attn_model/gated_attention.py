"""
Gated Attention Module for Spatial Transcriptomics
基于 Qwen3 的 Gated Attention 机制，针对空间转录组双模态数据优化

核心改进：
1. 使用标准 Softmax Attention 替代 Linear Attention（精确建模空间关系）
2. 支持 headwise/elementwise gating（动态调制信息流）
3. 针对扩散模型优化（时间步自适应）
"""
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GatedMultiHeadAttention(nn.Module):
    """
    多头门控注意力
    
    特点：
    - Softmax Attention (O(n²) 但精确)
    - Query-dependent gating (headwise 或 elementwise)
    - 支持双模态特征（表达量 + 位置 + 时间）
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        gate_type: str = 'headwise',  # 'headwise', 'elementwise', 'none'
        use_qk_norm: bool = True,     # 稳定训练
        qkv_bias: bool = True
    ):
        """
        初始化门控多头注意力
        
        Args:
            d_model: 模型维度（必须能被 num_heads 整除）
            num_heads: 注意力头数
            dropout: Dropout 比率
            gate_type: 门控类型
                - 'headwise': 每个头一个标量 gate (推荐，参数少)
                - 'elementwise': 每个元素一个 gate (更灵活)
                - 'none': 不使用 gating (baseline)
            use_qk_norm: 是否对 Q/K 进行 RMSNorm（提升稳定性）
            qkv_bias: QKV 投影是否使用 bias
        """
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.gate_type = gate_type
        self.use_qk_norm = use_qk_norm
        self.dropout_p = dropout
        
        # QKV 投影
        # 根据 gate_type 调整 Q 的输出维度
        if gate_type == 'headwise':
            # Q: [d_model] -> [d_model + num_heads]
            # 额外的 num_heads 维用于 gate scores
            self.q_proj = nn.Linear(d_model, d_model + num_heads, bias=qkv_bias)
        elif gate_type == 'elementwise':
            # Q: [d_model] -> [2 * d_model]
            # 前 d_model 是 query，后 d_model 是 gate
            self.q_proj = nn.Linear(d_model, 2 * d_model, bias=qkv_bias)
        else:  # 'none'
            self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        
        # QK Normalization（参考 Qwen3）
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: (B, N, d_model) 输入特征
            attention_mask: (B, 1, N, N) 或 (B, N, N) 注意力掩码
                - 0 表示保留，-inf 表示屏蔽
            output_attentions: 是否返回注意力权重
        
        Returns:
            output: (B, N, d_model) 输出特征
            attn_weights: (B, num_heads, N, N) 注意力权重（可选）
        """
        B, N, D = x.shape
        
        # === 1. QKV 投影 ===
        Q = self.q_proj(x)  # (B, N, d_model + extra)
        K = self.k_proj(x)  # (B, N, d_model)
        V = self.v_proj(x)  # (B, N, d_model)
        
        # === 2. 提取 Gate Scores（如果有）===
        if self.gate_type == 'headwise':
            # Q: (B, N, d_model + num_heads)
            # -> query: (B, N, d_model), gate: (B, N, num_heads)
            Q, gate_scores = torch.split(Q, [self.d_model, self.num_heads], dim=-1)
            gate_scores = gate_scores.unsqueeze(-1)  # (B, N, num_heads, 1)
            
        elif self.gate_type == 'elementwise':
            # Q: (B, N, 2 * d_model)
            # -> query: (B, N, d_model), gate: (B, N, d_model)
            Q, gate_scores = torch.split(Q, [self.d_model, self.d_model], dim=-1)
            # gate_scores: (B, N, d_model) -> (B, N, num_heads, head_dim)
            gate_scores = gate_scores.view(B, N, self.num_heads, self.head_dim)
        else:
            gate_scores = None
        
        # === 3. 重塑为多头格式 ===
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D_h)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # === 4. QK Normalization（可选）===
        if self.use_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        # === 5. Scaled Dot-Product Attention ===
        # Attention scores: (B, H, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用 attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, N, N)
            attn_scores = attn_scores + attention_mask
        
        # Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: (B, H, N, N) @ (B, H, N, D_h) -> (B, H, N, D_h)
        attn_output = torch.matmul(attn_weights, V)
        
        # === 6. 应用 Gating ===
        if gate_scores is not None:
            # 转置回 (B, N, H, D_h)
            attn_output = attn_output.transpose(1, 2).contiguous()
            
            # 应用 sigmoid gating
            attn_output = attn_output * torch.sigmoid(gate_scores)
            
            # 重新排列
            attn_output = attn_output.view(B, N, self.d_model)
        else:
            # 标准路径：(B, H, N, D_h) -> (B, N, H, D_h) -> (B, N, D)
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.d_model)
        
        # === 7. 输出投影 ===
        output = self.o_proj(attn_output)
        
        # 返回
        if output_attentions:
            return output, attn_weights
        else:
            return output, None


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    更简单、更高效的归一化方法，用于稳定训练
    参考：https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim) 任意形状，最后一维是 dim
        
        Returns:
            normalized: 与 x 相同形状
        """
        # 保存原始 dtype
        input_dtype = x.dtype
        
        # 转换到 float32 进行计算（数值稳定）
        x = x.float()
        
        # RMS: sqrt(mean(x^2))
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # 应用可学习的 scale
        return (self.weight * x).to(input_dtype)


class GatedSelfAttentionModel(nn.Module):
    """
    门控自注意力模型（替代原 SelfAttentionModel）
    
    专门针对空间转录组的双模态扩散建模：
    - 表达量 (B, N, expr_dim)
    - 位置 (B, N, 3)
    - 扩散时间 (B, time_dim)
    
    与原版的主要区别：
    1. 使用 GatedMultiHeadAttention 替代 LinearAttentionTransformer
    2. 保留相同的接口（无缝替换）
    """
    
    def __init__(
        self,
        expression_features_dim: int,
        diffusion_features_dim: int,
        position_features_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        gate_type: str = 'headwise',
        use_qk_norm: bool = True,
        last_layer: bool = False
    ):
        """
        初始化门控自注意力模型
        
        Args:
            expression_features_dim: 表达量特征维度
            diffusion_features_dim: 扩散时间特征维度
            position_features_dim: 位置特征维度
            num_heads: 注意力头数
            dropout: Dropout 比率
            gate_type: 'headwise' / 'elementwise' / 'none'
            use_qk_norm: 是否使用 QK 归一化
            last_layer: 是否为最后一层
        """
        super().__init__()
        self.expression_features_dim = expression_features_dim
        self.diffusion_features_dim = diffusion_features_dim
        self.position_features_dim = position_features_dim
        self.last_layer = last_layer

        # 位置编码器（同时保留方向和距离信息）
        self.transform_positions_for_attn_mlp = nn.Sequential(
            nn.Linear(in_features=4, out_features=64),  # 3 (方向) + 1 (范数)
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.position_features_dim)
        )

        # 总特征维度
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
        
        # === 核心改动：使用 GatedMultiHeadAttention ===
        self.attention = GatedMultiHeadAttention(
            d_model=total_features_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            gate_type=gate_type,
            use_qk_norm=use_qk_norm
        )
        
        # 输出投影
        self.head_features_to_position = nn.Linear(
            in_features=total_features_out_dim,
            out_features=3
        )
        self.head_features_to_expression = nn.Linear(
            in_features=total_features_out_dim,
            out_features=expression_features_dim
        )
        
        if not self.last_layer:
            self.y_y = nn.Linear(
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
        
        # 处理时间维度
        if transformed_diffusion_time is not None:
            time_for_concat = transformed_diffusion_time
        else:
            time_for_concat = diffusion_time
        
        # 确保时间维度正确扩展到 (B, N, dim)
        if len(time_for_concat.shape) == 2:
            time_expanded = time_for_concat.unsqueeze(1).expand(-1, n, -1)
        elif len(time_for_concat.shape) == 3:
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
        
        # === 门控注意力计算 ===
        head_outputs, _ = self.attention(concatenated_features)
        
        # 输出投影
        output_positions = self.head_features_to_position(head_outputs)
        output_expression = self.head_features_to_expression(head_outputs)
        
        return output_expression, output_positions, transformed_diffusion_time
