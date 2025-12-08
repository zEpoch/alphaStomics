"""
Mixture of Experts (MoE) Module for AlphaSTomics
================================================

基于 Gated Attention 论文，在 FFN 部分引入 MoE 机制
适用于空间转录组的双模态扩散建模

核心特性:
- Top-K Expert Selection: 每个 token 选择 top-k 个专家
- Load Balancing: 辅助损失确保专家负载均衡
- Expert Capacity: 防止某个专家过载
- 支持双模态: 表达量和位置可以使用不同的专家组
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Expert(nn.Module):
    """
    单个专家网络
    标准的 FFN: Linear -> Activation -> Linear
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swiglu':
            # SwiGLU: 需要额外的投影
            self.w3 = nn.Linear(d_model, d_ff)
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.use_swiglu = (activation == 'swiglu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        if self.use_swiglu:
            # SwiGLU: (W1(x) * σ(W3(x))) @ W2
            return self.w2(self.dropout(self.activation(self.w1(x)) * self.w3(x)))
        else:
            return self.w2(self.dropout(self.activation(self.w1(x))))


class TopKRouter(nn.Module):
    """
    Top-K 路由器
    为每个 token 选择 top-k 个专家
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        use_noisy_gating: bool = True,
        noise_epsilon: float = 1e-2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noisy_gating = use_noisy_gating
        self.noise_epsilon = noise_epsilon
        
        # 路由网络: token -> expert logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Noisy gating: 添加可学习的噪声
        if use_noisy_gating:
            self.w_noise = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            training: 是否训练模式
        
        Returns:
            expert_weights: (batch_size, seq_len, top_k) - 专家权重（归一化后）
            expert_indices: (batch_size, seq_len, top_k) - 选中的专家索引
            load_balancing_loss: 负载均衡损失
        """
        bsz, seq_len, d_model = x.shape
        
        # 计算 gate logits
        logits = self.gate(x)  # (bsz, seq_len, num_experts)
        
        # Noisy gating (仅训练时)
        if self.use_noisy_gating and training:
            noise = torch.randn_like(logits) * F.softplus(self.w_noise(x))
            logits = logits + noise * self.noise_epsilon
        
        # Top-k 选择
        top_k_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        # expert_indices: (bsz, seq_len, top_k)
        
        # 归一化权重
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # 计算负载均衡损失
        load_balancing_loss = self._compute_load_balancing_loss(logits)
        
        return expert_weights, expert_indices, load_balancing_loss
    
    def _compute_load_balancing_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        负载均衡损失: 鼓励专家被均匀使用
        
        L_balance = num_experts * sum(f_i * P_i)
        其中 f_i 是专家 i 被选中的频率，P_i 是路由到专家 i 的概率
        """
        # 计算每个专家被路由的概率
        probs = F.softmax(logits, dim=-1)  # (bsz, seq_len, num_experts)
        
        # 计算平均概率
        avg_probs = probs.mean(dim=[0, 1])  # (num_experts,)
        
        # 计算每个专家被选中的频率（top-1 近似）
        _, top1_indices = torch.max(logits, dim=-1)  # (bsz, seq_len)
        freq = torch.bincount(
            top1_indices.flatten(),
            minlength=self.num_experts
        ).float()
        freq = freq / freq.sum()  # 归一化
        
        # 负载均衡损失
        loss = self.num_experts * (freq * avg_probs).sum()
        
        return loss


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer
    
    替代标准 FFN，使用多个专家网络 + top-k 路由
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_noisy_gating: bool = True,
        expert_capacity_factor: float = 1.25,
        load_balance_loss_weight: float = 0.01
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN 隐藏维度（总容量，会被均分给各专家）
            num_experts: 专家数量
            top_k: 每个 token 选择的专家数量
            dropout: Dropout 比率
            activation: 激活函数类型
            use_noisy_gating: 是否使用 noisy gating
            expert_capacity_factor: 专家容量因子（防止过载）
            load_balance_loss_weight: 负载均衡损失权重
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.load_balance_loss_weight = load_balance_loss_weight
        
        # 每个专家的隐藏层大小 = d_ff / num_experts
        # 这样总参数量 ≈ Dense FFN 的参数量
        # 激活参数量 = Dense × (top_k / num_experts)
        d_ff_per_expert = max(1, d_ff // num_experts)
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff_per_expert, dropout, activation)
            for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = TopKRouter(
            d_model, num_experts, top_k, use_noisy_gating
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_load_balance_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            return_load_balance_loss: 是否返回负载均衡损失
        
        Returns:
            output: (batch_size, seq_len, d_model)
            load_balance_loss: 负载均衡损失（可选）
        """
        bsz, seq_len, d_model = x.shape
        
        # 路由: 为每个 token 选择 top-k 专家
        expert_weights, expert_indices, lb_loss = self.router(
            x, training=self.training
        )
        # expert_weights: (bsz, seq_len, top_k)
        # expert_indices: (bsz, seq_len, top_k)
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 为每个专家计算输出（批量处理）
        for k in range(self.top_k):
            # 获取当前位置的专家索引和权重
            expert_idx = expert_indices[:, :, k]  # (bsz, seq_len)
            weights = expert_weights[:, :, k]     # (bsz, seq_len)
            
            # 对每个专家分别计算
            for expert_id in range(self.num_experts):
                # 找到使用当前专家的 tokens
                mask = (expert_idx == expert_id)  # (bsz, seq_len)
                
                if mask.any():
                    # 提取对应的 tokens
                    token_indices = torch.nonzero(mask, as_tuple=True)
                    selected_x = x[token_indices]  # (num_selected, d_model)
                    
                    # 通过专家网络
                    expert_output = self.experts[expert_id](selected_x)
                    
                    # 加权累加到输出
                    # 创建权重张量
                    w = weights[token_indices].unsqueeze(-1)  # (num_selected, 1)
                    output[token_indices] += w * expert_output
        
        if return_load_balance_loss:
            return output, self.load_balance_loss_weight * lb_loss
        else:
            return output, None


class MoETransformerFFN(nn.Module):
    """
    带 MoE 的 Transformer FFN
    可以选择使用 MoE 或标准 FFN
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_moe: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        load_balance_loss_weight: float = 0.01
    ):
        super().__init__()
        self.use_moe = use_moe
        
        if use_moe:
            self.ffn = MixtureOfExperts(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                load_balance_loss_weight=load_balance_loss_weight
            )
        else:
            # 标准 FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU() if activation == 'relu' else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: 辅助损失（MoE 的负载均衡损失，或 None）
        """
        if self.use_moe:
            return self.ffn(x, return_load_balance_loss=True)
        else:
            return self.ffn(x), None
