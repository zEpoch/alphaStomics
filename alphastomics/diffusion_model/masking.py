"""
Masked Diffusion 模块

对每个细胞的表达量（基因级）和坐标（维度级）进行 masking
增强模型的重建能力和泛化性能

核心组件:
- MaskGenerator: 生成 mask
- MaskToken: 可学习的 mask 占位符
- MaskedNoiseModel: 带 masking 的噪声模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class MaskInfo:
    """存储 mask 信息的数据类"""
    expression_mask: Optional[torch.Tensor] = None  # (B, N, G) bool
    position_mask: Optional[torch.Tensor] = None    # (B, N, 3) bool
    original_expression: Optional[torch.Tensor] = None
    original_position: Optional[torch.Tensor] = None
    
    def has_mask(self) -> bool:
        """检查是否有任何 mask"""
        return (self.expression_mask is not None or 
                self.position_mask is not None)


class MaskGenerator(nn.Module):
    """
    Mask 生成器
    
    支持三种策略:
    - 'random': 完全随机 mask
    - 'block': 连续块 mask（模拟基因模块）
    - 'structured': 结构化 mask（优先 mask 高表达基因）
    """
    
    def __init__(
        self,
        expression_mask_ratio: float = 0.4,
        position_mask_ratio: float = 0.33,
        mask_strategy: Literal['random', 'block', 'structured'] = 'random',
        min_visible_genes: int = 10,
        min_visible_dims: int = 1
    ):
        """
        Args:
            expression_mask_ratio: 表达量 mask 比例 (0.0-1.0)
            position_mask_ratio: 坐标 mask 比例 (0.0-1.0)
            mask_strategy: mask 策略
            min_visible_genes: 最少保留基因数
            min_visible_dims: 最少保留坐标维度数
        """
        super().__init__()
        self.expression_mask_ratio = expression_mask_ratio
        self.position_mask_ratio = position_mask_ratio
        self.mask_strategy = mask_strategy
        self.min_visible_genes = min_visible_genes
        self.min_visible_dims = min_visible_dims
    
    def generate_expression_mask(
        self,
        expression: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        生成表达量 mask
        
        Args:
            expression: (B, N, G) 表达量
            mask_ratio: 可选的 mask 比例（覆盖默认值）
        
        Returns:
            mask: (B, N, G) bool 张量，True 表示被 mask
        """
        B, N, G = expression.shape
        ratio = mask_ratio if mask_ratio is not None else self.expression_mask_ratio
        device = expression.device
        
        num_masked = max(1, int(G * ratio))
        num_masked = min(num_masked, G - self.min_visible_genes)
        
        if self.mask_strategy == 'random':
            # 随机生成每个样本的 mask
            # 优化：批量生成随机数
            rand_matrix = torch.rand(B, N, G, device=device)
            threshold = torch.kthvalue(rand_matrix, G - num_masked, dim=-1, keepdim=True).values
            mask = rand_matrix >= threshold
            
        elif self.mask_strategy == 'block':
            # 连续块 mask
            mask = torch.zeros(B, N, G, dtype=torch.bool, device=device)
            for b in range(B):
                start_idx = torch.randint(0, G - num_masked + 1, (1,), device=device).item()
                mask[b, :, start_idx:start_idx + num_masked] = True
                
        elif self.mask_strategy == 'structured':
            # 结构化 mask：优先 mask 高表达基因
            expr_abs = torch.abs(expression)
            _, sorted_indices = torch.sort(expr_abs, dim=-1, descending=True)
            
            mask = torch.zeros(B, N, G, dtype=torch.bool, device=device)
            for b in range(B):
                for n in range(N):
                    mask[b, n, sorted_indices[b, n, :num_masked]] = True
        else:
            raise ValueError(f"Unknown mask strategy: {self.mask_strategy}")
        
        return mask
    
    def generate_position_mask(
        self,
        position: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        生成坐标 mask
        
        Args:
            position: (B, N, 3) 坐标
            mask_ratio: 可选的 mask 比例
        
        Returns:
            mask: (B, N, 3) bool 张量
        """
        B, N, _ = position.shape
        ratio = mask_ratio if mask_ratio is not None else self.position_mask_ratio
        device = position.device
        
        # 对于 3D 坐标，随机决定 mask 几个维度
        mask = torch.zeros(B, N, 3, dtype=torch.bool, device=device)
        
        # 生成随机数决定每个细胞 mask 几个维度
        rand = torch.rand(B, N, device=device)
        
        # mask 1 个维度的概率
        mask_1_prob = ratio
        # mask 2 个维度的概率（更低）
        mask_2_prob = ratio * 0.3
        
        for b in range(B):
            for n in range(N):
                r = rand[b, n].item()
                if r < mask_2_prob:
                    # mask 2 个维度
                    dims = torch.randperm(3, device=device)[:2]
                    mask[b, n, dims] = True
                elif r < mask_1_prob:
                    # mask 1 个维度
                    dim = torch.randint(0, 3, (1,), device=device).item()
                    mask[b, n, dim] = True
        
        return mask


class MaskToken(nn.Module):
    """
    可学习的 Mask Token
    
    为被 mask 的特征提供占位符
    """
    
    def __init__(
        self,
        expression_dim: int,
        position_dim: int = 3,
        init_std: float = 0.02
    ):
        """
        Args:
            expression_dim: 表达量维度
            position_dim: 坐标维度
            init_std: 初始化标准差
        """
        super().__init__()
        
        self.expression_mask_token = nn.Parameter(
            torch.randn(1, 1, expression_dim) * init_std
        )
        self.position_mask_token = nn.Parameter(
            torch.randn(1, 1, position_dim) * init_std
        )
    
    def apply_expression_mask(
        self,
        expression: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        对表达量应用 mask
        
        Args:
            expression: (B, N, G) 表达量
            mask: (B, N, G) bool mask
        
        Returns:
            masked: (B, N, G) masked 表达量
        """
        B, N, G = expression.shape
        mask_token = self.expression_mask_token.expand(B, N, G)
        return torch.where(mask, mask_token, expression)
    
    def apply_position_mask(
        self,
        position: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        对坐标应用 mask
        
        Args:
            position: (B, N, 3) 坐标
            mask: (B, N, 3) bool mask
        
        Returns:
            masked: (B, N, 3) masked 坐标
        """
        B, N, _ = position.shape
        mask_token = self.position_mask_token.expand(B, N, 3)
        return torch.where(mask, mask_token, position)


class MaskedDiffusionLoss(nn.Module):
    """
    Masked Diffusion 重建损失
    
    只在被 mask 的位置计算重建损失
    """
    
    def __init__(
        self,
        lambda_expression_recon: float = 1.0,
        lambda_position_recon: float = 1.0
    ):
        """
        Args:
            lambda_expression_recon: 表达量重建损失权重
            lambda_position_recon: 坐标重建损失权重
        """
        super().__init__()
        self.lambda_expression_recon = lambda_expression_recon
        self.lambda_position_recon = lambda_position_recon
    
    def forward(
        self,
        pred_expression: torch.Tensor,
        pred_position: torch.Tensor,
        mask_info: MaskInfo,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算重建损失
        
        Args:
            pred_expression: (B, N, G) 预测的表达量
            pred_position: (B, N, 3) 预测的位置
            mask_info: MaskInfo 实例
            node_mask: (B, N) 节点掩码
        
        Returns:
            total_loss: 总重建损失
            log_dict: 损失日志
        """
        device = pred_expression.device
        log_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 表达量重建损失
        if mask_info.expression_mask is not None:
            expr_mask = mask_info.expression_mask
            original = mask_info.original_expression
            
            # 应用节点掩码
            if node_mask is not None:
                effective_mask = expr_mask & node_mask.unsqueeze(-1).bool()
            else:
                effective_mask = expr_mask
            
            if effective_mask.sum() > 0:
                masked_pred = pred_expression[effective_mask]
                masked_target = original[effective_mask]
                expr_recon_loss = F.mse_loss(masked_pred, masked_target)
                total_loss = total_loss + self.lambda_expression_recon * expr_recon_loss
                log_dict['loss/expression_reconstruction'] = expr_recon_loss.item()
        
        # 坐标重建损失
        if mask_info.position_mask is not None:
            pos_mask = mask_info.position_mask
            original = mask_info.original_position
            
            if node_mask is not None:
                effective_mask = pos_mask & node_mask.unsqueeze(-1).bool()
            else:
                effective_mask = pos_mask
            
            if effective_mask.sum() > 0:
                masked_pred = pred_position[effective_mask]
                masked_target = original[effective_mask]
                pos_recon_loss = F.mse_loss(masked_pred, masked_target)
                total_loss = total_loss + self.lambda_position_recon * pos_recon_loss
                log_dict['loss/position_reconstruction'] = pos_recon_loss.item()
        
        log_dict['loss/total_reconstruction'] = total_loss.item()
        return total_loss, log_dict


class MaskingConfig:
    """Masking 配置类"""
    
    def __init__(
        self,
        enable: bool = True,
        expression_mask_ratio: float = 0.4,
        position_mask_ratio: float = 0.33,
        mask_strategy: str = 'random',
        mask_expression: bool = True,
        mask_position: bool = True,
        reconstruction_weight: float = 0.5,
        masking_probability: float = 0.5,
        progressive_masking: bool = False,
        progressive_steps: int = 10000
    ):
        self.enable = enable
        self.expression_mask_ratio = expression_mask_ratio
        self.position_mask_ratio = position_mask_ratio
        self.mask_strategy = mask_strategy
        self.mask_expression = mask_expression
        self.mask_position = mask_position
        self.reconstruction_weight = reconstruction_weight
        self.masking_probability = masking_probability
        self.progressive_masking = progressive_masking
        self.progressive_steps = progressive_steps
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MaskingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})
    
    def to_dict(self) -> dict:
        return {
            'enable': self.enable,
            'expression_mask_ratio': self.expression_mask_ratio,
            'position_mask_ratio': self.position_mask_ratio,
            'mask_strategy': self.mask_strategy,
            'mask_expression': self.mask_expression,
            'mask_position': self.mask_position,
            'reconstruction_weight': self.reconstruction_weight,
            'masking_probability': self.masking_probability,
            'progressive_masking': self.progressive_masking,
            'progressive_steps': self.progressive_steps
        }
    
    def get_current_mask_ratio(self, current_step: int) -> Tuple[float, float]:
        """
        获取当前步骤的 mask ratio（支持渐进式）
        
        Args:
            current_step: 当前训练步骤
        
        Returns:
            (expression_ratio, position_ratio)
        """
        if not self.progressive_masking:
            return self.expression_mask_ratio, self.position_mask_ratio
        
        # 渐进式增加 mask ratio
        progress = min(1.0, current_step / self.progressive_steps)
        expr_ratio = self.expression_mask_ratio * progress
        pos_ratio = self.position_mask_ratio * progress
        return expr_ratio, pos_ratio


class MaskedDiffusionModule(nn.Module):
    """
    Masked Diffusion 完整模块
    
    集成了 MaskGenerator、MaskToken 和 MaskedDiffusionLoss
    可以直接集成到训练流程中
    """
    
    def __init__(
        self,
        expression_dim: int,
        position_dim: int = 3,
        config: Optional[MaskingConfig] = None
    ):
        """
        Args:
            expression_dim: 表达量维度
            position_dim: 坐标维度
            config: MaskingConfig 配置
        """
        super().__init__()
        
        self.config = config or MaskingConfig()
        self.expression_dim = expression_dim
        self.position_dim = position_dim
        
        # 初始化组件
        self.mask_generator = MaskGenerator(
            expression_mask_ratio=self.config.expression_mask_ratio,
            position_mask_ratio=self.config.position_mask_ratio,
            mask_strategy=self.config.mask_strategy
        )
        
        self.mask_token = MaskToken(
            expression_dim=expression_dim,
            position_dim=position_dim
        )
        
        self.recon_loss = MaskedDiffusionLoss(
            lambda_expression_recon=1.0,
            lambda_position_recon=1.0
        )
        
        # 训练步骤计数器
        self._current_step = 0
    
    def set_step(self, step: int):
        """设置当前训练步骤（用于渐进式 masking）"""
        self._current_step = step
    
    def should_apply_masking(self) -> bool:
        """决定是否应用 masking（基于概率）"""
        if not self.config.enable:
            return False
        return torch.rand(1).item() < self.config.masking_probability
    
    def apply_masking(
        self,
        expression: torch.Tensor,
        position: torch.Tensor,
        apply: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, MaskInfo]:
        """
        应用 masking
        
        Args:
            expression: (B, N, G) 表达量（通常是加噪后的）
            position: (B, N, 3) 坐标（通常是加噪后的）
            apply: 是否实际应用 masking
        
        Returns:
            masked_expression: masked 后的表达量
            masked_position: masked 后的坐标
            mask_info: MaskInfo 实例
        """
        if not apply or not self.config.enable:
            return expression, position, MaskInfo()
        
        # 获取当前 mask ratio
        expr_ratio, pos_ratio = self.config.get_current_mask_ratio(self._current_step)
        
        mask_info = MaskInfo(
            original_expression=expression.clone(),
            original_position=position.clone()
        )
        
        masked_expression = expression
        masked_position = position
        
        # 表达量 masking
        if self.config.mask_expression and expr_ratio > 0:
            expr_mask = self.mask_generator.generate_expression_mask(
                expression, mask_ratio=expr_ratio
            )
            masked_expression = self.mask_token.apply_expression_mask(
                expression, expr_mask
            )
            mask_info.expression_mask = expr_mask
        
        # 坐标 masking
        if self.config.mask_position and pos_ratio > 0:
            pos_mask = self.mask_generator.generate_position_mask(
                position, mask_ratio=pos_ratio
            )
            masked_position = self.mask_token.apply_position_mask(
                position, pos_mask
            )
            mask_info.position_mask = pos_mask
        
        return masked_expression, masked_position, mask_info
    
    def compute_reconstruction_loss(
        self,
        pred_expression: torch.Tensor,
        pred_position: torch.Tensor,
        mask_info: MaskInfo,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算重建损失
        
        Args:
            pred_expression: 模型预测的表达量
            pred_position: 模型预测的位置
            mask_info: MaskInfo 实例
            node_mask: 节点掩码
        
        Returns:
            loss: 重建损失
            log_dict: 损失日志
        """
        if not mask_info.has_mask():
            return torch.tensor(0.0, device=pred_expression.device), {}
        
        loss, log_dict = self.recon_loss(
            pred_expression, pred_position, mask_info, node_mask
        )
        
        return loss * self.config.reconstruction_weight, log_dict
