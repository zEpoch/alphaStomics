"""
DualModalLoss: 双模态损失函数
- 表达量: MSE Loss
- 位置: 距离矩阵 MSE Loss（旋转平移不变）
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class DualModalLoss(nn.Module):
    """
    双模态损失函数
    
    Loss = λ_expr * L_expression + λ_pos * L_position
    
    其中:
    - L_expression: 表达量的 MSE 损失
    - L_position: 位置距离矩阵的 MSE 损失（旋转平移不变）
    """
    
    def __init__(
        self,
        lambda_expression: float = 1.0,
        lambda_position: float = 1.0,
        use_distance_matrix: bool = True
    ):
        """
        初始化损失函数
        
        Args:
            lambda_expression: 表达量损失权重
            lambda_position: 位置损失权重
            use_distance_matrix: 是否使用距离矩阵计算位置损失（旋转不变）
        """
        super().__init__()
        self.lambda_expression = lambda_expression
        self.lambda_position = lambda_position
        self.use_distance_matrix = use_distance_matrix
        self.mse = nn.MSELoss(reduction='none')
        
    def compute_expression_loss(
        self,
        pred_expression: torch.Tensor,
        true_expression: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算表达量损失（MSE）
        
        Args:
            pred_expression: (B, N, G) 预测的表达量
            true_expression: (B, N, G) 真实的表达量
            node_mask: (B, N) 有效节点掩码
        
        Returns:
            loss: 标量损失值
        """
        # 计算 MSE
        mse = self.mse(pred_expression, true_expression)  # (B, N, G)
        
        # 应用掩码
        mask_expanded = node_mask.unsqueeze(-1)  # (B, N, 1)
        masked_mse = mse * mask_expanded
        
        # 计算平均损失
        num_valid = node_mask.sum() * pred_expression.shape[-1]
        num_valid = torch.clamp(num_valid, min=1)
        
        loss = masked_mse.sum() / num_valid
        return loss
    
    def compute_position_loss_mse(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算位置损失（直接 MSE，不具有旋转不变性）
        
        Args:
            pred_positions: (B, N, 3) 预测的位置
            true_positions: (B, N, 3) 真实的位置
            node_mask: (B, N) 有效节点掩码
        
        Returns:
            loss: 标量损失值
        """
        mse = self.mse(pred_positions, true_positions)  # (B, N, 3)
        
        mask_expanded = node_mask.unsqueeze(-1)  # (B, N, 1)
        masked_mse = mse * mask_expanded
        
        num_valid = node_mask.sum() * 3
        num_valid = torch.clamp(num_valid, min=1)
        
        loss = masked_mse.sum() / num_valid
        return loss
    
    def compute_position_loss_distance_matrix(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算位置损失（距离矩阵 MSE，具有旋转平移不变性）
        
        Args:
            pred_positions: (B, N, 3) 预测的位置
            true_positions: (B, N, 3) 真实的位置
            node_mask: (B, N) 有效节点掩码
        
        Returns:
            loss: 标量损失值
        """
        batch_size = pred_positions.shape[0]
        losses = []
        
        for b in range(batch_size):
            mask = node_mask[b].bool()
            
            # 提取有效节点
            pred_pos = pred_positions[b][mask]  # (n_valid, 3)
            true_pos = true_positions[b][mask]  # (n_valid, 3)
            
            if pred_pos.shape[0] < 2:
                # 节点数太少，跳过
                continue
            
            # 计算成对距离矩阵
            pred_dist = torch.cdist(pred_pos, pred_pos, p=2)  # (n_valid, n_valid)
            true_dist = torch.cdist(true_pos, true_pos, p=2)  # (n_valid, n_valid)
            
            # 计算距离矩阵的 MSE
            dist_mse = torch.mean((pred_dist - true_dist) ** 2)
            losses.append(dist_mse)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=pred_positions.device)
        
        return torch.stack(losses).mean()
    
    def forward(
        self,
        pred_expression: Optional[torch.Tensor],
        pred_positions: Optional[torch.Tensor],
        true_expression: Optional[torch.Tensor],
        true_positions: Optional[torch.Tensor],
        node_mask: torch.Tensor,
        compute_expression: bool = True,
        compute_position: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            pred_expression: (B, N, G) 预测的表达量
            pred_positions: (B, N, 3) 预测的位置
            true_expression: (B, N, G) 真实的表达量
            true_positions: (B, N, 3) 真实的位置
            node_mask: (B, N) 有效节点掩码
            compute_expression: 是否计算表达量损失
            compute_position: 是否计算位置损失
        
        Returns:
            total_loss: 总损失
            log_dict: 各项损失的日志字典
        """
        total_loss = torch.tensor(0.0, device=node_mask.device)
        log_dict = {}
        
        # 表达量损失
        if compute_expression and pred_expression is not None and true_expression is not None:
            expr_loss = self.compute_expression_loss(
                pred_expression, true_expression, node_mask
            )
            total_loss = total_loss + self.lambda_expression * expr_loss
            log_dict["loss/expression_mse"] = expr_loss.item()
        
        # 位置损失
        if compute_position and pred_positions is not None and true_positions is not None:
            if self.use_distance_matrix:
                pos_loss = self.compute_position_loss_distance_matrix(
                    pred_positions, true_positions, node_mask
                )
                log_dict["loss/position_dist_matrix"] = pos_loss.item()
            else:
                pos_loss = self.compute_position_loss_mse(
                    pred_positions, true_positions, node_mask
                )
                log_dict["loss/position_mse"] = pos_loss.item()
            
            total_loss = total_loss + self.lambda_position * pos_loss
        
        log_dict["loss/total"] = total_loss.item()
        
        return total_loss, log_dict


class ExpressionOnlyLoss(nn.Module):
    """仅表达量的损失函数"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        pred_expression: torch.Tensor,
        true_expression: torch.Tensor,
        node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse = self.mse(pred_expression, true_expression)
        mask_expanded = node_mask.unsqueeze(-1)
        masked_mse = mse * mask_expanded
        
        num_valid = node_mask.sum() * pred_expression.shape[-1]
        num_valid = torch.clamp(num_valid, min=1)
        
        loss = masked_mse.sum() / num_valid
        
        return loss, {"loss/expression_mse": loss.item()}


class PositionOnlyLoss(nn.Module):
    """仅位置的损失函数（距离矩阵）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred_positions: torch.Tensor,
        true_positions: torch.Tensor,
        node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = pred_positions.shape[0]
        losses = []
        
        for b in range(batch_size):
            mask = node_mask[b].bool()
            pred_pos = pred_positions[b][mask]
            true_pos = true_positions[b][mask]
            
            if pred_pos.shape[0] < 2:
                continue
            
            pred_dist = torch.cdist(pred_pos, pred_pos, p=2)
            true_dist = torch.cdist(true_pos, true_pos, p=2)
            
            dist_mse = torch.mean((pred_dist - true_dist) ** 2)
            losses.append(dist_mse)
        
        if len(losses) == 0:
            loss = torch.tensor(0.0, device=pred_positions.device)
        else:
            loss = torch.stack(losses).mean()
        
        return loss, {"loss/position_dist_matrix": loss.item()}
