"""
扩散模型工具函数
包括噪声调度、辅助计算等
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


def cosine_beta_schedule_discrete(
    timesteps: int, 
    nu_arr: np.ndarray, 
    s: float = 0.008
) -> np.ndarray:
    """
    生成离散的 cosine beta 调度
    
    Args:
        timesteps: 扩散步数
        nu_arr: 每个组件的 nu 值数组，控制噪声调度速度
                [nu_expression, nu_position]
        s: cosine 调度的偏移参数
    
    Returns:
        betas: (timesteps, num_components) 的 beta 值数组
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    x = np.expand_dims(x, 0)  # (1, steps)

    nu_arr = np.array(nu_arr)  # (components,)
    nu_arr = np.expand_dims(nu_arr, 1)  # (components, 1)

    # 计算 alpha_cumprod
    alphas_cumprod = (
        np.cos(0.5 * np.pi * (((x / steps) ** nu_arr) + s) / (1 + s)) ** 2
    )  # (components, steps)
    
    # 归一化
    alphas_cumprod_new = alphas_cumprod / np.expand_dims(alphas_cumprod[:, 0], 1)
    
    # 计算 alphas
    alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]
    
    # 计算 betas
    betas = 1 - alphas  # (components, steps)
    betas = np.swapaxes(betas, 0, 1)  # (steps, components)

    return betas


def sample_gaussian_with_mask(
    size: tuple, 
    node_mask: torch.Tensor
) -> torch.Tensor:
    """
    采样带掩码的高斯噪声
    
    Args:
        size: 张量形状
        node_mask: (B, N) 节点掩码
    
    Returns:
        masked_noise: 掩码后的噪声
    """
    noise = torch.randn(size, device=node_mask.device)
    if len(size) == 3:
        noise = noise * node_mask.unsqueeze(-1)
    else:
        noise = noise * node_mask
    return noise


def remove_mean_with_mask(
    x: torch.Tensor, 
    node_mask: torch.Tensor
) -> torch.Tensor:
    """
    移除位置的均值（考虑掩码）
    
    Args:
        x: (B, N, D) 需要中心化的张量
        node_mask: (B, N) 有效节点掩码
    
    Returns:
        centered_x: 中心化后的张量
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


def inflate_batch_array(
    array: torch.Tensor, 
    target_shape: tuple
) -> torch.Tensor:
    """
    将批次数组扩展到目标形状
    
    Args:
        array: (B,) 或 (B, 1) 的数组
        target_shape: 目标形状
    
    Returns:
        inflated: 扩展后的数组
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def gaussian_KL(
    q_mu: torch.Tensor, 
    q_sigma: torch.Tensor
) -> torch.Tensor:
    """
    计算高斯分布与标准正态分布之间的 KL 散度
    
    Args:
        q_mu: 分布 q 的均值
        q_sigma: 分布 q 的标准差
    
    Returns:
        kl: KL 散度
    """
    return torch.log(1 / q_sigma) + 0.5 * (q_sigma**2 + q_mu**2) - 0.5


def SNR(gamma: torch.Tensor) -> torch.Tensor:
    """计算信噪比 (alpha^2/sigma^2)"""
    return torch.exp(-gamma)


def assert_correctly_masked(
    variable: torch.Tensor, 
    node_mask: torch.Tensor
) -> None:
    """
    断言张量被正确掩码
    
    Args:
        variable: 输入张量
        node_mask: 节点掩码
    
    Raises:
        AssertionError: 如果存在 NaN 或未正确掩码
    """
    assert not torch.isnan(variable).any(), f"NaN detected, shape: {variable.shape}"
    
    masked_variable = variable * (1 - node_mask.long().unsqueeze(-1))
    max_val = masked_variable.abs().max().item()
    assert max_val < 1e-4, f"Variables not masked properly: max={max_val}"
