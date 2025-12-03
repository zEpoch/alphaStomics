"""
NoiseModel: 双模态噪声模型
支持同时对表达量和位置加噪
支持 Masked Diffusion (对特征维度进行 masking)
"""
import torch
from typing import Optional, Tuple, TYPE_CHECKING
from alphastomics.utils.dataholder import DataHolder, remove_mean_with_mask
from alphastomics.diffusion_model.diffusion_utils import (
    cosine_beta_schedule_discrete,
    sample_gaussian_with_mask,
)

if TYPE_CHECKING:
    from alphastomics.diffusion_model.masking import MaskedDiffusionModule, MaskInfo


class NoiseModel:
    """
    双模态噪声模型
    
    同时处理表达量和位置的扩散过程：
    - 表达量: z_t^{expr} = α_t * x_{expr} + σ_t * ε_{expr}
    - 位置:   z_t^{pos}  = α_t * x_{pos}  + σ_t * ε_{pos}
    
    支持 Masked Diffusion:
    - 可选地对加噪后的特征进行 masking
    - 强迫模型从部分观测重建完整信息
    """
    
    def __init__(self, cfg: dict):
        """
        初始化噪声模型
        
        Args:
            cfg: 配置字典，包含:
                - diffusion_steps: 扩散步数
                - diffusion_noise_schedule: 噪声调度类型 ("cosine")
                - nu_expression: 表达量的 nu 参数
                - nu_position: 位置的 nu 参数
        """
        # 组件映射: 0=expression, 1=position
        self.mapping = ["expr", "pos"]
        self.inverse_mapping = {m: i for i, m in enumerate(self.mapping)}
        
        # 提取 nu 值
        self.nu_expr = cfg.get("nu_expression", 1.0)
        self.nu_pos = cfg.get("nu_position", 1.0)
        self.nu_arr = [self.nu_expr, self.nu_pos]
        
        # 扩散参数
        self.noise_schedule = cfg.get("diffusion_noise_schedule", "cosine")
        self.timesteps = cfg.get("diffusion_steps", 1000)
        self.max_diffusion_steps = self.timesteps
        
        # 初始化 beta 值
        if self.noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(self.timesteps, self.nu_arr)
        else:
            raise NotImplementedError(f"Unknown noise schedule: {self.noise_schedule}")
        
        # 计算 alpha 和相关参数
        self._betas = torch.from_numpy(betas).float()
        self._alphas = 1 - torch.clamp(self._betas, min=0, max=0.9999)
        log_alpha = torch.log(self._alphas)
        
        # 累积 log alpha
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self._log_alpha_bar = log_alpha_bar
        self._alphas_bar = torch.exp(log_alpha_bar)
        
        # 计算 sigma
        self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
        self._sigma_bar = torch.sqrt(self._sigma2_bar)
        
    def get_alpha_bar(
        self,
        t_int: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """
        获取 alpha_bar(t)
        
        Args:
            t_int: (B, 1) 整数时间步
            key: "expr" 或 "pos"
        
        Returns:
            alpha_bar: (B,) 对应的 alpha_bar 值
        """
        idx = self.inverse_mapping[key]
        a = self._alphas_bar.to(t_int.device)[t_int.long(), idx]
        return a.float().squeeze(-1)  # 确保返回 (B,)
    
    def get_sigma_bar(
        self,
        t_int: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """
        获取 sigma_bar(t)
        
        Args:
            t_int: (B, 1) 整数时间步
            key: "expr" 或 "pos"
        
        Returns:
            sigma_bar: (B,) 对应的 sigma_bar 值
        """
        idx = self.inverse_mapping[key]
        s = self._sigma_bar.to(t_int.device)[t_int.long(), idx]
        return s.float().squeeze(-1)  # 确保返回 (B,)
    
    def get_alpha_ts(
        self,
        s_int: torch.Tensor,
        t_int: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """
        计算 alpha_t / alpha_s
        
        Args:
            s_int: 目标时间步
            t_int: 源时间步
            key: "expr" 或 "pos"
        
        Returns:
            ratio: alpha_t / alpha_s
        """
        idx = self.inverse_mapping[key]
        log_a_bar = self._log_alpha_bar[..., idx].to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int.long()] - log_a_bar[s_int.long()])
        return ratio.float().squeeze(-1)
    
    def get_alpha_ts_sq(
        self,
        s_int: torch.Tensor,
        t_int: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """计算 (alpha_t / alpha_s)^2"""
        idx = self.inverse_mapping[key]
        log_a_bar = self._log_alpha_bar[..., idx].to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int.long()] - 2 * log_a_bar[s_int.long()])
        return ratio.float().squeeze(-1)
    
    def get_sigma_sq_ratio(
        self,
        s_int: torch.Tensor,
        t_int: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """计算 sigma_s^2 / sigma_t^2"""
        idx = self.inverse_mapping[key]
        log_a_bar = self._log_alpha_bar[..., idx].to(t_int.device)
        s2_s = -torch.expm1(2 * log_a_bar[s_int.long()])
        s2_t = -torch.expm1(2 * log_a_bar[t_int.long()])
        ratio = torch.exp(torch.log(s2_s + 1e-8) - torch.log(s2_t + 1e-8))
        return ratio.float().squeeze(-1)
    
    def get_prefactor(
        self,
        s_int: torch.Tensor,
        t_int: torch.Tensor,
        key: str
    ) -> torch.Tensor:
        """
        计算后验均值中 x_0 的系数
        prefactor = α_s * (1 - α_{t/s}^2 * σ_s^2 / σ_t^2)
        """
        a_s = self.get_alpha_bar(t_int=s_int, key=key)
        alpha_ratio_sq = self.get_alpha_ts_sq(s_int=s_int, t_int=t_int, key=key)
        sigma_ratio_sq = self.get_sigma_sq_ratio(s_int=s_int, t_int=t_int, key=key)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()
    
    def apply_noise(
        self,
        data: DataHolder,
        noise_expression: bool = True,
        noise_position: bool = True,
        masking_module: Optional['MaskedDiffusionModule'] = None,
        apply_masking: bool = False
    ) -> Tuple[DataHolder, Optional['MaskInfo']]:
        """
        对数据应用噪声（前向扩散过程）
        
        Args:
            data: 包含原始数据的 DataHolder
            noise_expression: 是否对表达量加噪
            noise_position: 是否对位置加噪
            masking_module: MaskedDiffusionModule 实例（可选）
            apply_masking: 是否应用 masking
        
        Returns:
            noisy_data: 包含噪声数据的 DataHolder
            mask_info: MaskInfo 实例（如果应用了 masking）
        """
        batch_size = data.expression.shape[0]
        device = data.expression.device
        
        # 随机采样时间步 t ∈ [1, T]
        t_int = torch.randint(
            1, self.max_diffusion_steps + 1,
            size=(batch_size, 1),
            device=device
        )
        t_float = t_int.float() / self.max_diffusion_steps
        
        noisy_data = data.copy()
        noisy_data.t_int = t_int
        noisy_data.t = t_float
        noisy_data.diffusion_time = t_float
        
        # 对表达量加噪
        if noise_expression and data.expression is not None:
            noise_expr = torch.randn_like(data.expression)
            noise_expr = noise_expr * data.node_mask.unsqueeze(-1)
            
            alpha_expr = self.get_alpha_bar(t_int=t_int, key="expr").view(-1, 1, 1)  # (B, 1, 1)
            sigma_expr = self.get_sigma_bar(t_int=t_int, key="expr").view(-1, 1, 1)  # (B, 1, 1)
            
            noisy_data.noisy_expression = alpha_expr * data.expression + sigma_expr * noise_expr
        else:
            noisy_data.noisy_expression = data.expression
        
        # 对位置加噪
        if noise_position and data.positions is not None:
            noise_pos = torch.randn_like(data.positions)
            noise_pos = noise_pos * data.node_mask.unsqueeze(-1)
            noise_pos = remove_mean_with_mask(noise_pos, data.node_mask)
            
            alpha_pos = self.get_alpha_bar(t_int=t_int, key="pos").view(-1, 1, 1)  # (B, 1, 1)
            sigma_pos = self.get_sigma_bar(t_int=t_int, key="pos").view(-1, 1, 1)  # (B, 1, 1)
            
            noisy_data.noisy_positions = alpha_pos * data.positions + sigma_pos * noise_pos
        else:
            noisy_data.noisy_positions = data.positions
        
        # 应用 Masking（在加噪后）
        mask_info = None
        if masking_module is not None and apply_masking:
            masked_expr, masked_pos, mask_info = masking_module.apply_masking(
                expression=noisy_data.noisy_expression,
                position=noisy_data.noisy_positions,
                apply=True
            )
            noisy_data.noisy_expression = masked_expr
            noisy_data.noisy_positions = masked_pos
            
        return noisy_data, mask_info
    
    def sample_limit_dist(
        self,
        expression: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
        cell_class: Optional[torch.Tensor] = None,
        cell_ID: Optional[torch.Tensor] = None,
        noise_expression: bool = True,
        noise_position: bool = True
    ) -> DataHolder:
        """
        从极限分布（纯噪声）采样初始状态
        
        Args:
            expression: (B, N, G) 原始表达量（作为条件或全噪声）
            positions: (B, N, 3) 原始位置（作为条件或全噪声）
            node_mask: (B, N) 有效节点掩码
            cell_class: 细胞类型
            cell_ID: 细胞ID
            noise_expression: 是否对表达量使用纯噪声
            noise_position: 是否对位置使用纯噪声
        
        Returns:
            z_T: 初始噪声状态
        """
        batch_size, num_nodes, num_genes = expression.shape
        device = expression.device
        
        # 初始化噪声表达量
        if noise_expression:
            noisy_expr = torch.randn(batch_size, num_nodes, num_genes, device=device)
            noisy_expr = noisy_expr * node_mask.unsqueeze(-1)
        else:
            noisy_expr = expression  # 使用原始表达量作为条件
        
        # 初始化噪声位置
        if noise_position:
            noisy_pos = torch.randn(batch_size, num_nodes, 3, device=device)
            noisy_pos = noisy_pos * node_mask.unsqueeze(-1)
            noisy_pos = remove_mean_with_mask(noisy_pos, node_mask)
        else:
            noisy_pos = positions  # 使用原始位置作为条件
        
        # 创建时间信息
        t_array = torch.ones((batch_size, 1), device=device)
        t_int_array = (self.max_diffusion_steps * t_array).long()
        
        return DataHolder(
            expression=expression,
            positions=positions,
            node_mask=node_mask,
            cell_class=cell_class,
            cell_ID=cell_ID,
            t_int=t_int_array,
            t=t_array,
            diffusion_time=t_array,
            noisy_expression=noisy_expr,
            noisy_positions=noisy_pos,
        )
    
    def sample_zs_from_zt_and_pred(
        self,
        z_t: DataHolder,
        pred_expression: torch.Tensor,
        pred_positions: torch.Tensor,
        s_int: torch.Tensor,
        denoise_expression: bool = True,
        denoise_position: bool = True
    ) -> DataHolder:
        """
        从 z_t 和预测采样 z_s（逆向扩散一步）
        
        p(z_s | z_t) 的采样
        
        Args:
            z_t: 当前状态
            pred_expression: 模型预测的原始表达量
            pred_positions: 模型预测的原始位置
            s_int: 目标时间步
            denoise_expression: 是否对表达量去噪
            denoise_position: 是否对位置去噪
        
        Returns:
            z_s: 去噪后的状态
        """
        node_mask = z_t.node_mask
        t_int = z_t.t_int
        device = z_t.expression.device
        
        z_s = z_t.copy()
        z_s.t_int = s_int
        z_s.t = s_int.float() / self.max_diffusion_steps
        z_s.diffusion_time = z_s.t
        
        # 对表达量去噪
        if denoise_expression and z_t.noisy_expression is not None:
            sigma_sq_ratio = self.get_sigma_sq_ratio(s_int=s_int, t_int=t_int, key="expr")
            z_t_prefactor = (self.get_alpha_ts(s_int=s_int, t_int=t_int, key="expr") * sigma_sq_ratio)
            pred_prefactor = self.get_prefactor(s_int=s_int, t_int=t_int, key="expr")
            
            # 确保系数是 (B, 1, 1) 用于广播
            z_t_prefactor = z_t_prefactor.view(-1, 1, 1)  # (B, 1, 1)
            pred_prefactor = pred_prefactor.view(-1, 1, 1)  # (B, 1, 1)
            
            # 计算均值
            mu_expr = z_t_prefactor * z_t.noisy_expression + pred_prefactor * pred_expression
            
            # 采样噪声
            noise_expr = torch.randn_like(z_t.noisy_expression) * node_mask.unsqueeze(-1)
            
            # 计算噪声系数
            sigma_t = self.get_sigma_bar(t_int=t_int, key="expr")
            sigma_s = self.get_sigma_bar(t_int=s_int, key="expr")
            alpha_ts_sq = self.get_alpha_ts_sq(s_int=s_int, t_int=t_int, key="expr")
            sigma2_t_s = sigma_t - sigma_s * alpha_ts_sq
            noise_prefactor = torch.sqrt(torch.clamp(sigma2_t_s * sigma_sq_ratio, min=0))
            noise_prefactor = noise_prefactor.view(-1, 1, 1)  # (B, 1, 1)
            
            z_s.noisy_expression = mu_expr + noise_prefactor * noise_expr
        
        # 对位置去噪
        if denoise_position and z_t.noisy_positions is not None:
            sigma_sq_ratio = self.get_sigma_sq_ratio(s_int=s_int, t_int=t_int, key="pos")
            z_t_prefactor = (self.get_alpha_ts(s_int=s_int, t_int=t_int, key="pos") * sigma_sq_ratio)
            pred_prefactor = self.get_prefactor(s_int=s_int, t_int=t_int, key="pos")
            
            # 确保系数是 1D (B,) 然后扩展到 (B, 1, 1) 用于广播
            z_t_prefactor = z_t_prefactor.view(-1, 1, 1)  # (B, 1, 1)
            pred_prefactor = pred_prefactor.view(-1, 1, 1)  # (B, 1, 1)
            
            # 计算均值
            mu_pos = z_t_prefactor * z_t.noisy_positions + pred_prefactor * pred_positions
            
            # 采样噪声
            noise_pos = torch.randn_like(z_t.noisy_positions) * node_mask.unsqueeze(-1)
            noise_pos = remove_mean_with_mask(noise_pos, node_mask)
            
            # 计算噪声系数
            sigma_t = self.get_sigma_bar(t_int=t_int, key="pos")
            sigma_s = self.get_sigma_bar(t_int=s_int, key="pos")
            alpha_ts_sq = self.get_alpha_ts_sq(s_int=s_int, t_int=t_int, key="pos")
            sigma2_t_s = sigma_t - sigma_s * alpha_ts_sq
            noise_prefactor = torch.sqrt(torch.clamp(sigma2_t_s * sigma_sq_ratio, min=0))
            noise_prefactor = noise_prefactor.view(-1, 1, 1)  # (B, 1, 1)
            
            z_s.noisy_positions = mu_pos + noise_prefactor * noise_pos
            # 中心化
            z_s.noisy_positions = remove_mean_with_mask(z_s.noisy_positions, node_mask)
        
        return z_s
