"""
DiffusionSampler: 扩散模型采样器
支持多种采样模式：
1. 表达量 → 位置
2. 位置 → 表达量
3. 联合生成（两者都是噪声）
"""
import torch
from typing import Optional, Tuple, Literal
from alphastomics.utils.dataholder import DataHolder
from alphastomics.diffusion_model.noise_model import NoiseModel


class DiffusionSampler:
    """
    扩散模型采样器
    
    支持三种采样模式:
    - "expr_to_pos": 表达量作为条件，生成位置
    - "pos_to_expr": 位置作为条件，生成表达量
    - "joint": 联合生成表达量和位置
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        noise_model: NoiseModel,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        初始化采样器
        
        Args:
            model: 扩散模型（预测 x_0）
            noise_model: 噪声模型
            device: 设备
        """
        self.model = model
        self.noise_model = noise_model
        self.device = device
        self.max_diffusion_steps = noise_model.max_diffusion_steps
    
    @torch.no_grad()
    def sample(
        self,
        expression: torch.Tensor,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
        mode: Literal["expr_to_pos", "pos_to_expr", "joint"] = "expr_to_pos",
        cell_class: Optional[torch.Tensor] = None,
        cell_ID: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行采样
        
        Args:
            expression: (B, N, G) 表达量
            positions: (B, N, 3) 位置
            node_mask: (B, N) 有效节点掩码
            mode: 采样模式
                - "expr_to_pos": 使用表达量预测位置
                - "pos_to_expr": 使用位置预测表达量
                - "joint": 同时生成表达量和位置
            cell_class: 细胞类型
            cell_ID: 细胞ID
            num_steps: 采样步数（默认使用 noise_model 的步数）
            verbose: 是否打印进度
        
        Returns:
            sampled_expression: (B, N, G) 采样的表达量
            sampled_positions: (B, N, 3) 采样的位置
        """
        self.model.eval()
        
        if num_steps is None:
            num_steps = self.max_diffusion_steps
        
        # 根据模式确定哪些需要加噪
        if mode == "expr_to_pos":
            noise_expression = False
            noise_position = True
        elif mode == "pos_to_expr":
            noise_expression = True
            noise_position = False
        elif mode == "joint":
            noise_expression = True
            noise_position = True
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 初始化：从极限分布采样
        z_t = self.noise_model.sample_limit_dist(
            expression=expression,
            positions=positions,
            node_mask=node_mask,
            cell_class=cell_class,
            cell_ID=cell_ID,
            noise_expression=noise_expression,
            noise_position=noise_position
        )
        z_t = z_t.device_as(expression)
        
        if verbose:
            print(f"Sampling mode: {mode}")
            print(f"Number of nodes: {node_mask.sum().item()}")
            print(f"Number of steps: {num_steps}")
        
        # 逐步去噪
        step_interval = max(1, self.max_diffusion_steps // num_steps)
        
        for s_int_val in reversed(range(0, self.max_diffusion_steps, step_interval)):
            s_int = torch.full(
                (expression.shape[0], 1),
                s_int_val,
                dtype=torch.long,
                device=self.device
            )
            
            # 模型预测
            pred_expression, pred_positions = self.model(
                expression_features=z_t.noisy_expression,
                diffusion_time=z_t.diffusion_time,
                position_features=z_t.noisy_positions,
                node_mask=node_mask
            )
            
            # 采样下一步
            z_t = self.noise_model.sample_zs_from_zt_and_pred(
                z_t=z_t,
                pred_expression=pred_expression,
                pred_positions=pred_positions,
                s_int=s_int,
                denoise_expression=noise_expression,
                denoise_position=noise_position
            )
            
            if verbose and s_int_val % 100 == 0:
                print(f"  Step {self.max_diffusion_steps - s_int_val}/{self.max_diffusion_steps}")
        
        # 返回结果
        if mode == "expr_to_pos":
            return expression, z_t.noisy_positions
        elif mode == "pos_to_expr":
            return z_t.noisy_expression, positions
        else:  # joint
            return z_t.noisy_expression, z_t.noisy_positions
    
    @torch.no_grad()
    def sample_expression_from_position(
        self,
        positions: torch.Tensor,
        node_mask: torch.Tensor,
        num_genes: int,
        num_steps: Optional[int] = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        从位置生成表达量
        
        Args:
            positions: (B, N, 3) 已知的位置
            node_mask: (B, N) 有效节点掩码
            num_genes: 基因数量
            num_steps: 采样步数
            verbose: 是否打印进度
        
        Returns:
            expression: (B, N, G) 生成的表达量
        """
        batch_size, num_nodes, _ = positions.shape
        
        # 创建占位表达量（会被替换为噪声）
        dummy_expression = torch.zeros(
            batch_size, num_nodes, num_genes,
            device=self.device
        )
        
        expression, _ = self.sample(
            expression=dummy_expression,
            positions=positions,
            node_mask=node_mask,
            mode="pos_to_expr",
            num_steps=num_steps,
            verbose=verbose
        )
        
        return expression
    
    @torch.no_grad()
    def sample_position_from_expression(
        self,
        expression: torch.Tensor,
        node_mask: torch.Tensor,
        num_steps: Optional[int] = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        从表达量生成位置（原始 LUNA 任务）
        
        Args:
            expression: (B, N, G) 已知的表达量
            node_mask: (B, N) 有效节点掩码
            num_steps: 采样步数
            verbose: 是否打印进度
        
        Returns:
            positions: (B, N, 3) 生成的位置
        """
        batch_size, num_nodes, _ = expression.shape
        
        # 创建占位位置（会被替换为噪声）
        dummy_positions = torch.zeros(
            batch_size, num_nodes, 3,
            device=self.device
        )
        
        _, positions = self.sample(
            expression=expression,
            positions=dummy_positions,
            node_mask=node_mask,
            mode="expr_to_pos",
            num_steps=num_steps,
            verbose=verbose
        )
        
        return positions
    
    @torch.no_grad()
    def sample_joint(
        self,
        batch_size: int,
        num_nodes: int,
        num_genes: int,
        node_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        联合生成表达量和位置
        
        Args:
            batch_size: 批次大小
            num_nodes: 节点数量
            num_genes: 基因数量
            node_mask: 有效节点掩码（如果为 None，则所有节点有效）
            num_steps: 采样步数
            verbose: 是否打印进度
        
        Returns:
            expression: (B, N, G) 生成的表达量
            positions: (B, N, 3) 生成的位置
        """
        if node_mask is None:
            node_mask = torch.ones(batch_size, num_nodes, device=self.device)
        
        # 创建占位数据（会被替换为噪声）
        dummy_expression = torch.zeros(
            batch_size, num_nodes, num_genes,
            device=self.device
        )
        dummy_positions = torch.zeros(
            batch_size, num_nodes, 3,
            device=self.device
        )
        
        expression, positions = self.sample(
            expression=dummy_expression,
            positions=dummy_positions,
            node_mask=node_mask,
            mode="joint",
            num_steps=num_steps,
            verbose=verbose
        )
        
        return expression, positions
