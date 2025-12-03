"""
AlphaSTomics 训练模块
基于 PyTorch Lightning 的训练框架

支持:
- 多种训练模式 (expr_to_pos, pos_to_expr, joint)
- 自动数据加载和预处理
- 训练/验证/测试评估
- 模型检查点保存
- WandB/TensorBoard 日志
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Dict, Optional, Tuple, Literal, List, Any
from pathlib import Path
import yaml
import logging

from alphastomics.attn_model.model import Model
from alphastomics.diffusion_model.noise_model import NoiseModel
from alphastomics.diffusion_model.loss import DualModalLoss
from alphastomics.diffusion_model.sample import DiffusionSampler
from alphastomics.diffusion_model.masking import (
    MaskedDiffusionModule,
    MaskingConfig,
    MaskInfo,
)
from alphastomics.utils.dataholder import DataHolder
from alphastomics.utils.metrics import MetricsCalculator, print_metrics

logger = logging.getLogger(__name__)


class AlphaSTomicsModule(pl.LightningModule):
    """
    AlphaSTomics PyTorch Lightning 模块
    
    支持三种训练模式:
    - "expr_to_pos": 表达量 → 位置（原始 LUNA 任务）
    - "pos_to_expr": 位置 → 表达量
    - "joint": 联合训练（两者都加噪）
    
    支持 Masked Diffusion:
    - 对加噪后的特征进行 masking
    - 强迫模型从部分观测重建完整信息
    - 可配置 mask 比例、策略等
    """
    
    def __init__(
        self,
        cfg: dict,
        num_genes: int,
    ):
        """
        初始化训练模块
        
        Args:
            cfg: 配置字典
            num_genes: 基因数量
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self.num_genes = num_genes
        
        # 训练模式
        self.training_mode = cfg.get("training_mode", "joint")
        
        # 初始化模型
        self.model = Model(
            input_dims=num_genes,
            mlp_in_expression_setting=cfg["mlp_in_expression_setting"],
            mlp_in_diffusion_time_setting=cfg["mlp_in_diffusion_time_setting"],
            PositionMLP_setting=cfg["PositionMLP_setting"],
            TransformerLayer_setting=cfg["TransformerLayer_setting"],
            mlp_out_expression_setting=cfg["mlp_out_expression_setting"],
            mlp_out_position_norm_setting=cfg["mlp_out_position_norm_setting"],
        )
        
        # 初始化噪声模型
        diffusion_cfg = cfg.get("diffusion", {})
        self.noise_model = NoiseModel(diffusion_cfg)
        self.max_diffusion_steps = self.noise_model.max_diffusion_steps
        
        # 初始化损失函数
        loss_cfg = cfg.get("loss", {})
        masking_cfg = cfg.get("masking", {})
        
        self.loss_fn = DualModalLoss(
            lambda_expression=loss_cfg.get("lambda_expression", 1.0),
            lambda_position=loss_cfg.get("lambda_position", 1.0),
            use_distance_matrix=loss_cfg.get("use_distance_matrix", True),
            lambda_masked_reconstruction=masking_cfg.get("reconstruction_weight", 0.5)
        )
        
        # 初始化 Masked Diffusion 模块（可选）
        self.masking_module = None
        self.masking_config = None
        
        if masking_cfg.get("enable", False):
            self.masking_config = MaskingConfig.from_dict(masking_cfg)
            self.masking_module = MaskedDiffusionModule(
                expression_dim=num_genes,
                position_dim=3,
                config=self.masking_config
            )
            logger.info(f"Masked Diffusion 已启用: {self.masking_config.to_dict()}")
        
        # 评估指标计算器
        self.metrics_calculator = MetricsCalculator(k_neighbors=10)
        
        # 测试阶段收集的结果
        self.test_outputs: List[Dict] = []
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self._global_step = 0
    
    def forward(
        self,
        expression: torch.Tensor,
        positions: torch.Tensor,
        diffusion_time: torch.Tensor,
        node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            expression: (B, N, G) 加噪后的表达量
            positions: (B, N, 3) 加噪后的位置
            diffusion_time: (B, 1) 扩散时间
            node_mask: (B, N) 节点掩码
        
        Returns:
            pred_expression: (B, N, G) 预测的原始表达量
            pred_positions: (B, N, 3) 预测的原始位置
        """
        return self.model(
            expression_features=expression,
            diffusion_time=diffusion_time,
            position_features=positions,
            node_mask=node_mask
        )
    
    def _prepare_batch(self, batch: Any) -> DataHolder:
        """
        将 batch 转换为 DataHolder 格式
        
        支持 dict 格式（来自 dataloader）和 DataHolder 格式
        """
        if isinstance(batch, DataHolder):
            return batch
        
        if isinstance(batch, dict):
            return DataHolder(
                expression=batch['expression'],
                positions=batch['positions'],
                node_mask=batch['node_mask'],
                cell_class=batch.get('cell_types'),
            )
        
        raise ValueError(f"Unknown batch type: {type(batch)}")
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 数据批次（dict 或 DataHolder）
            batch_idx: 批次索引
        
        Returns:
            loss: 训练损失
        """
        batch = self._prepare_batch(batch)
        
        # 确定是否对各模态加噪
        noise_expression = self.training_mode in ["pos_to_expr", "joint"]
        noise_position = self.training_mode in ["expr_to_pos", "joint"]
        
        # 确定是否应用 masking
        apply_masking = False
        if self.masking_module is not None:
            self.masking_module.set_step(self._global_step)
            apply_masking = self.masking_module.should_apply_masking()
        
        # 应用噪声（和可选的 masking）
        noisy_data, mask_info = self.noise_model.apply_noise(
            batch,
            noise_expression=noise_expression,
            noise_position=noise_position,
            masking_module=self.masking_module,
            apply_masking=apply_masking
        )
        
        # 模型预测
        pred_expression, pred_positions = self.forward(
            expression=noisy_data.noisy_expression,
            positions=noisy_data.noisy_positions,
            diffusion_time=noisy_data.diffusion_time,
            node_mask=batch.node_mask
        )
        
        # 计算扩散损失
        loss, log_dict = self.loss_fn(
            pred_expression=pred_expression,
            pred_positions=pred_positions,
            true_expression=batch.expression,
            true_positions=batch.positions,
            node_mask=batch.node_mask,
            compute_expression=noise_expression,
            compute_position=noise_position
        )
        
        # 计算 Masked Reconstruction 损失（如果启用）
        if mask_info is not None and mask_info.has_mask():
            recon_loss, recon_log_dict = self.loss_fn.compute_masked_reconstruction_loss(
                pred_expression=pred_expression,
                pred_positions=pred_positions,
                mask_info=mask_info,
                node_mask=batch.node_mask
            )
            loss = loss + recon_loss
            log_dict.update(recon_log_dict)
            log_dict["masking/applied"] = 1.0
        else:
            log_dict["masking/applied"] = 0.0
        
        # 记录日志
        self.log_dict(
            {f"train/{k}": v for k, v in log_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
        )
        
        self._global_step += 1

        
        return loss
    
    def validation_step(self, batch: DataHolder, batch_idx: int) -> torch.Tensor:
        """验证步骤（不使用 masking）"""
        noise_expression = self.training_mode in ["pos_to_expr", "joint"]
        noise_position = self.training_mode in ["expr_to_pos", "joint"]
        
        # 验证时不使用 masking
        noisy_data, _ = self.noise_model.apply_noise(
            batch,
            noise_expression=noise_expression,
            noise_position=noise_position,
            masking_module=None,
            apply_masking=False
        )
        
        pred_expression, pred_positions = self.forward(
            expression=noisy_data.noisy_expression,
            positions=noisy_data.noisy_positions,
            diffusion_time=noisy_data.diffusion_time,
            node_mask=batch.node_mask
        )
        
        loss, log_dict = self.loss_fn(
            pred_expression=pred_expression,
            pred_positions=pred_positions,
            true_expression=batch.expression,
            true_positions=batch.positions,
            node_mask=batch.node_mask,
            compute_expression=noise_expression,
            compute_position=noise_position
        )
        
        self.log_dict(
            {f"val/{k}": v for k, v in log_dict.items()},
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size
        )
        
        return loss
    
    def test_step(self, batch: Any, batch_idx: int) -> Dict:
        """
        测试步骤（执行完整采样并计算指标）
        """
        batch = self._prepare_batch(batch)
        
        sampler = DiffusionSampler(
            model=self.model,
            noise_model=self.noise_model,
            device=self.device
        )
        
        sampling_cfg = self.cfg.get("sampling", {})
        num_steps = sampling_cfg.get("num_steps", 100)
        
        # 根据模式执行采样
        if self.training_mode == "expr_to_pos":
            _, sampled_positions = sampler.sample(
                expression=batch.expression,
                positions=batch.positions,
                node_mask=batch.node_mask,
                mode="expr_to_pos",
                num_steps=num_steps,
                verbose=False
            )
            output = {
                "pred_positions": sampled_positions.cpu(),
                "true_positions": batch.positions.cpu(),
                "true_expression": batch.expression.cpu(),
                "node_mask": batch.node_mask.cpu()
            }
        
        elif self.training_mode == "pos_to_expr":
            sampled_expression, _ = sampler.sample(
                expression=batch.expression,
                positions=batch.positions,
                node_mask=batch.node_mask,
                mode="pos_to_expr",
                num_steps=num_steps,
                verbose=False
            )
            output = {
                "pred_expression": sampled_expression.cpu(),
                "true_expression": batch.expression.cpu(),
                "true_positions": batch.positions.cpu(),
                "node_mask": batch.node_mask.cpu()
            }
        
        else:  # joint
            sampled_expression, sampled_positions = sampler.sample(
                expression=batch.expression,
                positions=batch.positions,
                node_mask=batch.node_mask,
                mode="joint",
                num_steps=num_steps,
                verbose=False
            )
            output = {
                "pred_expression": sampled_expression.cpu(),
                "pred_positions": sampled_positions.cpu(),
                "true_expression": batch.expression.cpu(),
                "true_positions": batch.positions.cpu(),
                "node_mask": batch.node_mask.cpu()
            }
        
        self.test_outputs.append(output)
        return output
    
    def on_test_epoch_end(self):
        """测试 epoch 结束时计算综合指标"""
        import numpy as np
        
        # 收集所有输出
        all_pred_expr = []
        all_pred_pos = []
        all_true_expr = []
        all_true_pos = []
        all_masks = []
        
        for output in self.test_outputs:
            if 'pred_expression' in output:
                all_pred_expr.append(output['pred_expression'].numpy())
            if 'pred_positions' in output:
                all_pred_pos.append(output['pred_positions'].numpy())
            all_true_expr.append(output['true_expression'].numpy())
            all_true_pos.append(output['true_positions'].numpy())
            all_masks.append(output['node_mask'].numpy())
        
        # 计算指标
        if self.training_mode == "expr_to_pos":
            pred_pos = np.concatenate(all_pred_pos, axis=0)
            true_pos = np.concatenate(all_true_pos, axis=0)
            masks = np.concatenate(all_masks, axis=0)
            
            metrics = self.metrics_calculator.compute_position_only(pred_pos, true_pos, masks)
            
        elif self.training_mode == "pos_to_expr":
            pred_expr = np.concatenate(all_pred_expr, axis=0)
            true_expr = np.concatenate(all_true_expr, axis=0)
            masks = np.concatenate(all_masks, axis=0)
            
            metrics = self.metrics_calculator.compute_expression_only(pred_expr, true_expr, masks)
            
        else:  # joint
            pred_expr = np.concatenate(all_pred_expr, axis=0)
            pred_pos = np.concatenate(all_pred_pos, axis=0)
            true_expr = np.concatenate(all_true_expr, axis=0)
            true_pos = np.concatenate(all_true_pos, axis=0)
            masks = np.concatenate(all_masks, axis=0)
            
            metrics = self.metrics_calculator.compute_all(
                pred_expr, pred_pos, true_expr, true_pos, masks
            )
        
        # 记录指标
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        
        # 打印结果
        print_metrics(metrics, prefix=f"Test ({self.training_mode})")
        
        # 清空缓存
        self.test_outputs.clear()
    
    def on_validation_epoch_end(self):
        """验证 epoch 结束时的钩子"""
        # 可以在这里添加额外的验证逻辑
        pass


    
    def configure_optimizers(self):
        """配置优化器"""
        train_cfg = self.cfg.get("training", {})
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            amsgrad=True
        )
        
        # 可选的学习率调度器
        scheduler_cfg = train_cfg.get("scheduler", None)
        if scheduler_cfg is not None:
            if scheduler_cfg.get("type") == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_cfg.get("T_max", 100),
                    eta_min=scheduler_cfg.get("eta_min", 1e-6)
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch"
                    }
                }
        
        return optimizer


def create_model_from_config(cfg: dict, num_genes: int) -> AlphaSTomicsModule:
    """
    从配置创建模型
    
    Args:
        cfg: 配置字典
        num_genes: 基因数量
    
    Returns:
        model: AlphaSTomicsModule 实例
    """
    return AlphaSTomicsModule(cfg=cfg, num_genes=num_genes)


def get_callbacks(cfg: dict) -> List[pl.Callback]:
    """
    获取训练回调函数
    
    Args:
        cfg: 配置字典
    
    Returns:
        回调函数列表
    """
    callbacks = []
    
    logging_cfg = cfg.get("logging", {})
    save_dir = logging_cfg.get("save_dir", "./outputs")
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{save_dir}/checkpoints",
        filename="alphastomics-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=logging_cfg.get("save_top_k", 3),
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    training_cfg = cfg.get("training", {})
    if training_cfg.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=training_cfg.get("patience", 10),
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def get_logger(cfg: dict) -> Optional[pl.loggers.Logger]:
    """
    获取日志记录器
    
    Args:
        cfg: 配置字典
    
    Returns:
        日志记录器
    """
    logging_cfg = cfg.get("logging", {})
    save_dir = logging_cfg.get("save_dir", "./outputs")
    
    logger_type = logging_cfg.get("logger", "tensorboard")
    
    if logger_type == "wandb":
        return WandbLogger(
            project=logging_cfg.get("wandb_project", "alphastomics"),
            save_dir=save_dir,
            log_model=True,
        )
    elif logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=save_dir,
            name="alphastomics",
        )
    else:
        return None


def get_distributed_strategy(cfg: dict):
    """
    获取分布式训练策略
    
    Args:
        cfg: 配置字典
    
    Returns:
        分布式策略对象或字符串
    """
    dist_cfg = cfg.get("distributed", {})
    
    if not dist_cfg.get("enabled", False):
        return "auto"  # 单卡或自动检测
    
    strategy = dist_cfg.get("strategy", "ddp")
    
    if strategy == "ddp":
        from pytorch_lightning.strategies import DDPStrategy
        return DDPStrategy(
            find_unused_parameters=dist_cfg.get("find_unused_parameters", False),
            static_graph=not dist_cfg.get("find_unused_parameters", False),
        )
    
    elif strategy == "ddp_find_unused_parameters_true":
        from pytorch_lightning.strategies import DDPStrategy
        return DDPStrategy(
            find_unused_parameters=True,
        )
    
    elif strategy == "fsdp":
        from pytorch_lightning.strategies import FSDPStrategy
        fsdp_cfg = dist_cfg.get("fsdp", {})
        return FSDPStrategy(
            sharding_strategy=fsdp_cfg.get("sharding_strategy", "FULL_SHARD"),
            cpu_offload=fsdp_cfg.get("cpu_offload", False),
        )
    
    elif strategy == "deepspeed":
        from pytorch_lightning.strategies import DeepSpeedStrategy
        ds_cfg = dist_cfg.get("deepspeed", {})
        stage = ds_cfg.get("stage", 2)
        return DeepSpeedStrategy(
            stage=stage,
            offload_optimizer=ds_cfg.get("offload_optimizer", False),
            offload_parameters=ds_cfg.get("offload_parameters", False) if stage == 3 else False,
        )
    
    else:
        return strategy


def train(
    cfg: dict,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_genes: int,
    resume_from: Optional[str] = None,
) -> AlphaSTomicsModule:
    """
    训练模型（支持多机多卡分布式训练）
    
    Args:
        cfg: 配置字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_genes: 基因数量
        resume_from: 恢复训练的检查点路径
    
    Returns:
        训练好的模型
    
    多机多卡训练用法:
        # 1. 单机多卡 (DDP)
        torchrun --nproc_per_node=4 -m alphastomics.main train --config config.yaml
        
        # 2. 多机多卡 (DDP)
        # 在节点 0 (master):
        torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
            --master_addr=<master_ip> --master_port=29500 \
            -m alphastomics.main train --config config.yaml
        
        # 在节点 1:
        torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
            --master_addr=<master_ip> --master_port=29500 \
            -m alphastomics.main train --config config.yaml
    """
    # 创建模型
    model = create_model_from_config(cfg, num_genes)
    
    # 获取回调和日志
    callbacks = get_callbacks(cfg)
    pl_logger = get_logger(cfg)
    
    # 训练配置
    training_cfg = cfg.get("training", {})
    dist_cfg = cfg.get("distributed", {})
    
    # 获取分布式策略
    strategy = get_distributed_strategy(cfg)
    
    # 设备配置
    if dist_cfg.get("enabled", False):
        num_nodes = dist_cfg.get("num_nodes", 1)
        devices = dist_cfg.get("devices", "auto")
        # 如果启用分布式训练，考虑同步 BatchNorm
        if dist_cfg.get("sync_batchnorm", True) and num_nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        num_nodes = 1
        devices = "auto"
    
    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=training_cfg.get("epochs", 100),
        accelerator="auto",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=training_cfg.get("precision", "16-mixed"),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 10),
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        # 分布式训练的额外配置
        use_distributed_sampler=True,  # 自动分配数据到各 GPU
    )
    
    # 打印分布式信息
    if trainer.global_rank == 0:
        logger.info(f"分布式训练配置:")
        logger.info(f"  策略: {strategy}")
        logger.info(f"  节点数: {num_nodes}")
        logger.info(f"  每节点 GPU 数: {trainer.num_devices}")
        logger.info(f"  总 GPU 数: {trainer.world_size}")
    
    # 训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )
    
    return model


def evaluate(
    model: AlphaSTomicsModule,
    test_loader: torch.utils.data.DataLoader,
    cfg: Optional[dict] = None,
) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        cfg: 配置字典
    
    Returns:
        评估指标字典
    """
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
    )
    
    results = trainer.test(model, dataloaders=test_loader)
    return results[0] if results else {}


def load_checkpoint(
    checkpoint_path: str,
    cfg: Optional[dict] = None,
) -> AlphaSTomicsModule:
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点路径
        cfg: 配置字典（如果需要覆盖）
    
    Returns:
        加载的模型
    """
    model = AlphaSTomicsModule.load_from_checkpoint(checkpoint_path)
    
    if cfg is not None:
        # 更新配置（例如更改采样参数）
        model.cfg.update(cfg)
    
    return model
