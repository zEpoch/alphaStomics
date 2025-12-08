#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaSTomics Gated Attention + MoE Demo
========================================
使用 Gated Attention 和 MoE 测试整个训练流程

测试内容:
1. 数据生成和 DataLoader 创建
2. 模型初始化（启用 Gated Attention 和 MoE）
3. 训练循环 (forward + backward)
4. 验证循环
5. 采样测试

运行方式:
    python demo_gated_moe.py

作者: AlphaSTomics Team
日期: 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Optional, Tuple

# 导入 AlphaSTomics 模块
from alphastomics.diffusion_model import (
    AlphaSTomicsModule,
    NoiseModel,
    DiffusionSampler,
    MaskGenerator,
    MaskingConfig,
)
from alphastomics.utils.dataholder import DataHolder

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Demo 配置 ====================
DEMO_CONFIG = {
    # 训练模式: "expr_to_pos", "pos_to_expr", "joint"
    "training_mode": "joint",
    
    # 扩散参数
    "diffusion": {
        "diffusion_steps": 100,  # Demo 使用较少步数加快测试
        "diffusion_noise_schedule": "cosine",
        "nu_expression": 1.0,
        "nu_position": 1.0,
    },
    
    # 损失函数
    "loss": {
        "lambda_expression": 1.0,
        "lambda_position": 1.0,
        "use_distance_matrix": True,
    },
    
    # Masked Diffusion (可开启/关闭)
    "masking": {
        "enable": True,  # 启用 Masked Diffusion
        "expression_mask_ratio": 0.3,
        "position_mask_ratio": 0.33,
        "mask_strategy": "random",
        "mask_expression": True,
        "mask_position": True,
        "reconstruction_weight": 0.5,
        "masking_probability": 0.5,
        "progressive_masking": False,
    },
    
    # 训练参数
    "training": {
        "learning_rate": 1e-3,  # Demo 使用较大学习率
        "weight_decay": 1e-5,
        "batch_size": 4,
        "epochs": 5,  # Demo 只训练几个 epoch
    },
    
    # 采样参数
    "sampling": {
        "num_steps": 20,  # Demo 使用较少步数
    },
    
    # 模型架构 (小型模型用于测试)
    "mlp_in_expression_setting": {
        "mlp_in_expression_dims": 64,
        "mlp_out_expression_dims": 128,
    },
    "mlp_in_diffusion_time_setting": {
        "mlp_in_diffusion_time_dims": 32,
        "mlp_out_diffusion_time_dims": 32,
    },
    "PositionMLP_setting": {
        "hidden_dims": 32,
    },
    "TransformerLayer_setting": {
        "num_layers": 2,  # Demo 使用较少层数
        "num_heads": 4,
        "dim_ff_expression": 256,
        "dim_ff_diffusion_time": 64,
        "dropout": 0.1,
        "layer_norm_eps": 1e-6,
        # Gated Attention 参数
        "use_gated_attention": True,
        "gate_type": "headwise",  # 'headwise' / 'elementwise' / 'none'
        "use_qk_norm": True,
        # MoE 参数
        "use_moe": True,
        "num_experts": 4,  # 4 个专家
        "moe_top_k": 2,  # 每次激活 2 个专家
        "moe_load_balance_loss_weight": 0.01,  # 负载均衡权重
    },
    "mlp_out_expression_setting": {
        "hidden_dims": 128,
    },
    "mlp_out_position_norm_setting": {
        "hidden_dims": 64,
    },
}


# ==================== Demo 数据集 ====================
class DemoDataset(Dataset):
    """
    Demo 数据集：生成模拟的空间转录组数据
    
    模拟数据特点:
    - 表达量: 基因表达矩阵 (N_cells, N_genes)
    - 位置: 3D 空间坐标 (N_cells, 3)
    - 支持可变数量的细胞
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_genes: int = 50,
        min_cells: int = 20,
        max_cells: int = 100,
        seed: int = 42,
    ):
        """
        初始化 Demo 数据集
        
        Args:
            num_samples: 样本数量（模拟切片数量）
            num_genes: 基因数量
            min_cells: 每个样本最少细胞数
            max_cells: 每个样本最多细胞数
            seed: 随机种子
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_genes = num_genes
        self.min_cells = min_cells
        self.max_cells = max_cells
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 预生成所有数据
        self.data = []
        for i in range(num_samples):
            sample = self._generate_sample()
            self.data.append(sample)
    
    def _generate_sample(self) -> Dict:
        """
        生成单个模拟样本
        
        模拟策略:
        - 位置: 在 3D 空间中的聚类分布（模拟组织结构）
        - 表达量: 根据位置生成（模拟空间基因表达模式）
        """
        # 随机细胞数量
        num_cells = np.random.randint(self.min_cells, self.max_cells + 1)
        
        # 生成 3D 位置 (模拟组织切片)
        # 使用高斯混合模型生成聚类
        num_clusters = np.random.randint(2, 5)
        cluster_centers = np.random.randn(num_clusters, 3) * 2
        
        positions = []
        cluster_labels = []
        cells_per_cluster = num_cells // num_clusters
        
        for c in range(num_clusters):
            if c < num_clusters - 1:
                n_c = cells_per_cluster
            else:
                # 最后一个 cluster 补足剩余的细胞数
                n_c = num_cells - sum(p.shape[0] for p in positions)
            n_c = max(1, n_c)  # 确保至少有 1 个细胞
            pos_c = cluster_centers[c] + np.random.randn(n_c, 3) * 0.5
            positions.append(pos_c)
            cluster_labels.extend([c] * n_c)
        
        positions = np.vstack(positions)
        cluster_labels = np.array(cluster_labels)
        
        # 确保数组大小匹配
        num_cells = positions.shape[0]
        
        # 中心化位置
        positions = positions - positions.mean(axis=0)
        
        # 生成表达量 (与位置/聚类相关)
        expression = np.zeros((num_cells, self.num_genes))
        
        # 每个聚类有不同的表达模式
        for c in range(num_clusters):
            mask = cluster_labels == c
            # 基础表达
            base_expr = np.random.rand(self.num_genes) * 2
            # 添加噪声
            expression[mask] = base_expr + np.random.randn(mask.sum(), self.num_genes) * 0.3
        
        # 添加空间梯度效应
        for g in range(min(10, self.num_genes)):  # 前10个基因有空间梯度
            gradient = positions[:, g % 3]  # 使用某个坐标轴
            expression[:, g] += gradient * 0.5
        
        # 确保非负
        expression = np.maximum(expression, 0)
        
        return {
            "expression": torch.tensor(expression, dtype=torch.float32),
            "positions": torch.tensor(positions, dtype=torch.float32),
            "num_cells": num_cells,
            "cluster_labels": torch.tensor(cluster_labels, dtype=torch.long),
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_fn(batch: list) -> Dict:
    """
    将不同大小的样本 padding 成统一的 batch
    
    Returns:
        dict with:
            - expression: (B, max_N, G)
            - positions: (B, max_N, 3)
            - node_mask: (B, max_N) - 1 表示有效细胞，0 表示 padding
    """
    batch_size = len(batch)
    num_genes = batch[0]["expression"].shape[1]
    
    # 找到最大细胞数
    max_cells = max(sample["num_cells"] for sample in batch)
    
    # 初始化 padded tensors
    expression = torch.zeros(batch_size, max_cells, num_genes)
    positions = torch.zeros(batch_size, max_cells, 3)
    node_mask = torch.zeros(batch_size, max_cells)
    
    # 填充数据
    for i, sample in enumerate(batch):
        n = sample["num_cells"]
        expression[i, :n, :] = sample["expression"]
        positions[i, :n, :] = sample["positions"]
        node_mask[i, :n] = 1.0
    
    return {
        "expression": expression,
        "positions": positions,
        "node_mask": node_mask,
    }


# ==================== Demo DataModule ====================
class DemoDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for demo data"""
    
    def __init__(
        self,
        num_genes: int = 50,
        batch_size: int = 4,
        num_train: int = 80,
        num_val: int = 10,
        num_test: int = 10,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
    
    def setup(self, stage: Optional[str] = None):
        """创建数据集"""
        self.train_dataset = DemoDataset(
            num_samples=self.num_train,
            num_genes=self.num_genes,
            seed=42,
        )
        self.val_dataset = DemoDataset(
            num_samples=self.num_val,
            num_genes=self.num_genes,
            seed=123,
        )
        self.test_dataset = DemoDataset(
            num_samples=self.num_test,
            num_genes=self.num_genes,
            seed=456,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
        )


# ==================== 主函数 ====================
def test_forward_pass():
    """测试前向传播"""
    logger.info("=" * 60)
    logger.info("测试 1: 前向传播")
    logger.info("=" * 60)
    
    num_genes = 50
    batch_size = 2
    num_cells = 30
    
    # 创建模型
    model = AlphaSTomicsModule(cfg=DEMO_CONFIG, num_genes=num_genes)
    print(model)
    # 创建假数据
    expression = torch.randn(batch_size, num_cells, num_genes)
    positions = torch.randn(batch_size, num_cells, 3)
    diffusion_time = torch.rand(batch_size, 1)
    node_mask = torch.ones(batch_size, num_cells)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(
            expression=expression,
            positions=positions,
            diffusion_time=diffusion_time,
            node_mask=node_mask,
        )
        # 处理返回值：可能是 2 个或 3 个值
        if len(output) == 3:
            pred_expr, pred_pos, moe_aux_loss = output
            logger.info(f"   MoE 辅助损失: {moe_aux_loss.item() if moe_aux_loss is not None else 'N/A'}")
        else:
            pred_expr, pred_pos = output
    
    logger.info(f"✅ 前向传播成功!")
    logger.info(f"   输入 expression: {expression.shape}")
    logger.info(f"   输入 positions: {positions.shape}")
    logger.info(f"   输出 pred_expression: {pred_expr.shape}")
    logger.info(f"   输出 pred_positions: {pred_pos.shape}")
    
    return True


def test_noise_model():
    """测试噪声模型"""
    logger.info("=" * 60)
    logger.info("测试 2: 噪声模型")
    logger.info("=" * 60)
    
    num_genes = 50
    batch_size = 2
    num_cells = 30
    
    # 创建噪声模型
    noise_model = NoiseModel(DEMO_CONFIG["diffusion"])
    
    # 创建原始数据
    data = DataHolder(
        expression=torch.randn(batch_size, num_cells, num_genes),
        positions=torch.randn(batch_size, num_cells, 3),
        node_mask=torch.ones(batch_size, num_cells),
    )
    
    # 应用噪声
    noisy_data, mask_info = noise_model.apply_noise(
        data,
        noise_expression=True,
        noise_position=True,
        masking_module=None,
        apply_masking=False,
    )
    
    logger.info(f"✅ 噪声模型测试成功!")
    logger.info(f"   原始 expression: {data.expression.shape}")
    logger.info(f"   加噪 expression: {noisy_data.noisy_expression.shape}")
    logger.info(f"   扩散时间: {noisy_data.diffusion_time.shape}")
    
    return True


def test_masked_diffusion():
    """测试 Masked Diffusion"""
    logger.info("=" * 60)
    logger.info("测试 3: Masked Diffusion")
    logger.info("=" * 60)
    
    num_genes = 50
    batch_size = 2
    num_cells = 30
    
    # 创建 MaskGenerator（注意：接口只需要 mask 配置参数）
    mask_generator = MaskGenerator(
        expression_mask_ratio=0.3,
        position_mask_ratio=0.33,
        mask_strategy='random',
    )
    
    # 生成测试数据
    expression = torch.randn(batch_size, num_cells, num_genes)
    positions = torch.randn(batch_size, num_cells, 3)
    node_mask = torch.ones(batch_size, num_cells)
    
    # 生成 expression mask
    expr_mask = mask_generator.generate_expression_mask(expression)
    pos_mask = mask_generator.generate_position_mask(positions)
    
    logger.info(f"✅ Masked Diffusion 测试成功!")
    logger.info(f"   Expression mask shape: {expr_mask.shape}")
    logger.info(f"   Position mask shape: {pos_mask.shape}")
    logger.info(f"   Expression mask ratio: {expr_mask.float().mean().item():.2%}")
    logger.info(f"   Position mask ratio: {pos_mask.float().mean().item():.2%}")
    
    # 应用 mask（将 masked 位置设为 0）
    masked_expr = expression.clone()
    masked_expr[expr_mask] = 0
    masked_pos = positions.clone()
    masked_pos[pos_mask] = 0
    
    logger.info(f"   Masked expression: {masked_expr.shape}")
    logger.info(f"   Masked positions: {masked_pos.shape}")
    
    return True


def test_training_step():
    """测试单步训练"""
    logger.info("=" * 60)
    logger.info("测试 4: 训练步骤 (forward + backward)")
    logger.info("=" * 60)
    
    num_genes = 50
    batch_size = 2
    num_cells = 30
    
    # 创建模型
    model = AlphaSTomicsModule(cfg=DEMO_CONFIG, num_genes=num_genes)
    model.train()
    
    # 创建假 batch，转换为 DataHolder 避免 self.log() 警告
    from alphastomics.utils.dataholder import DataHolder
    batch = DataHolder(
        expression=torch.randn(batch_size, num_cells, num_genes),
        positions=torch.randn(batch_size, num_cells, 3),
        node_mask=torch.ones(batch_size, num_cells),
    )
    
    # 手动执行训练步骤（不使用 training_step 以避免 log 警告）
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = model.training_step(batch, batch_idx=0)
    
    logger.info(f"✅ 训练步骤成功!")
    logger.info(f"   Loss: {loss.item():.4f}")
    
    # 反向传播测试
    loss.backward()
    logger.info(f"✅ 反向传播成功!")
    
    return True


def test_full_training():
    """测试完整训练流程"""
    logger.info("=" * 60)
    logger.info("测试 5: 完整训练流程 (PyTorch Lightning)")
    logger.info("=" * 60)
    
    num_genes = 50
    batch_size = 4
    
    # 创建 DataModule
    datamodule = DemoDataModule(
        num_genes=num_genes,
        batch_size=batch_size,
        num_train=20,  # 小数据集快速测试
        num_val=5,
        num_test=5,
    )
    
    # 创建模型
    model = AlphaSTomicsModule(cfg=DEMO_CONFIG, num_genes=num_genes)
    
    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=2,  # 只训练 2 个 epoch
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,  # 禁用日志记录
        enable_checkpointing=False,  # 禁用检查点
        num_sanity_val_steps=1,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.fit(model, datamodule=datamodule)
    
    logger.info(f"✅ 完整训练流程成功!")
    
    return True


def test_sampling():
    """测试采样"""
    logger.info("=" * 60)
    logger.info("测试 6: 扩散采样")
    logger.info("=" * 60)
    
    num_genes = 50
    batch_size = 2
    num_cells = 30
    
    # 创建模型（禁用 masking 以简化测试）
    sampling_config = DEMO_CONFIG.copy()
    sampling_config["masking"] = {"enable": False}
    
    model = AlphaSTomicsModule(cfg=sampling_config, num_genes=num_genes)
    model.eval()
    
    # 创建采样器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 首先测试直接前向传播
    logger.info("测试前向传播 (使用 (B, 1) 时间)...")
    expression = torch.randn(batch_size, num_cells, num_genes).to(device)
    positions = torch.randn(batch_size, num_cells, 3).to(device)
    node_mask = torch.ones(batch_size, num_cells).to(device)
    diffusion_time = torch.ones(batch_size, 1).to(device)  # (B, 1)
    
    with torch.no_grad():
        output = model.model(
            expression_features=expression,
            diffusion_time=diffusion_time,
            position_features=positions,
            node_mask=node_mask,
        )
        # 处理返回值：可能是 2 个或 3 个值
        if len(output) == 3:
            pred_expr, pred_pos, _ = output
        else:
            pred_expr, pred_pos = output
    logger.info(f"  前向传播成功! pred_expr: {pred_expr.shape}, pred_pos: {pred_pos.shape}")
    
    # 创建采样器
    sampler = DiffusionSampler(
        model=model.model,
        noise_model=model.noise_model,
        device=device,
    )
    
    # 执行采样 (expr_to_pos: 从表达量预测位置)
    logger.info("执行采样 (expr_to_pos)...")
    with torch.no_grad():
        sampled_expr, sampled_pos = sampler.sample(
            expression=expression,
            positions=positions,
            node_mask=node_mask,
            mode="expr_to_pos",
            num_steps=10,  # 少量步数快速测试
            verbose=True,
        )
    
    logger.info(f"✅ 采样成功!")
    logger.info(f"   采样得到的 expression: {sampled_expr.shape}")
    logger.info(f"   采样得到的 positions: {sampled_pos.shape}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "=" * 60)
    logger.info("AlphaSTomics Demo 训练测试")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("前向传播", test_forward_pass),
        ("噪声模型", test_noise_model),
        ("Masked Diffusion", test_masked_diffusion),
        ("训练步骤", test_training_step),
        ("完整训练", test_full_training),
        ("扩散采样", test_sampling),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            success = test_fn()
            results[name] = "✅ 通过" if success else "❌ 失败"
        except Exception as e:
            results[name] = f"❌ 错误: {str(e)}"
            logger.error(f"测试 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    for name, result in results.items():
        logger.info(f"  {name}: {result}")
    
    all_passed = all("通过" in r for r in results.values())
    if all_passed:
        logger.info("\n🎉 所有测试通过! 训练流程可以正常运行。")
    else:
        logger.info("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    # 检查是否只运行特定测试
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "forward":
            test_forward_pass()
        elif test_name == "noise":
            test_noise_model()
        elif test_name == "mask":
            test_masked_diffusion()
        elif test_name == "step":
            test_training_step()
        elif test_name == "train":
            test_full_training()
        elif test_name == "sample":
            test_sampling()
        else:
            print(f"未知测试: {test_name}")
            print("可用测试: forward, noise, mask, step, train, sample")
    else:
        # 运行所有测试
        run_all_tests()
