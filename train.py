#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaSTomics Training Script
============================
主训练脚本，支持命令行参数配置

作者: AlphaSTomics Team
日期: 2024
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import logging

from alphastomics.diffusion_model import AlphaSTomicsModule

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='AlphaSTomics Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    # 数据参数
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--num_genes',
        type=int,
        required=True,
        help='Number of genes in the dataset'
    )
    
    # 训练参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--training_mode',
        type=str,
        default=None,
        choices=['joint', 'expr_to_pos', 'pos_to_expr'],
        help='Training mode (overrides config)'
    )
    
    # Gated Attention 参数
    parser.add_argument(
        '--use_gated_attention',
        action='store_true',
        help='Enable Gated Attention'
    )
    parser.add_argument(
        '--no_gated_attention',
        action='store_true',
        help='Disable Gated Attention'
    )
    parser.add_argument(
        '--gate_type',
        type=str,
        default='elementwise',
        choices=['headwise', 'elementwise'],
        help='Gate type for Gated Attention (elementwise: best performance [default], headwise: fewer parameters)'
    )
    
    # MoE 参数
    parser.add_argument(
        '--use_moe',
        action='store_true',
        help='Enable Mixture of Experts'
    )
    parser.add_argument(
        '--no_moe',
        action='store_true',
        help='Disable Mixture of Experts'
    )
    parser.add_argument(
        '--num_experts',
        type=int,
        default=None,
        help='Number of experts in MoE'
    )
    parser.add_argument(
        '--moe_top_k',
        type=int,
        default=None,
        help='Number of experts to activate per token'
    )
    
    # Masked Diffusion 参数
    parser.add_argument(
        '--enable_masking',
        action='store_true',
        help='Enable Masked Diffusion'
    )
    parser.add_argument(
        '--disable_masking',
        action='store_true',
        help='Disable Masked Diffusion'
    )
    parser.add_argument(
        '--expression_mask_ratio',
        type=float,
        default=None,
        help='Expression masking ratio'
    )
    parser.add_argument(
        '--position_mask_ratio',
        type=float,
        default=None,
        help='Position masking ratio'
    )
    
    # 输出和日志
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='alphastomics',
        help='Experiment name'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # PyTorch Lightning 参数
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision training'
    )
    
    return parser.parse_args()


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args: argparse.Namespace):
    """用命令行参数覆盖配置"""
    # 训练参数
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.training_mode is not None:
        config['training_mode'] = args.training_mode
    
    # Gated Attention
    if args.use_gated_attention:
        config['TransformerLayer_setting']['use_gated_attention'] = True
    if args.no_gated_attention:
        config['TransformerLayer_setting']['use_gated_attention'] = False
    if args.gate_type is not None:
        config['TransformerLayer_setting']['gate_type'] = args.gate_type
    
    # MoE
    if args.use_moe:
        config['TransformerLayer_setting']['use_moe'] = True
    if args.no_moe:
        config['TransformerLayer_setting']['use_moe'] = False
    if args.num_experts is not None:
        config['TransformerLayer_setting']['num_experts'] = args.num_experts
    if args.moe_top_k is not None:
        config['TransformerLayer_setting']['moe_top_k'] = args.moe_top_k
    
    # Masked Diffusion
    if args.enable_masking:
        config['masking']['enable'] = True
    if args.disable_masking:
        config['masking']['enable'] = False
    if args.expression_mask_ratio is not None:
        config['masking']['expression_mask_ratio'] = args.expression_mask_ratio
    if args.position_mask_ratio is not None:
        config['masking']['position_mask_ratio'] = args.position_mask_ratio
    
    return config


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 加载配置
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # 用命令行参数覆盖配置
    config = override_config(config, args)
    
    # 打印配置
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Training mode: {config['training_mode']}")
    logger.info(f"  Gated Attention: {config['TransformerLayer_setting'].get('use_gated_attention', False)}")
    if config['TransformerLayer_setting'].get('use_gated_attention'):
        logger.info(f"    - Gate type: {config['TransformerLayer_setting'].get('gate_type', 'headwise')}")
    logger.info(f"  MoE: {config['TransformerLayer_setting'].get('use_moe', False)}")
    if config['TransformerLayer_setting'].get('use_moe'):
        logger.info(f"    - Num experts: {config['TransformerLayer_setting'].get('num_experts', 8)}")
        logger.info(f"    - Top-K: {config['TransformerLayer_setting'].get('moe_top_k', 2)}")
    logger.info(f"  Masked Diffusion: {config['masking'].get('enable', False)}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info("=" * 80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存最终配置
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {config_save_path}")
    
    # 创建数据加载器
    from alphastomics.preprocessing import create_dataloaders
    
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        streaming=False,  # 可以添加 --streaming 参数来启用
    )
    
    # 获取基因数量
    num_genes = metadata.get('n_genes', args.num_genes)
    logger.info(f"Loaded data: {num_genes} genes")
    
    # 创建模型
    logger.info("Creating model...")
    model = AlphaSTomicsModule(
        cfg=config,
        num_genes=num_genes
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    
    # 创建 callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='alphastomics-{epoch:02d}-{val_loss:.4f}',
            monitor='val/loss/total',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # 可选：Early stopping
    if config['training'].get('early_stopping', False):
        callbacks.append(
            EarlyStopping(
                monitor='val/loss/total',
                patience=config['training'].get('early_stopping_patience', 10),
                mode='min',
            )
        )
    
    # 创建 logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs',
    )
    
    # 创建 Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=callbacks,
        logger=tb_logger,
        precision=16 if args.fp16 else 32,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        log_every_n_steps=10,
        val_check_interval=config['training'].get('val_check_interval', 1.0),
    )
    
    # 开始训练
    logger.info("Starting training...")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from
    )
    
    # 测试
    logger.info("Running test...")
    trainer.test(model, dataloaders=test_loader)
    
    logger.info("Training completed!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
