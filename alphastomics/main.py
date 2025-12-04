"""
AlphaSTomics - 双模态空间转录组扩散模型

主入口文件

用法:
    # 预处理（两阶段，输出 Parquet 格式）
    python main.py preprocess --input_dir ./raw_h5ad --output_dir ./dataset
    
    # 训练模型
    python main.py train --config config.yaml --data_dir ./dataset
    python main.py train --config config.yaml --data_dir ./dataset --streaming  # 流式加载大数据集
    
    # 测试模型
    python main.py test --config config.yaml --checkpoint ./outputs/checkpoints/last.ckpt
    
    # 采样生成
    python main.py sample --checkpoint ./outputs/checkpoints/last.ckpt --input_data ./input.pt
    
    # 提取 embedding
    python main.py extract --checkpoint ./outputs/checkpoints/last.ckpt --data_dir ./dataset

随机种子控制:
    所有命令都支持 --seed 参数来设置全局随机种子，确保实验可复现：
    python main.py train --config config.yaml --seed 42
    python main.py preprocess --input_dir ./raw --output_dir ./out --seed 42
"""
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path
import logging

from alphastomics.utils.seed import set_seed, get_seed_info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cmd_preprocess(args):
    """预处理 h5ad 数据（两阶段预处理）
    
    注意: 这是统一的预处理入口，内部调用两阶段预处理流程
    """
    from alphastomics.preprocessing import Stage1Preprocessor, Stage2BatchGenerator
    
    logger.info("=" * 50)
    logger.info("数据预处理 (Parquet 格式)")
    logger.info("=" * 50)
    
    # 加载配置
    cfg = load_config(args.config) if args.config else {}
    preprocess_cfg = cfg.get("data", {}).get("preprocessing", {})
    
    output_dir = Path(args.output_dir)
    intermediate_dir = output_dir / '_intermediate_slices'
    
    # 判断是否跳过 log normalize 和 scale
    do_log_normalize = not getattr(args, 'skip_log_normalize', False)
    do_scale = not getattr(args, 'skip_scale', False)
    
    # ===== 阶段一：切片预处理 =====
    logger.info("")
    if do_log_normalize:
        logger.info("阶段一：切片预处理 (log normalize + scale)")
    else:
        logger.info("阶段一：切片预处理 (跳过 log normalize)")
    logger.info("-" * 40)
    
    preprocessor = Stage1Preprocessor(
        gene_list_file=args.gene_list_file or preprocess_cfg.get("gene_list_file"),
        position_key=args.position_key or preprocess_cfg.get("position_key", "spatial"),
        position_is_3d=getattr(args, 'position_is_3d', False) or preprocess_cfg.get("position_is_3d", False),
        z_spacing=args.z_spacing or preprocess_cfg.get("z_spacing", 10.0),
        cell_type_key=getattr(args, 'cell_type_key', 'cell_type') or preprocess_cfg.get("cell_type_key", "cell_type"),
        log_normalize=do_log_normalize,
        scale=do_scale,
        normalize_position=preprocess_cfg.get("normalize_position", False),
    )
    
    # 查找 h5ad 文件
    input_path = Path(args.input_dir)
    h5ad_files = sorted(input_path.glob("*.h5ad"))
    
    if not h5ad_files:
        raise ValueError(f"在 {args.input_dir} 中未找到 h5ad 文件")
    
    logger.info(f"找到 {len(h5ad_files)} 个 h5ad 文件")
    logger.info(f"Log normalize: {do_log_normalize}, Scale: {do_scale}")
    
    stage1_metadata = preprocessor.process(args.input_dir, str(intermediate_dir))
    
    # ===== 阶段二：生成数据集 =====
    logger.info("")
    logger.info("阶段二：生成数据集 (无放回采样)")
    logger.info("-" * 40)
    
    generator = Stage2BatchGenerator(
        train_ratio=getattr(args, 'train_ratio', 0.8),
        val_ratio=getattr(args, 'val_ratio', 0.1),
        seed=args.seed,
        max_shard_size=getattr(args, 'max_shard_size', 100000),
    )
    
    stats = generator.process(
        str(intermediate_dir), 
        str(output_dir), 
        output_format="parquet"
    )
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("预处理完成!")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  训练集: {stats['train']['n_cells']:,} 细胞")
    logger.info(f"  验证集: {stats['validation']['n_cells']:,} 细胞")
    logger.info(f"  测试集: {stats['test']['n_cells']:,} 细胞")
    logger.info("=" * 50)


def cmd_train(args):
    """训练模型"""
    from alphastomics.diffusion_model.train import (
        train, 
        create_model_from_config,
        get_callbacks,
        get_logger,
    )
    from alphastomics.preprocessing import create_dataloaders
    
    logger.info("=" * 50)
    logger.info("AlphaSTomics - 模型训练")
    logger.info("=" * 50)
    
    # 加载配置
    cfg = load_config(args.config)
    
    logger.info(f"训练模式: {cfg.get('training_mode', 'joint')}")
    logger.info(f"扩散步数: {cfg.get('diffusion', {}).get('diffusion_steps', 1000)}")
    
    # 数据配置
    data_dir = args.data_dir or cfg.get("data", {}).get("processed_data_dir", "./data/processed")
    loading_cfg = cfg.get("data", {}).get("loading", {})
    training_cfg = cfg.get("training", {})
    
    batch_size = args.batch_size or training_cfg.get("batch_size", 1024)
    
    # 创建数据加载器 (Parquet 格式)
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=loading_cfg.get("num_workers", 4),
        streaming=args.streaming,
        buffer_size=args.buffer_size or 10000,
    )
    
    num_genes = metadata.get('n_genes', metadata.get('features', {}).get('expression', {}).get('shape', [3000])[0])
    
    logger.info(f"基因数量: {num_genes}")
    
    # 训练
    model = train(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        num_genes=num_genes,
        resume_from=args.resume,
    )
    
    logger.info("训练完成!")
    
    # 测试
    if args.test_after_train:
        logger.info("运行测试...")
        from alphastomics.diffusion_model.train import evaluate
        results = evaluate(model, test_loader, cfg)
        logger.info(f"测试结果: {results}")


def cmd_test(args):
    """测试模型"""
    from alphastomics.preprocessing import create_dataloaders
    from alphastomics.diffusion_model.train import load_checkpoint, evaluate
    from alphastomics.utils.metrics import print_metrics
    
    logger.info("=" * 50)
    logger.info("AlphaSTomics - 模型测试")
    logger.info("=" * 50)
    
    if args.checkpoint is None:
        raise ValueError("测试模式需要提供 --checkpoint 参数")
    
    # 加载配置
    cfg = load_config(args.config) if args.config else {}
    
    # 加载模型
    logger.info(f"从检查点加载模型: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, cfg)
    
    # 创建测试数据加载器
    data_dir = args.data_dir or cfg.get("data", {}).get("processed_data_dir", "./data/processed")
    loading_cfg = cfg.get("data", {}).get("loading", {})
    
    _, _, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size or 1024,
        num_workers=loading_cfg.get("num_workers", 4),
        streaming=False,  # 测试时不使用流式加载
    )
    
    # 评估
    results = evaluate(model, test_loader, cfg)
    logger.info("测试完成!")


def cmd_sample(args):
    """采样生成"""
    from alphastomics.diffusion_model.train import load_checkpoint
    from alphastomics.diffusion_model.sample import DiffusionSampler
    import numpy as np
    
    logger.info("=" * 50)
    logger.info("AlphaSTomics - 采样生成")
    logger.info("=" * 50)
    
    if args.checkpoint is None:
        raise ValueError("采样模式需要提供 --checkpoint 参数")
    
    # 加载配置
    cfg = load_config(args.config) if args.config else {}
    
    # 加载模型
    logger.info(f"从检查点加载模型: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, cfg)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建采样器
    sampler = DiffusionSampler(
        model=model.model,
        noise_model=model.noise_model,
        device=device
    )
    
    # 加载输入数据
    if args.input_data:
        data = torch.load(args.input_data)
        expression = data.get("expression")
        positions = data.get("positions")
        node_mask = data.get("node_mask")
    else:
        raise ValueError("采样模式需要提供 --input_data 参数")
    
    # 采样
    sampling_cfg = cfg.get("sampling", {})
    num_steps = args.num_steps or sampling_cfg.get("num_steps", 100)
    mode = args.sample_mode or sampling_cfg.get("mode", "joint")
    
    logger.info(f"采样模式: {mode}")
    logger.info(f"采样步数: {num_steps}")
    
    pred_expression, pred_positions = sampler.sample(
        expression=expression.to(device) if expression is not None else None,
        positions=positions.to(device) if positions is not None else None,
        node_mask=node_mask.to(device) if node_mask is not None else None,
        mode=mode,
        num_steps=num_steps,
        verbose=True,
    )
    
    # 保存结果
    output_path = args.output or "sampled_results.pt"
    torch.save({
        "pred_expression": pred_expression.cpu(),
        "pred_positions": pred_positions.cpu(),
        "node_mask": node_mask,
    }, output_path)
    
    logger.info(f"采样结果已保存到 {output_path}")


def cmd_extract(args):
    """提取 embedding"""
    from alphastomics.diffusion_model.train import load_checkpoint
    from alphastomics.utils.embedding_extractor import EmbeddingExtractor
    from alphastomics.preprocessing import create_dataloaders
    
    logger.info("=" * 50)
    logger.info("AlphaSTomics - Embedding 提取")
    logger.info("=" * 50)
    
    if args.checkpoint is None:
        raise ValueError("提取模式需要提供 --checkpoint 参数")
    
    # 加载配置
    cfg = load_config(args.config) if args.config else {}
    
    # 加载模型
    logger.info(f"从检查点加载模型: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建提取器
    extractor = EmbeddingExtractor(model.model, device=device)
    
    # 创建数据加载器
    data_dir = args.data_dir or cfg.get("data", {}).get("processed_data_dir", "./data/processed")
    loading_cfg = cfg.get("data", {}).get("loading", {})
    
    _, _, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size or 1024,
        num_workers=loading_cfg.get("num_workers", 4),
        streaming=False,  # 提取时不使用流式加载
    )
    
    # 提取 embedding
    logger.info("提取 embedding...")
    embeddings = extractor.extract_batch(
        dataloader=test_loader,
        mode=args.extract_mode or "transformer",
        max_samples=args.max_samples,
        verbose=True,
    )
    
    # 保存
    output_path = args.output or "embeddings.npz"
    extractor.save_embeddings(embeddings, output_path, format=args.format or "npz")
    
    logger.info(f"Embedding 已保存到 {output_path}")
    logger.info(f"Expression embedding shape: {embeddings['expression_embeddings'].shape}")
    logger.info(f"Position embedding shape: {embeddings['position_embeddings'].shape}")


def cmd_upload(args):
    """上传数据集到 HuggingFace Hub"""
    from alphastomics.preprocessing.upload_to_hf import upload_dataset_to_hf
    
    logger.info("=" * 50)
    logger.info("AlphaSTomics - 上传到 HuggingFace Hub")
    logger.info("=" * 50)
    
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"目标仓库: {args.repo_id}")
    logger.info(f"私有仓库: {args.private}")
    
    upload_dataset_to_hf(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token
    )
    
    logger.info("上传完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AlphaSTomics - 双模态空间转录组扩散模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ==================== 全局参数 ====================
    parser.add_argument("--seed", type=int, default=42, 
                        help="全局随机种子，用于确保实验可复现 (默认: 42)")
    parser.add_argument("--deterministic", action="store_true",
                        help="启用确定性模式（可能降低性能，但保证完全复现）")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ==================== 预处理命令 ====================
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理 h5ad 数据（两阶段，Parquet 输出）")
    preprocess_parser.add_argument("--input_dir", "-i", type=str, required=True, help="输入 h5ad 文件目录")
    preprocess_parser.add_argument("--output_dir", "-o", type=str, required=True, help="输出数据目录")
    preprocess_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    preprocess_parser.add_argument("--gene_list_file", "-g", type=str, default=None, help="固定基因列表文件")
    preprocess_parser.add_argument("--position_key", type=str, default="spatial", help="坐标 key")
    preprocess_parser.add_argument("--position_is_3d", action="store_true", help="坐标是否为 3D")
    preprocess_parser.add_argument("--z_spacing", type=float, default=10.0, help="切片间 z 间距")
    preprocess_parser.add_argument("--cell_type_key", type=str, default="cell_type", help="细胞类型 key")
    preprocess_parser.add_argument("--skip-log-normalize", action="store_true", dest="skip_log_normalize",
                                   help="跳过 log normalize（数据已预处理时使用）")
    preprocess_parser.add_argument("--skip-scale", action="store_true", dest="skip_scale",
                                   help="跳过 scale（z-score 标准化）")
    preprocess_parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    preprocess_parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    preprocess_parser.add_argument("--max_shard_size", type=int, default=100000, help="每个分片最大细胞数")
    
    # ==================== 训练命令 ====================
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    train_parser.add_argument("--data_dir", type=str, default=None, help="预处理数据目录")
    train_parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    train_parser.add_argument("--streaming", action="store_true", help="流式加载数据（大数据集推荐）")
    train_parser.add_argument("--buffer_size", type=int, default=None, help="流式加载的 shuffle buffer 大小")
    train_parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    train_parser.add_argument("--test_after_train", action="store_true", help="训练后运行测试")
    
    # ==================== 测试命令 ====================
    test_parser = subparsers.add_parser("test", help="测试模型")
    test_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    test_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    test_parser.add_argument("--data_dir", type=str, default=None, help="预处理数据目录")
    test_parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    
    # ==================== 采样命令 ====================
    sample_parser = subparsers.add_parser("sample", help="采样生成")
    sample_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    sample_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    sample_parser.add_argument("--input_data", type=str, required=True, help="输入数据路径 (.pt)")
    sample_parser.add_argument("--output", type=str, default=None, help="输出路径")
    sample_parser.add_argument("--sample_mode", type=str, choices=["expr_to_pos", "pos_to_expr", "joint"])
    sample_parser.add_argument("--num_steps", type=int, default=None, help="采样步数")
    
    # ==================== 提取 embedding 命令 ====================
    extract_parser = subparsers.add_parser("extract", help="提取 embedding")
    extract_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    extract_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    extract_parser.add_argument("--data_dir", type=str, default=None, help="预处理数据目录")
    extract_parser.add_argument("--output", type=str, default=None, help="输出路径")
    extract_parser.add_argument("--extract_mode", type=str, choices=["encoder", "transformer", "all_layers"])
    extract_parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    extract_parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    extract_parser.add_argument("--format", type=str, choices=["npz", "pt", "h5"], default="npz")
    
    # ==================== 上传到 HuggingFace ====================
    upload_parser = subparsers.add_parser("upload", help="上传数据集到 HuggingFace Hub")
    upload_parser.add_argument("--data_dir", "-d", type=str, required=True, help="数据目录")
    upload_parser.add_argument("--repo_id", "-r", type=str, required=True, help="HuggingFace 仓库 ID")
    upload_parser.add_argument("--private", action="store_true", help="设为私有仓库")
    upload_parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # ==================== 设置全局随机种子 ====================
    set_seed(
        seed=args.seed,
        deterministic=args.deterministic,
        benchmark=not args.deterministic,  # 非确定性模式下启用 benchmark 提升性能
    )
    logger.info(f"随机种子设置信息: {get_seed_info()}")
    
    # 执行命令
    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "sample":
        cmd_sample(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "upload":
        cmd_upload(args)


if __name__ == "__main__":
    main()