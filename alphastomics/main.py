"""
AlphaSTomics - 双模态空间转录组扩散模型

主入口文件

用法:
    # 预处理数据
    python main.py preprocess --input_dir ./raw_h5ad --output_dir ./processed
    
    # 训练模型
    python main.py train --config config.yaml --data_dir ./processed
    
    # 测试模型
    python main.py test --config config.yaml --checkpoint ./outputs/checkpoints/last.ckpt
    
    # 提取 embedding
    python main.py extract --checkpoint ./outputs/checkpoints/last.ckpt --data_dir ./processed
"""
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path
import logging

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
    """预处理 h5ad 数据"""
    from alphastomics.utils.dataloader import SpatialDataPreprocessor
    
    logger.info("=" * 50)
    logger.info("数据预处理")
    logger.info("=" * 50)
    
    # 加载配置
    cfg = load_config(args.config) if args.config else {}
    preprocess_cfg = cfg.get("data", {}).get("preprocessing", {})
    
    # 初始化预处理器
    preprocessor = SpatialDataPreprocessor(
        gene_list_file=args.gene_list_file or preprocess_cfg.get("gene_list_file"),
        position_key=args.position_key or preprocess_cfg.get("position_key", "ccf"),
        position_is_3d=preprocess_cfg.get("position_is_3d", True),
        z_spacing=args.z_spacing or preprocess_cfg.get("z_spacing", 1.0),
        cell_type_key=preprocess_cfg.get("cell_type_key", "cell_type"),
        scale=preprocess_cfg.get("scale", True),
        normalize_position=preprocess_cfg.get("normalize_position", True),
    )
    
    # 查找 h5ad 文件
    input_path = Path(args.input_dir)
    h5ad_files = sorted(input_path.glob("*.h5ad"))
    
    if not h5ad_files:
        raise ValueError(f"在 {args.input_dir} 中未找到 h5ad 文件")
    
    logger.info(f"找到 {len(h5ad_files)} 个 h5ad 文件")
    
    # 预处理并保存
    metadata = preprocessor.preprocess_and_save(
        h5ad_files=[str(f) for f in h5ad_files],
        output_dir=args.output_dir,
    )
    
    logger.info("预处理完成!")
    logger.info(f"  基因数: {metadata['n_genes']}")
    logger.info(f"  切片数: {metadata['n_slices']}")
    logger.info(f"  总细胞数: {metadata['total_cells']}")


def cmd_train(args):
    """训练模型"""
    from alphastomics.utils.dataloader import create_dataloaders
    from alphastomics.diffusion_model.train import (
        train, 
        create_model_from_config,
        get_callbacks,
        get_logger,
    )
    
    logger.info("=" * 50)
    logger.info("AlphaSTomics - 模型训练")
    logger.info("=" * 50)
    
    # 加载配置
    cfg = load_config(args.config)
    
    logger.info(f"训练模式: {cfg.get('training_mode', 'joint')}")
    logger.info(f"扩散步数: {cfg.get('diffusion', {}).get('diffusion_steps', 1000)}")
    
    # 创建数据加载器
    data_dir = args.data_dir or cfg.get("data", {}).get("processed_data_dir", "./data/processed")
    loading_cfg = cfg.get("data", {}).get("loading", {})
    training_cfg = cfg.get("training", {})
    
    # 选择训练模式：slice（切片级）或 cell（细胞级）
    data_mode = args.data_mode or loading_cfg.get("mode", "slice")
    batch_size = training_cfg.get("batch_size", 1 if data_mode == "slice" else 256)
    
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        mode=data_mode,
        batch_size=batch_size,
        num_workers=loading_cfg.get("num_workers", 4),
    )
    
    num_genes = metadata.get("n_genes", len(metadata.get("selected_genes", [])))
    logger.info(f"数据模式: {data_mode}")
    logger.info(f"基因数量: {num_genes}")
    logger.info(f"训练样本: {len(train_loader.dataset)}")
    logger.info(f"验证样本: {len(val_loader.dataset)}")
    
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
    from alphastomics.utils.dataloader import create_dataloaders
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
    data_mode = args.data_mode or loading_cfg.get("mode", "slice")
    
    _, _, test_loader, metadata = create_dataloaders(
        data_dir=data_dir,
        mode=data_mode,
        batch_size=1 if data_mode == "slice" else 256,
        num_workers=loading_cfg.get("num_workers", 4),
    )
    
    # 评估
    results = evaluate(model, test_loader, cfg)
    logger.info("测试完成!")


def cmd_sample(args):
    """采样生成"""
    from alphastomics.diffusion_model.train import load_checkpoint
    from alphastomics.diffusion_model.sample import DiffusionSampler
    from alphastomics.utils.dataloader import SpatialDataPreprocessor
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
    from alphastomics.utils.dataloader import create_dataloaders
    
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
        max_cells=loading_cfg.get("max_cells", 5000),
        max_cells_per_batch=loading_cfg.get("max_cells_per_batch", 50000),
        num_workers=loading_cfg.get("num_workers", 4),
        augment_train=False,
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AlphaSTomics - 双模态空间转录组扩散模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理 h5ad 数据")
    preprocess_parser.add_argument("--input_dir", type=str, required=True, help="输入 h5ad 文件目录")
    preprocess_parser.add_argument("--output_dir", type=str, required=True, help="输出预处理数据目录")
    preprocess_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    preprocess_parser.add_argument("--gene_list_file", type=str, default=None, help="固定基因列表文件")
    preprocess_parser.add_argument("--position_key", type=str, default=None, help="坐标 key（默认 ccf）")
    preprocess_parser.add_argument("--z_spacing", type=float, default=None, help="切片间 z 间距（仅2D模式）")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    train_parser.add_argument("--data_dir", type=str, default=None, help="预处理数据目录")
    train_parser.add_argument("--data_mode", type=str, choices=["slice", "cell"], default=None, 
                              help="数据模式: slice(切片级) 或 cell(细胞级)")
    train_parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    train_parser.add_argument("--test_after_train", action="store_true", help="训练后运行测试")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="测试模型")
    test_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    test_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    test_parser.add_argument("--data_dir", type=str, default=None, help="预处理数据目录")
    test_parser.add_argument("--data_mode", type=str, choices=["slice", "cell"], default=None,
                              help="数据模式: slice(切片级) 或 cell(细胞级)")
    
    # 采样命令
    sample_parser = subparsers.add_parser("sample", help="采样生成")
    sample_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    sample_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    sample_parser.add_argument("--input_data", type=str, required=True, help="输入数据路径 (.pt)")
    sample_parser.add_argument("--output", type=str, default=None, help="输出路径")
    sample_parser.add_argument("--sample_mode", type=str, choices=["expr_to_pos", "pos_to_expr", "joint"])
    sample_parser.add_argument("--num_steps", type=int, default=None, help="采样步数")
    
    # 提取 embedding 命令
    extract_parser = subparsers.add_parser("extract", help="提取 embedding")
    extract_parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    extract_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    extract_parser.add_argument("--data_dir", type=str, default=None, help="预处理数据目录")
    extract_parser.add_argument("--output", type=str, default=None, help="输出路径")
    extract_parser.add_argument("--extract_mode", type=str, choices=["encoder", "transformer", "all_layers"])
    extract_parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    extract_parser.add_argument("--format", type=str, choices=["npz", "pt", "h5"], default="npz")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
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


if __name__ == "__main__":
    main()

