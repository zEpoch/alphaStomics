"""
随机种子设置模块

提供统一的随机种子设置，确保实验可复现

覆盖的随机源:
- Python random 模块
- NumPy random
- PyTorch (CPU)
- PyTorch CUDA (GPU)
- PyTorch Lightning
- cuDNN (确定性算法)

用法:
    from alphastomics.utils.seed import set_seed, get_seed_info

    # 设置全局随机种子
    set_seed(42)

    # 获取当前随机种子设置信息
    info = get_seed_info()
"""
import os
import random
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 全局默认种子
DEFAULT_SEED = 42

# 存储当前设置的种子
_current_seed: Optional[int] = None


def set_seed(
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    benchmark: bool = False,
    warn_only: bool = False,
) -> None:
    """
    设置全局随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性算法（可能降低性能，但保证复现性）
        benchmark: 是否启用 cuDNN benchmark（True 可能提高性能但降低复现性）
        warn_only: 如果为 True，则只警告而不强制确定性模式
    
    Note:
        - deterministic=True 会强制使用确定性 CUDA 算法，可能降低性能 10-20%
        - benchmark=False 禁用 cuDNN 自动优化，保证相同输入产生相同输出
        - 在分布式训练时，需要在每个进程中调用此函数
    
    Example:
        # 最严格的可复现设置
        set_seed(42, deterministic=True, benchmark=False)
        
        # 性能优先（可能有微小差异）
        set_seed(42, deterministic=False, benchmark=True)
    """
    global _current_seed
    _current_seed = seed
    
    # 1. Python random
    random.seed(seed)
    
    # 2. 环境变量 (影响某些库的行为)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    
    # 5. PyTorch CUDA (所有 GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU
    
    # 6. cuDNN 设置
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
    
    # 7. PyTorch 确定性算法 (PyTorch >= 1.8)
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        except Exception as e:
            logger.warning(f"无法设置确定性算法: {e}")
    
    # 8. PyTorch Lightning (如果可用)
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch Lightning seed_everything 失败: {e}")
    
    logger.info(f"随机种子已设置: seed={seed}, deterministic={deterministic}, benchmark={benchmark}")


def get_seed() -> Optional[int]:
    """获取当前设置的随机种子"""
    return _current_seed


def get_seed_info() -> Dict[str, Any]:
    """
    获取当前随机种子设置的详细信息
    
    Returns:
        包含各组件随机状态信息的字典
    """
    info = {
        'seed': _current_seed,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not set'),
        'numpy_seed': 'set' if _current_seed else 'not set',
        'torch_seed': torch.initial_seed() if hasattr(torch, 'initial_seed') else 'unknown',
    }
    
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_device_count'] = torch.cuda.device_count()
    else:
        info['cuda_available'] = False
    
    if torch.backends.cudnn.is_available():
        info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    
    return info


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker 初始化函数
    
    在多进程数据加载时，确保每个 worker 有不同但可复现的随机状态
    
    用法:
        from alphastomics.utils.seed import worker_init_fn
        
        loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=worker_init_fn
        )
    """
    # 获取全局种子
    seed = _current_seed if _current_seed is not None else DEFAULT_SEED
    
    # 每个 worker 使用 (全局种子 + worker_id) 作为种子
    worker_seed = seed + worker_id
    
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    创建一个具有指定种子的 PyTorch Generator
    
    用于需要独立随机状态的场景，如 DataLoader 的 generator 参数
    
    Args:
        seed: 随机种子，如果为 None 则使用全局种子
    
    Returns:
        配置好的 torch.Generator
    
    Example:
        from alphastomics.utils.seed import get_generator
        
        loader = DataLoader(
            dataset,
            shuffle=True,
            generator=get_generator(42)
        )
    """
    if seed is None:
        seed = _current_seed if _current_seed is not None else DEFAULT_SEED
    
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class SeedContext:
    """
    随机种子上下文管理器
    
    在特定代码块中临时设置不同的随机种子
    
    Example:
        # 使用不同种子进行数据增强
        with SeedContext(123):
            augmented_data = augment(data)
        
        # 退出后恢复原来的随机状态
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.saved_state = {}
    
    def __enter__(self):
        # 保存当前状态
        self.saved_state['random'] = random.getstate()
        self.saved_state['numpy'] = np.random.get_state()
        self.saved_state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            self.saved_state['cuda'] = torch.cuda.get_rng_state_all()
        
        # 设置新种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        return self
    
    def __exit__(self, *args):
        # 恢复之前的状态
        random.setstate(self.saved_state['random'])
        np.random.set_state(self.saved_state['numpy'])
        torch.set_rng_state(self.saved_state['torch'])
        if torch.cuda.is_available() and 'cuda' in self.saved_state:
            torch.cuda.set_rng_state_all(self.saved_state['cuda'])
