"""
AlphaSTomics - 双模态空间转录组扩散模型

一个先进的空间转录组学数据建模框架，基于扩散模型和Transformer架构。
支持同时对基因表达量和3D空间坐标进行扩散建模。

主要特性:
- 双模态扩散: 同时支持表达量和位置的扩散过程
- 灵活的采样模式: 表达量→位置, 位置→表达量, 联合生成
- 旋转平移不变性: 位置损失使用距离矩阵，保证几何不变性
- Linear Attention: 高效处理大规模空间转录组数据

使用方法:
    from alphastomics import AlphaSTomicsModule
    from alphastomics.diffusion_model import NoiseModel, DiffusionSampler, DualModalLoss
    from alphastomics.utils import DataHolder
"""

__version__ = "0.1.0"
__author__ = "AlphaSTomics Team"

from alphastomics.attn_model import Model
from alphastomics.diffusion_model import NoiseModel, DualModalLoss, DiffusionSampler
from alphastomics.diffusion_model.train import AlphaSTomicsModule
from alphastomics.utils.dataholder import DataHolder

__all__ = [
    "Model",
    "NoiseModel", 
    "DualModalLoss",
    "DiffusionSampler",
    "AlphaSTomicsModule",
    "DataHolder",
]
