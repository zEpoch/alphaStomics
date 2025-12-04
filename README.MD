# AlphaSTomics - 双模态空间转录组扩散模型

<p align="center">
<b>同时对基因表达量和3D空间坐标进行扩散建模</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
<img src="https://img.shields.io/badge/Lightning-2.0+-purple.svg" alt="Lightning">
<img src="https://img.shields.io/badge/HuggingFace-Compatible-yellow.svg" alt="HuggingFace">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 🎯 项目简介

AlphaSTomics 是一个先进的空间转录组学数据建模框架，基于扩散模型和 Transformer 架构。支持**双模态扩散**，可以同时对基因表达量和空间坐标进行生成与重建。

## ✨ 核心特性

- 🧬 **双模态生成**：表达量 + 3D 坐标联合扩散
- 🎭 **Masked Diffusion**：特征级 masking，提升重建质量
- 🔄 **训练模式**：\`expr_to_pos\` / \`pos_to_expr\` / \`joint\`
- 📐 **旋转不变损失**：位置使用距离矩阵 MSE
- ⚡ **高效注意力**：Linear Attention，近 O(n) 复杂度
- 🚀 **分布式训练**：支持 DDP / FSDP / DeepSpeed
- 📦 **HuggingFace 兼容**：Parquet 格式，支持流式加载

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/zEpoch/alphaStomics.git
cd alphaStomics

# 创建环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install --upgrade pip
pip install torch pytorch-lightning pyyaml wandb scanpy anndata pyarrow
pip install -e .

# 验证安装
python -c "from alphastomics.diffusion_model import MaskedDiffusionModule; print('OK')"
```

### 完整工作流程

```bash
# 1. 预处理数据（h5ad → parquet）
python -m alphastomics.main preprocess2 \\
    --input_dir ./raw_h5ad \\
    --output_dir ./processed_data \\
    --gene_list_file ./gene_list.txt \\
    --train_ratio 0.8 \\
    --val_ratio 0.1

# 2. 训练模型
python -m alphastomics.main train \\
    --config ./config.yaml \\
    --data_dir ./processed_data \\
    --batch_size 64

# 3. 测试模型
python -m alphastomics.main test \\
    --checkpoint ./checkpoints/best.ckpt \\
    --data_dir ./processed_data

# 4. 采样生成
python -m alphastomics.main sample \\
    --checkpoint ./checkpoints/best.ckpt \\
    --input_data ./test_input.pt \\
    --sample_mode joint

# 5. 提取 embedding
python -m alphastomics.main extract \\
    --checkpoint ./checkpoints/best.ckpt \\
    --data_dir ./processed_data \\
    --output ./embeddings.npz
```

---

## 📁 项目结构

```plaintext
alphaStomics/
├── alphastomics/
│   ├── main.py                 # 统一 CLI 入口
│   ├── diffusion_model/        # 扩散模型核心
│   │   ├── module.py           # PyTorch Lightning 模块
│   │   ├── noise.py            # 噪声模型（cosine schedule）
│   │   ├── loss.py             # 双模态损失函数
│   │   ├── sample.py           # 采样器
│   │   └── train.py            # 训练逻辑
│   ├── attn_model/             # 注意力模型
│   │   ├── model.py            # 主模型
│   │   ├── transformer.py      # Transformer 编码器
│   │   ├── self_attention.py   # 多头注意力
│   │   └── layers.py           # 基础层
│   ├── preprocessing/          # 数据预处理
│   │   ├── preprocess.py       # 两阶段预处理器
│   │   ├── batch_loader.py     # 数据加载器
│   │   └── upload_to_hf.py     # HuggingFace 上传
│   └── utils/                  # 工具函数
│       ├── dataloader.py       # 旧版数据加载
│       ├── dataholder.py       # 数据容器
│       └── embedding_extractor.py
├── config.yaml                 # 默认配置
├── docs/                       # 文档
└── README.MD
```

---

## 📋 命令行接口

### 预处理命令

#### `preprocess2`（推荐）- 两阶段预处理

```bash
python -m alphastomics.main preprocess2 \\
    --input_dir ./raw_h5ad \\        # 输入 h5ad 文件目录
    --output_dir ./processed \\      # 输出目录
    --gene_list_file genes.txt \\    # 固定基因列表（可选）
    --position_key spatial \\        # 坐标字段名
    --position_is_3d \\              # 坐标是否为 3D
    --z_spacing 10.0 \\              # 切片间 z 间距
    --cell_type_key cell_type \\     # 细胞类型字段名
    --train_ratio 0.8 \\             # 训练集比例
    --val_ratio 0.1 \\               # 验证集比例
    --format parquet \\              # 输出格式：parquet 或 pickle
    --max_shard_size 100000 \\       # 每个分片最大细胞数
    --seed 42                        # 随机种子
```

**两阶段处理流程**：
1. **Stage 1**：对每个切片进行 log normalize + scale，保存中间格式
2. **Stage 2**：跨切片随机采样，生成 train/val/test 分割，序列化为 Parquet

**特点**：
- 内存友好：不需要一次性加载所有数据
- 无放回采样：每个细胞恰好出现一次
- 细胞级分割：同一切片可能被分到不同集合

### 训练命令

```bash
python -m alphastomics.main train \\
    --config ./config.yaml \\        # 配置文件
    --data_dir ./processed \\        # 数据目录
    --data_format auto \\            # 数据格式：auto/parquet/pickle
    --batch_size 64 \\               # 批次大小
    --streaming \\                   # 流式加载（大数据集推荐）
    --buffer_size 50000 \\           # shuffle buffer 大小
    --resume ./checkpoint.ckpt \\    # 恢复训练
    --test_after_train               # 训练后自动测试
```

### 其他命令

```bash
# 测试
python -m alphastomics.main test --checkpoint model.ckpt --data_dir ./data

# 采样
python -m alphastomics.main sample --checkpoint model.ckpt --input_data input.pt --sample_mode joint

# 提取 embedding
python -m alphastomics.main extract --checkpoint model.ckpt --data_dir ./data --output embeddings.npz

# 上传到 HuggingFace
python -m alphastomics.main upload --data_dir ./processed --repo_id org/dataset-name --private
```

---

## 📊 输入数据格式

### h5ad 文件要求

```python
import scanpy as sc
adata = sc.read_h5ad("slice.h5ad")

# 必需字段
adata.X                    # 基因表达矩阵 (n_cells, n_genes)
adata.obsm['spatial']      # 空间坐标 (n_cells, 2) 或 (n_cells, 3)

# 可选字段
adata.obs['cell_type']     # 细胞类型标签
adata.var_names            # 基因名称
```

### 预处理输出格式（Parquet）

```
processed_data/
├── train/
│   ├── shard_0000.parquet
│   ├── shard_0001.parquet
│   └── ...
├── val/
│   └── shard_0000.parquet
├── test/
│   └── shard_0000.parquet
└── metadata.json          # 基因列表、统计信息
```

每个 Parquet 文件包含：
- `expression`: 归一化后的基因表达向量
- `position`: 标准化后的 3D 坐标
- `cell_type`: 细胞类型（可选）
- `slice_id`: 来源切片 ID

---

## ⚙️ 配置文件

```yaml
# config.yaml
model:
  num_genes: 20000           # 基因数量
  d_model: 256               # 模型维度
  n_heads: 8                 # 注意力头数
  n_layers: 6                # Transformer 层数
  dropout: 0.1

diffusion:
  num_timesteps: 1000        # 扩散步数
  schedule: cosine           # 噪声调度
  prediction_type: x0        # 预测类型

training:
  mode: joint                # 训练模式：joint/expr_to_pos/pos_to_expr
  batch_size: 64
  learning_rate: 1e-4
  max_epochs: 100

masking:
  enable: true               # 启用 Masked Diffusion
  expression_mask_ratio: 0.4
  position_mask_ratio: 0.3

data:
  processed_data_dir: ./processed
  loading:
    max_cells: 5000          # 每批最大细胞数
    num_workers: 4
```

---

## 🔬 核心功能详解

### 功能一：双模态生成

同时建模基因表达 `(B, N, G)` 和 3D 坐标 `(B, N, 3)`。

```python
from alphastomics.diffusion_model import DiffusionSampler

sampler = DiffusionSampler(model, noise_model)
expression, positions = sampler.sample(expr, pos, mask, mode='joint')
```

### 功能二：Masked Diffusion (MDT)

对每个细胞内部的特征维度进行 mask，重建损失仅在 mask 上计算。

```python
cfg['masking'] = {
    'enable': True, 
    'expression_mask_ratio': 0.4,
    'position_mask_ratio': 0.3
}
model = AlphaSTomicsModule(cfg, num_genes)
```

### 功能三：训练模式

- `expr_to_pos`: 表达量 → 位置
- `pos_to_expr`: 位置 → 表达量
- `joint`: 联合扩散

### 功能四：损失设计

- **表达量损失**: MSE
- **位置损失**: 距离矩阵 MSE（旋转/平移不变）
- **Masked Reconstruction**: 可选加权

```python
loss = DualModalLoss(
    lambda_expression=1.0, 
    lambda_position=1.0, 
    lambda_masked_reconstruction=0.5
)
```

### 功能五：流式数据加载

支持 HuggingFace Datasets 风格的流式加载，适用于大规模数据集。

```python
from alphastomics.preprocessing.batch_loader import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="./processed",
    batch_size=64,
    streaming=True,       # 启用流式加载
    buffer_size=50000     # shuffle buffer 大小
)
```

---

## 📊 数学原理

### 前向扩散过程
 
对于时间步 $t \in [1, T]$：

$$z_t = \alpha_t \cdot x_0 + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\alpha_t$ 和 $\sigma_t$ 基于 cosine schedule 计算。

### 逆向采样过程

从 $z_T$ 逐步去噪到 $z_0$：

$$z_s = c_1 \cdot z_t + c_2 \cdot \hat{x}_0 + \sigma_{t \to s} \cdot \epsilon$$

其中 $\hat{x}_0$ 是模型对原始数据的预测（x₀-prediction）。

### 损失函数

**表达量损失** (MSE)：

$$\mathcal{L}_{expr} = \frac{1}{N \cdot G} \sum_{i,g} (x_{expr}^{(i,g)} - \hat{x}_{expr}^{(i,g)})^2$$

**位置损失** (距离矩阵 MSE，旋转平移不变)：

$$\mathcal{L}_{pos} = \frac{1}{N^2} \sum_{i,j} (D_{true}^{(i,j)} - D_{pred}^{(i,j)})^2$$

其中 $D^{(i,j)} = \|pos_i - pos_j\|_2$

---

## 📈 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **表达量** | MSE, Pearson r, Cosine Sim | 重建精度 |
| **位置** | Distance Matrix MSE, Procrustes | 几何重建精度 |
| **空间** | kNN Preservation, Moran's I | 空间结构保持 |
| **聚类** | ARI, NMI, Accuracy | 细胞类型聚类效果 |

---

## 🔧 相对于 LUNA 的改进

| 模块 | LUNA | AlphaSTomics | 改进说明 |
|------|------|--------------|---------|
| **坐标维度** | 2D | 3D CCF | 支持配准后的三维空间数据 |
| **扩散模态** | 仅位置 | 表达量 + 位置 | 双模态联合扩散 |
| **基因选择** | 高变异基因 | 全基因 | 支持多平台数据统一 |
| **注意力** | Linear Attention | Linear Attention | O(n²) → O(n) |
| **采样模式** | 单向 | 双向 + 联合 | 更灵活的生成能力 |
| **分布式** | 不支持 | DDP/FSDP/DeepSpeed | 多机多卡训练 |
| **数据格式** | 内存加载 | Parquet 流式 | 支持大规模数据集 |

---

## 📚 参考

- [LUNA: Tissue reassembly with generative AI](https://github.com/mlbio-epfl/LUNA)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)

---

## 📄 许可证

MIT License
