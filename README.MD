# AlphaSTomics - 双模态空间转录组扩散模型

<p align="center">
  <b>同时对基因表达量和3D空间坐标进行扩散建模</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.0+-purple.svg" alt="Lightning">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 🎯 项目简介

AlphaSTomics 是一个先进的空间转录组学数据建模框架，基于扩散模型和 Transformer 架构。支持**双模态扩散**，可以同时对基因表达量和空间坐标进行生成。

### 核心特性

- 🔄 **双模态生成**: 同时支持表达量和位置的生成
- 🎛️ **灵活采样模式**: 表达量→位置、位置→表达量、联合生成
- 🔒 **几何不变性**: 位置损失使用距离矩阵，保证旋转平移不变性
- ⚡ **高效注意力**: 使用 Linear Attention，复杂度 O(n) 支持大规模数据
- 📐 **3D CCF 坐标**: 支持配准后的三维空间坐标（原 LUNA 仅支持 2D）
- 🌐 **多平台支持**: 兼容 StereoSeq、MERFISH、MERSCOPE 等多种平台数据
- 🚀 **分布式训练**: 支持多机多卡分布式训练（DDP、FSDP、DeepSpeed）

---

## 📁 项目结构

```
alphastomics/
├── __init__.py                    # 包入口
├── main.py                        # 主程序入口（CLI）
├── config.yaml                    # 配置文件
├── attn_model/                    # 注意力模型
│   ├── layers.py                  # 基础层（MLP, PositionMLP, PositionNorm）
│   ├── model.py                   # 主模型
│   ├── self_attention.py          # 自注意力（Linear Attention）
│   └── transformer.py             # Transformer 层
├── diffusion_model/               # 扩散模型
│   ├── diffusion_utils.py         # 扩散工具函数
│   ├── noise_model.py             # 双模态噪声模型
│   ├── loss.py                    # 双模态损失函数
│   ├── sample.py                  # 采样器
│   └── train.py                   # PyTorch Lightning 训练模块
├── utils/                         # 工具模块
│   ├── dataholder.py              # 统一数据容器
│   ├── dataloader.py              # 数据加载器（切片级/细胞级）
│   ├── metrics.py                 # 评估指标
│   └── embedding_extractor.py     # Embedding 提取工具
└── scripts/
    └── run_distributed.sh         # 分布式训练启动脚本
```

---

## 🔧 相对于 LUNA 的优化与改进

### 1. 架构层面的改进

| 模块 | LUNA | AlphaSTomics | 改进说明 |
|------|------|--------------|---------|
| **坐标维度** | 2D | 3D | 支持三维空间转录组数据 |
| **扩散模态** | 仅位置 | 表达量 + 位置 | 双模态联合扩散 |
| **注意力类型** | 标准 Attention | Linear Attention | O(n²) → O(n)，支持更多细胞 |
| **采样模式** | 单向（表达量→位置） | 双向 + 联合 | 更灵活的生成能力 |

### 2. 代码修复

#### ✅ 修复：位置范数输出维度错误

**问题**: 原代码中 `mlp_out_position_norm` 输出 3 维向量，但应该输出 1 维标量范数

```python
# ❌ 原代码（错误）
self.mlp_out_position_norm = nn.Sequential(
    nn.Linear(4, hidden_dims),
    nn.Linear(hidden_dims, 3)  # 输出 3 维 - 错误！
)
new_pos = pos * (new_norm / norm)  # new_norm 是向量，语义错误

# ✅ 修复后（正确，与 LUNA 一致）
self.mlp_out_position_norm = nn.Sequential(
    nn.Linear(hidden_dims + 3 + 1, hidden_dims),  # 特征 + 位置 + 范数
    nn.Linear(hidden_dims, 1)  # 输出 1 维范数标量
)
new_pos = pos * new_norm / (norm + eps)  # 正确的范数缩放
```

#### ✅ 修复：位置编码丢失距离信息

**问题**: 原代码只保留方向，丢失了距离信息

```python
# ❌ 原代码（丢失距离）
normalized_positions = positions / (norm + 1e-7)  # 只有方向
transformed = mlp(normalized_positions)  # 3 维输入

# ✅ 修复后（保留方向 + 距离）
direction = positions / (norm + 1e-7)  # 方向
features = torch.cat([direction, norm], dim=-1)  # 方向 + 距离 = 4 维
transformed = mlp(features)
```

#### ✅ 修复：LayerNorm eps 被注释

```python
# ❌ 原代码
self.norm = LayerNorm(dim)  # eps 被注释掉

# ✅ 修复后
self.norm = LayerNorm(dim, eps=layer_norm_eps)  # 恢复 eps 参数
```

#### ✅ 修复：缺少 mask 处理和位置中心化

```python
# ✅ 添加 mask 处理
out_expression = out_expression * node_mask.unsqueeze(-1)
out_position = out_position * node_mask.unsqueeze(-1)

# ✅ 添加位置中心化
mean_pos = (out_position * mask).sum(dim=1) / mask.sum(dim=1)
out_position = (out_position - mean_pos) * mask
```

### 3. 新增功能

#### 🆕 双模态噪声模型 (`noise_model.py`)

支持同时对表达量和位置加噪：

```python
# 前向扩散
z_t^{expr} = α_t · x_{expr} + σ_t · ε_{expr}
z_t^{pos}  = α_t · x_{pos}  + σ_t · ε_{pos}

# 可以独立控制
noisy_data = noise_model.apply_noise(
    data,
    noise_expression=True,   # 是否对表达量加噪
    noise_position=True      # 是否对位置加噪
)
```

#### 🆕 双模态损失函数 (`loss.py`)

```python
# 总损失
L = λ_expr · L_expression + λ_pos · L_position

# 表达量损失: MSE
L_expression = MSE(pred_expr, true_expr)

# 位置损失: 距离矩阵 MSE（旋转平移不变）
D_pred[i,j] = ||pos_pred[i] - pos_pred[j]||
D_true[i,j] = ||pos_true[i] - pos_true[j]||
L_position = MSE(D_pred, D_true)
```

#### 🆕 灵活的采样器 (`sample.py`)

支持三种采样模式：

| 模式 | 输入条件 | 生成目标 | 应用场景 |
|------|---------|---------|---------|
| `expr_to_pos` | 表达量（清晰） | 位置 | 原始 LUNA 任务：从基因表达重建组织结构 |
| `pos_to_expr` | 位置（清晰） | 表达量 | 空间转录组推断：从位置预测基因表达 |
| `joint` | 两者都是噪声 | 两者 | 完全生成：生成虚拟组织 |

```python
sampler = DiffusionSampler(model, noise_model)

# 表达量 → 位置（原始任务）
_, positions = sampler.sample(expr, pos, mask, mode="expr_to_pos")

# 位置 → 表达量（新任务）
expression, _ = sampler.sample(expr, pos, mask, mode="pos_to_expr")

# 联合生成（完全生成）
expression, positions = sampler.sample(expr, pos, mask, mode="joint")
```

#### 🆕 统一数据容器 (`dataholder.py`)

```python
data = DataHolder(
    expression=expr,          # (B, N, G) 表达量
    positions=pos,            # (B, N, 3) 3D 位置
    node_mask=mask,           # (B, N) 有效节点掩码
    noisy_expression=...,     # 加噪后的表达量
    noisy_positions=...,      # 加噪后的位置
    diffusion_time=t,         # 扩散时间
)
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install torch pytorch-lightning
pip install linear_attention_transformer
pip install pyyaml wandb scanpy anndata
```

### 1. 数据预处理

```bash
# 从 h5ad 文件预处理数据
python -m alphastomics.main preprocess \
    --input_dir /path/to/h5ad_files \
    --output_dir /path/to/processed \
    --gene_list_file /path/to/gene_list.txt \
    --position_key ccf
```

**参数说明：**
- `--gene_list_file`: 固定基因列表文件（每行一个基因名），支持多平台数据统一
- `--position_key`: 坐标在 `adata.obsm` 中的 key（默认 `ccf`，已配准的 3D 坐标）

### 2. 训练模型

**单卡训练：**
```bash
python -m alphastomics.main train \
    --config config.yaml \
    --data_dir /path/to/processed \
    --data_mode slice  # 或 cell
```

**单机多卡训练：**
```bash
torchrun --nproc_per_node=4 \
    -m alphastomics.main train \
    --config config.yaml \
    --data_dir /path/to/processed
```

**多机多卡训练：**
```bash
# 节点 0 (Master)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=10.0.0.1 --master_port=29500 \
    -m alphastomics.main train --config config.yaml

# 节点 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --master_addr=10.0.0.1 --master_port=29500 \
    -m alphastomics.main train --config config.yaml
```

### 3. 测试模型

```bash
python -m alphastomics.main test \
    --checkpoint outputs/checkpoints/best.ckpt \
    --data_dir /path/to/processed
```

### 4. 采样生成

```bash
python -m alphastomics.main sample \
    --checkpoint outputs/checkpoints/best.ckpt \
    --input_data input.pt \
    --sample_mode expr_to_pos  # 或 pos_to_expr, joint
```

### 5. 提取 Embedding

```bash
python -m alphastomics.main extract \
    --checkpoint outputs/checkpoints/best.ckpt \
    --data_dir /path/to/processed \
    --output embeddings.npz \
    --extract_mode transformer
```

---

## ⚙️ 配置说明

编辑 `config.yaml`：

```yaml
# 训练模式
training_mode: "joint"  # "expr_to_pos", "pos_to_expr", "joint"

# 扩散参数
diffusion:
  diffusion_steps: 1000
  nu_expression: 1.0
  nu_position: 1.0

# 损失权重
loss:
  lambda_expression: 1.0
  lambda_position: 1.0
  use_distance_matrix: true  # 旋转平移不变性

# 数据参数
data:
  preprocessing:
    gene_list_file: null      # 固定基因列表
    position_key: "ccf"       # 3D 坐标 key
    position_is_3d: true      # 已经是 3D 坐标
  loading:
    mode: "slice"             # slice(切片级) 或 cell(细胞级)

# 分布式训练
distributed:
  enabled: true
  strategy: "ddp"             # ddp, fsdp, deepspeed
  num_nodes: 2
  devices: 4
```

---

## 📊 两种训练模式

| 模式 | 说明 | batch_size 含义 | 适用场景 |
|------|------|----------------|----------|
| `slice` | 每次输入一整张切片 | 每批切片数量 | 保持切片内空间结构 |
| `cell` | 每次采样固定数量细胞 | 每批细胞数量 | 大规模数据训练 |

---

---

## 📝 API 使用示例

### Python API

```python
from alphastomics import AlphaSTomicsModule, DiffusionSampler
from alphastomics.utils import create_dataloaders, SpatialDataPreprocessor

# 1. 数据预处理
preprocessor = SpatialDataPreprocessor(
    gene_list=['Gene1', 'Gene2', ...],  # 固定基因列表
    position_key='ccf',                  # 3D CCF 坐标
    position_is_3d=True,
)
metadata = preprocessor.preprocess_and_save(h5ad_files, output_dir)

# 2. 创建数据加载器
train_loader, val_loader, test_loader, meta = create_dataloaders(
    data_dir=output_dir,
    mode='slice',  # 或 'cell'
    batch_size=1,
)

# 3. 训练
import pytorch_lightning as pl
model = AlphaSTomicsModule(cfg=config, num_genes=meta['n_genes'])
trainer = pl.Trainer(max_epochs=100, devices=4, strategy='ddp')
trainer.fit(model, train_loader, val_loader)

# 4. 采样
sampler = DiffusionSampler(model.model, model.noise_model)

# 表达量 → 位置（原始任务）
_, positions = sampler.sample(expr, pos, mask, mode="expr_to_pos")

# 位置 → 表达量（新任务）
expression, _ = sampler.sample(expr, pos, mask, mode="pos_to_expr")

# 联合生成
expression, positions = sampler.sample(expr, pos, mask, mode="joint")
```

### 提取 Embedding

```python
from alphastomics.utils import EmbeddingExtractor

extractor = EmbeddingExtractor(model, mode='transformer')
embeddings = extractor.extract(data_loader)

# embeddings 包含:
# - expression_embeddings: 表达量特征
# - position_embeddings: 位置特征
# - cell_types: 细胞类型标签
```

---

## 📊 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **表达量** | MSE, MAE, PCC, Spearman | 预测精度 |
| **位置** | Distance Matrix MSE, Procrustes | 几何重建精度 |
| **空间** | kNN Preservation, Moran's I | 空间结构保持 |
| **聚类** | ARI, NMI, Accuracy | 细胞类型聚类效果 |

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

## 🔧 相对于 LUNA 的改进

| 模块 | LUNA | AlphaSTomics | 改进说明 |
|------|------|--------------|---------|
| **坐标维度** | 2D | 3D CCF | 支持配准后的三维空间数据 |
| **扩散模态** | 仅位置 | 表达量 + 位置 | 双模态联合扩散 |
| **基因选择** | 高变异基因 | 全基因 | 支持多平台数据统一 |
| **注意力** | Linear Attention | Linear Attention | O(n²) → O(n) |
| **采样模式** | 单向 | 双向 + 联合 | 更灵活的生成能力 |
| **分布式** | 不支持 | DDP/FSDP/DeepSpeed | 多机多卡训练 |

---

## 📚 参考

- [LUNA: Tissue reassembly with generative AI](https://github.com/mlbio-epfl/LUNA)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

---

## 📄 许可证

MIT License