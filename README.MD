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

### 基础能力
- 🧬 **双模态生成**：表达量 + 3D 坐标联合扩散
- 🎭 **Masked Diffusion**：特征级 masking，提升重建质量
- 🔄 **训练模式**：`expr_to_pos` / `pos_to_expr` / `joint`
- 📐 **旋转不变损失**：位置使用距离矩阵 MSE
- 🚀 **分布式训练**：支持 DDP / FSDP / DeepSpeed
- 📦 **HuggingFace 兼容**：Parquet 格式，支持流式加载

### 高级特性（新增）
- ⚡ **Gated Attention**：基于 Qwen3 的门控注意力机制（NeurIPS 2025 Oral）
  - Headwise / Elementwise 门控
  - QK Normalization (RMSNorm)
  - Attention-sink 防止
  - 参数减少 37%，性能提升
  
- 🎯 **Mixture of Experts (MoE)**：稀疏激活的专家网络
  - Top-K 专家选择（如 8 选 2）
  - 负载均衡机制
  - 总参数与 Dense 相近，激活参数仅 50%（4 选 2）
  - 适合大规模数据集

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/zEpoch/alphaStomics.git
cd alphaStomics

# 创建环境
conda create -n alphastomics python=3.12
conda activate alphastomics

# 安装依赖
pip install --upgrade pip
pip install torch pytorch-lightning pyyaml wandb scanpy anndata pyarrow
pip install -e .

# 验证安装
python -c "from alphastomics.diffusion_model import MaskedDiffusionModule; print('OK')"
```

### 训练模型

AlphaSTomics 提供两种训练方式：

#### 方式 1：使用独立训练脚本 `train.py`

适合快速实验和简单配置：

```bash
# 基础训练（使用默认配置）
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --exp_name my_experiment

# 启用 Gated Attention
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --use_gated_attention \
    --exp_name gated_experiment

# 启用 MoE（4 专家，激活 2 个）
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --use_moe \
    --num_experts 4 \
    --num_genes 2000 \
    --use_gated_attention \
    --use_moe \
    --num_experts 4 \
    --moe_top_k 2 \
    --exp_name gated_moe_experiment

# 启用 Masked Diffusion
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --enable_masking \
    --expression_mask_ratio 0.3 \
    --position_mask_ratio 0.33 \
    --exp_name masked_experiment

# 自定义训练参数
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --training_mode joint \
    --gpus 2 \
    --fp16 \
    --exp_name custom_experiment
```

#### 方式 2：使用模块化命令 `python -m alphastomics.main`

适合生产环境和流水线集成：

```bash
# 基础训练
python -m alphastomics.main train \
    --config config.yaml \
    --data_dir ./processed_data

# 启用 Gated Attention
python -m alphastomics.main train \
    --config config.yaml \
    --use_gated_attention

# 启用 MoE
python -m alphastomics.main train \
    --config config.yaml \
    --use_moe \
    --num_experts 4 \
    --moe_top_k 2

# 完整配置（Gated Attention + MoE + Masked Diffusion）
python -m alphastomics.main train \
    --config config.yaml \
    --use_gated_attention \
    --gate_type elementwise \
    --use_moe \
    --num_experts 8 \
    --moe_top_k 2 \
    --enable_masking \
    --expression_mask_ratio 0.3 \
    --position_mask_ratio 0.33 \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4
```

两种方式的主要区别：
- `train.py`: 独立脚本，需要指定 `--num_genes` 和 `--exp_name`
- `python -m alphastomics.main`: 模块化接口，支持完整的子命令流程（preprocess, train, test, sample 等）

### 快速测试

```bash
# 测试 Gated Attention
python demo_gated.py

# 测试 Gated Attention + MoE
python demo_gated_moe.py

# 对比四种配置
python compare_configs.py
```

---

## 📋 命令行参数

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config.yaml` |
| `--data_dir` | 数据目录（必需） | - |
| `--num_genes` | 基因数量（必需） | - |
| `--batch_size` | 批次大小 | 配置文件 |
| `--epochs` | 训练轮数 | 配置文件 |
| `--lr` | 学习率 | 配置文件 |
| `--training_mode` | 训练模式：`joint`/`expr_to_pos`/`pos_to_expr` | 配置文件 |

### Gated Attention 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_gated_attention` | 启用 Gated Attention | False |
| `--no_gated_attention` | 禁用 Gated Attention | - |
| `--gate_type` | 门控类型：`headwise`/`elementwise` | `headwise` |

### MoE 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_moe` | 启用 MoE | False |
| `--no_moe` | 禁用 MoE | - |
| `--num_experts` | 专家数量 | 4 |
| `--moe_top_k` | 每次激活的专家数 | 2 |

### Masked Diffusion 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_masking` | 启用 Masked Diffusion | False |
| `--disable_masking` | 禁用 Masked Diffusion | - |
| `--expression_mask_ratio` | 表达量遮罩比例 | 0.3 |
| `--position_mask_ratio` | 位置遮罩比例 | 0.33 |

### 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output_dir` | 输出目录 | `./outputs` |
| `--exp_name` | 实验名称 | `alphastomics` |
| `--resume_from` | 从检查点恢复 | None |
| `--gpus` | GPU 数量 | 1 |
| `--num_workers` | 数据加载线程数 | 4 |
| `--seed` | 随机种子 | 42 |
| `--fp16` | 混合精度训练 | False |

---

## 📁 项目结构

```plaintext
alphaStomics/
├── alphastomics/
│   ├── diffusion_model/        # 扩散模型核心
│   │   ├── module.py           # PyTorch Lightning 模块
│   │   ├── noise.py            # 噪声模型（cosine schedule）
│   │   ├── loss.py             # 双模态损失函数
│   │   ├── sample.py           # 采样器
│   │   └── train.py            # 训练逻辑
│   ├── attn_model/             # 注意力模型
│   │   ├── model.py            # 主模型
│   │   ├── transformer.py      # Transformer 编码器
│   │   ├── gated_attention.py  # Gated Attention（新增）
│   │   ├── moe.py              # Mixture of Experts（新增）
│   │   └── layers.py           # 基础层
│   └── utils/                  # 工具函数
│       ├── dataholder.py       # 数据容器
│       └── metrics.py          # 评估指标
├── train.py                    # 主训练脚本（新增）
├── demo_gated.py               # Gated Attention 演示
├── demo_gated_moe.py           # Gated + MoE 演示
├── compare_configs.py          # 配置对比脚本
├── config.yaml                 # 默认配置
├── GATED_ATTENTION.md          # Gated Attention 文档
├── GATED_MOE_GUIDE.md          # MoE 详细指南
├── DEMO_USAGE.md               # Demo 使用指南
└── README.MD
```

### 预处理命令

#### `preprocess` - 一步完成预处理（推荐单数据集）

```bash
python -m alphastomics.main preprocess \
    --input_dir ./raw_h5ad \           # 输入 h5ad 文件目录
    --output_dir ./processed \          # 输出目录
    --gene_list_file genes.txt \        # 固定基因列表（推荐）
    --position_key spatial \            # 坐标字段名（默认: spatial）
    --position_is_3d \                  # 坐标是否为 3D
    --z_spacing 10.0 \                  # 切片间 z 间距（2D 时使用）
    --cell_type_key cell_type \         # 细胞类型字段（可选，预训练不需要）
    --skip-log-normalize \              # 跳过 log normalize（数据已处理时）
    --skip-scale \                      # 跳过 scale
    --train_ratio 0.8 \                 # 训练集比例
    --val_ratio 0.1 \                   # 验证集比例
    --max_shard_size 100000             # 每个分片最大细胞数
```

#### `stage1` + `stage2` - 分步预处理（适合多数据集合并）

**场景**：不同数据集需要不同的预处理参数（如有的需要 log normalize，有的不需要）

```bash
# 数据集 A：需要 log normalize
python -m alphastomics.main stage1 \
    --input_dir ./dataset_A \
    --output_dir ./intermediate/A \
    --gene_list_file genes.txt          # 必须使用相同的基因列表！

# 数据集 B：已经 log normalized，跳过
python -m alphastomics.main stage1 \
    --input_dir ./dataset_B \
    --output_dir ./intermediate/B \
    --gene_list_file genes.txt \
    --skip-log-normalize

# 合并所有数据集，生成最终 Parquet
python -m alphastomics.main stage2 \
    --input_dirs ./intermediate/A ./intermediate/B \
    --output_dir ./final_dataset \
    --train_ratio 0.8 \
    --val_ratio 0.1
```

**⚠️ 重要**：合并多数据集时，所有 `stage1` 必须使用**相同的基因列表文件**！

**预处理流程**：
1. **Stage 1**：对每个切片进行 log normalize + scale（可选），保存中间格式
2. **Stage 2**：跨切片随机采样，生成 train/val/test 分割，序列化为 Parquet

**特点**：
- 内存友好：流式处理，不需要一次性加载所有数据
- 无放回采样：每个细胞恰好出现一次
- 细胞级分割：同一切片可能被分到不同集合
- 基因补全：缺失基因自动补 0

### 训练命令

```bash
python -m alphastomics.main train \
    --config ./config.yaml \         # 配置文件
    --data_dir ./processed \         # 数据目录
    --batch_size 1024 \              # 批次大小
    --streaming \                    # 流式加载（大数据集推荐）
    --buffer_size 50000 \            # shuffle buffer 大小
    --resume ./checkpoint.ckpt \     # 恢复训练
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
adata.obs['cell_type']     # 细胞类型标签（预训练时不需要）
adata.var_names            # 基因名称
```

### 预处理输出格式（Parquet）

```
processed_data/
├── data/
│   ├── train.parquet          # 或 train-00000.parquet, train-00001.parquet, ...
│   ├── validation.parquet
│   └── test.parquet
├── metadata.json              # 基因列表、统计信息
└── _intermediate_slices/      # 中间文件（可删除）
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

# 随机种子（确保可复现）
seed:
  global_seed: 42
  deterministic: false    # true 会降低性能但保证完全复现

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
  batch_size: 1024
  learning_rate: 1e-4
  max_epochs: 100

masking:
  enable: true               # 启用 Masked Diffusion
  expression_mask_ratio: 0.4
  position_mask_ratio: 0.3

data:
  processed_data_dir: ./processed
  preprocessing:
    normalize_position: false  # 不对坐标归一化（保持原始 CCF）
    scale: true
  loading:
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

## ⚙️ 配置对比与性能

### 四种注意力配置对比

运行 `python compare_configs.py` 可以对比以下四种配置的性能：

| 配置 | 总参数 | 激活参数 | FFN 激活比例 | 适用场景 |
|------|--------|----------|-------------|---------|
| **Baseline** (Linear Attn) | 707K | 707K | 100% | 基础模型 |
| **Gated Attention** | 445K (-37%) | 445K | 100% | **推荐**：参数少，性能好 |
| **MoE Only** (4 选 2) | 710K (+0.4%) | 644K (-9%) | 50.8% | 参数效率优先 |
| **Gated + MoE** | 448K (-37%) | 382K (-46%) | 50.8% | **最强**：大规模数据集 |

### 关键发现

**1. Gated Attention（优先推荐）**
- ✅ 参数减少 37%（移除 Linear Attention 依赖）
- ✅ 性能提升明显
- ✅ 训练稳定，易于调优
- ✅ 适合中小规模数据集

**2. MoE（参数效率）**
- ✅ 总参数几乎不变（+0.4%）
- ✅ FFN 层面激活 50%（4 选 2）
- ✅ 整体模型激活 90.7%（因为 MoE 只占一小部分）
- ⚠️ 需要足够大的数据集才能充分利用专家网络

**3. Gated + MoE（最强配置）**
- ✅ 参数减少 37%，激活参数减少 46%
- ✅ 最大模型容量和参数效率
- ✅ 适合大规模、复杂数据集
- ⚠️ 需要更多调优（load balancing）

### 使用建议

```bash
# 🥇 优先推荐：Gated Attention（elementwise 默认，性能最强）
python train.py --use_gated_attention ...

# 🥈 大规模数据：Gated + MoE
python train.py --use_gated_attention --use_moe --num_experts 8 --moe_top_k 2 ...

# 🥉 参数效率：仅 MoE
python train.py --use_moe --num_experts 4 --moe_top_k 2 ...
```

### MoE 参数说明

**激活参数计算：**
- 每个专家的隐藏层大小 = `d_ff / num_experts`
- 总参数 ≈ Dense FFN 的参数量（略多一点，因为有 Router）
- 激活参数 = Dense × (top_k / num_experts)

**示例（4 选 2）：**
- Dense FFN: 65,920 参数
- MoE 总参数: 67,328 参数（1.02x）
- MoE 激活参数: 34,176 参数（0.52x，约 50%）

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
| **坐标处理** | 归一化到 [-0.5, 0.5] | 保持原始坐标 | 不丢失坐标信息，可直接使用 |
| **扩散模态** | 仅位置 | 表达量 + 位置 | 双模态联合扩散 |
| **基因选择** | 高变异基因 | 全基因（固定列表） | 支持多平台数据统一，缺失补 0 |
| **注意力** | Linear Attention | **Gated Attention** | Qwen3 门控注意力，参数更少 |
| **专家网络** | - | **MoE (可选)** | 稀疏激活，提升模型容量 |
| **采样模式** | 单向 | 双向 + 联合 | 更灵活的生成能力 |
| **分布式** | 不支持 | DDP/FSDP/DeepSpeed | 多机多卡训练 |
| **数据格式** | 内存加载 | Parquet 流式 | 支持大规模数据集 |
| **可复现性** | 无 | 全局种子控制 | 完全可复现的实验 |

---

## 📚 参考文献

- [LUNA: Tissue reassembly with generative AI](https://github.com/mlbio-epfl/LUNA)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Qwen3 Gated Attention](https://arxiv.org/abs/...) - NeurIPS 2025 Oral
- [Mixture of Experts](https://arxiv.org/abs/...)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)

---

## 📖 详细文档

- [`GATED_ATTENTION.md`](GATED_ATTENTION.md) - Gated Attention 详细说明
- [`GATED_MOE_GUIDE.md`](GATED_MOE_GUIDE.md) - MoE 详细指南和最佳实践
- [`DEMO_USAGE.md`](DEMO_USAGE.md) - Demo 脚本使用指南

---

## 📄 许可证

MIT License
