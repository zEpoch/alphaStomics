# Masked Diffusion Transformer 使用文档

## 📌 概述

**Masked Diffusion** 是 AlphaSTomics 的一个可选特性，在扩散模型训练时对每个细胞的**表达量向量**和**坐标向量**进行特征级 masking。

### 核心思想

- 在加噪后的数据上，随机 mask 部分**基因表达**或**坐标维度**
- 模型需要从部分观测 + 扩散噪声中重建完整数据
- 增强模型鲁棒性，学习特征间依赖关系

### 与传统扩散的区别

| 维度 | 传统扩散 | Masked Diffusion |
|------|---------|-----------------|
| 输入破坏 | 添加高斯噪声 | 噪声 + 随机 mask 部分特征 |
| 训练目标 | 预测噪声/原始数据 | 预测噪声 + 重建被 mask 的特征 |
| 鲁棒性 | 对噪声鲁棒 | 对噪声 + 缺失数据鲁棒 |

---

## 🚀 快速开始

### 方法 1：通过配置文件启用

在 `config.yaml` 中设置：

```yaml
masking:
  enable: true                       # 启用 Masked Diffusion
  expression_mask_ratio: 0.4         # mask 40% 的基因
  position_mask_ratio: 0.33          # 平均 mask 1 个坐标维度
  mask_strategy: "random"            # 随机 mask
  mask_expression: true              # mask 表达量
  mask_position: true                # mask 坐标
  reconstruction_weight: 0.5         # 重建损失权重
  masking_probability: 0.5           # 50% 的 batch 应用 masking
```

然后正常训练：

```bash
python main.py --config config.yaml
```

### 方法 2：在代码中配置

```python
from diffusion_model import (
    AlphaSTomicsModule,
    MaskedDiffusionModule,
    MaskingConfig,
)

# 创建配置
cfg = {
    "training_mode": "joint",
    "masking": {
        "enable": True,
        "expression_mask_ratio": 0.4,
        "position_mask_ratio": 0.33,
        "mask_strategy": "random",
        "reconstruction_weight": 0.5,
    },
    # ... 其他配置
}

# 创建模块（自动启用 Masked Diffusion）
module = AlphaSTomicsModule(cfg, num_genes=2000)

# 训练
trainer.fit(module, train_loader, val_loader)
```

---

## 📊 Masking 策略详解

### 1. 表达量 Masking（基因级）

对每个细胞的 G 维基因表达向量：

```
原始: [g1, g2, g3, ..., gG]  # G=2000 个基因
Masked: [g1, [M], g3, ..., [M]]  # 随机 mask 40% 基因
```

**三种策略**：

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| `random` | 完全随机选择基因 | 通用，初步实验 |
| `block` | mask 连续的基因块 | 模拟基因模块/通路 |
| `structured` | 优先 mask 高表达基因 | 更有挑战性 |

### 2. 坐标 Masking（维度级）

对每个细胞的 3D 坐标 [x, y, z]：

```
原始: [x=1.5, y=2.3, z=-0.8]
Masked: [[M], y=2.3, z=-0.8]  # mask x 坐标
```

**应用场景**：
- 从 2D 切片重建 3D 结构
- 学习空间约束
- 处理不完整空间数据

---

## 🔧 核心模块

### MaskGenerator

生成 mask 的核心类：

```python
from diffusion_model.masking import MaskGenerator

generator = MaskGenerator(
    expression_mask_ratio=0.4,
    position_mask_ratio=0.33,
    mask_strategy='random'
)

# 生成表达量 mask
expr_mask = generator.generate_expression_mask(expression)  # (B, N, G) bool

# 生成坐标 mask
pos_mask = generator.generate_position_mask(position)  # (B, N, 3) bool
```

### MaskToken

可学习的 mask 占位符：

```python
from diffusion_model.masking import MaskToken

token = MaskToken(expression_dim=2000, position_dim=3)

# 应用 mask
masked_expr = token.apply_expression_mask(expression, expr_mask)
masked_pos = token.apply_position_mask(position, pos_mask)
```

### MaskedDiffusionModule

完整的 Masked Diffusion 模块：

```python
from diffusion_model.masking import (
    MaskedDiffusionModule,
    MaskingConfig,
)

config = MaskingConfig(
    enable=True,
    expression_mask_ratio=0.4,
    position_mask_ratio=0.33,
)

module = MaskedDiffusionModule(
    expression_dim=2000,
    position_dim=3,
    config=config
)

# 应用 masking
masked_expr, masked_pos, mask_info = module.apply_masking(
    expression=noisy_expr,
    position=noisy_pos,
    apply=True
)

# 计算重建损失
recon_loss, log_dict = module.compute_reconstruction_loss(
    pred_expression, pred_position, mask_info, node_mask
)
```

---

## 📈 训练流程

### 标准训练流程

```
1. 原始数据: expression (B, N, G), position (B, N, 3)
           ↓
2. 添加扩散噪声（NoiseModel.apply_noise）
   noisy_expr = α_t * expr + σ_t * ε_expr
   noisy_pos = α_t * pos + σ_t * ε_pos
           ↓
3. 应用 Masking（可选，MaskedDiffusionModule.apply_masking）
   masked_expr = mask_token.apply(noisy_expr, expr_mask)
   masked_pos = mask_token.apply(noisy_pos, pos_mask)
           ↓
4. 模型预测
   pred_expr, pred_pos = model(masked_expr, t, masked_pos)
           ↓
5. 计算损失
   diff_loss = MSE(pred_expr, expr) + DistMatrix(pred_pos, pos)
   recon_loss = MSE(pred[mask], original[mask])  # 只在 mask 位置
   total_loss = diff_loss + λ * recon_loss
           ↓
6. 反向传播
```

### 渐进式 Masking

建议分阶段训练：

```yaml
masking:
  enable: true
  progressive_masking: true    # 启用渐进式
  progressive_steps: 10000     # 在 10k 步内从 0 增加到目标 ratio
  expression_mask_ratio: 0.4   # 最终目标 ratio
```

或在代码中手动控制：

```python
# 阶段 1：前 10k 步，无 masking
if step < 10000:
    cfg['masking']['enable'] = False

# 阶段 2：10k-20k 步，轻度 masking
elif step < 20000:
    cfg['masking']['expression_mask_ratio'] = 0.2

# 阶段 3：20k+ 步，标准 masking
else:
    cfg['masking']['expression_mask_ratio'] = 0.4
```

---

## ⚙️ 超参数推荐

### 初始设置（保守）

```yaml
masking:
  enable: true
  expression_mask_ratio: 0.3    # 30%
  position_mask_ratio: 0.2      # 20%
  mask_strategy: "random"
  reconstruction_weight: 0.5
  masking_probability: 0.5      # 50% batch 应用
```

### 激进设置（更强正则化）

```yaml
masking:
  enable: true
  expression_mask_ratio: 0.5    # 50%
  position_mask_ratio: 0.4      # 40%
  mask_strategy: "structured"
  reconstruction_weight: 1.0
  masking_probability: 0.8
```

### 参数说明

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `expression_mask_ratio` | 0.3-0.5 | 太高会导致训练困难 |
| `position_mask_ratio` | 0.2-0.4 | 坐标只有 3 维，不宜过高 |
| `reconstruction_weight` | 0.3-1.0 | 平衡扩散损失和重建损失 |
| `masking_probability` | 0.3-0.8 | 控制应用 masking 的频率 |

---

## 🎯 应用场景

### 1. 缺失数据补全

```python
# 模拟缺失基因
incomplete_expr = expression.clone()
incomplete_expr[:, :, missing_genes] = 0

# Masked Diffusion 训练的模型能更好地处理
pred_expr, pred_pos = model(incomplete_expr, t, position)
```

### 2. 2D → 3D 重建

```yaml
masking:
  mask_expression: false        # 不 mask 表达量
  mask_position: true           # 只 mask 坐标
  position_mask_ratio: 0.5      # mask 更多坐标维度
```

### 3. 学习基因调控网络

分析重建误差：

```python
# 哪些基因容易从其他基因预测？
for gene_idx in range(num_genes):
    mask_single_gene(gene_idx)
    error = compute_reconstruction_error()
    
    # 低误差 → 基因冗余/可预测
    # 高误差 → 关键/独立基因
```

---

## 📊 监控指标

训练时会自动记录以下指标：

```
train/loss/total                    # 总损失
train/loss/expression_mse           # 表达量扩散损失
train/loss/position_dist_matrix     # 位置扩散损失
train/loss/masked_expr_reconstruction   # 表达量重建损失
train/loss/masked_pos_reconstruction    # 位置重建损失
train/masking/applied               # 是否应用了 masking (0/1)
```

建议监控：
- `masked_expr_reconstruction` 应该逐渐下降
- `masking/applied` 应该约等于 `masking_probability`

---

## ⚠️ 注意事项

### 1. 训练稳定性

- 建议先训练无 masking 版本至收敛
- 再启用 masking 继续训练
- 或使用 `progressive_masking`

### 2. 推理时

- 必须禁用 masking
- `AlphaSTomicsModule` 的验证/测试步骤自动禁用

### 3. 内存占用

```python
# Mask 张量内存估算
# (B=32, N=10000, G=2000) → ~640 MB 额外内存
```

### 4. 计算开销

- Masking 本身：<1% 额外时间
- 重建损失计算：~5-10% 额外时间

---

## 🆘 常见问题

**Q: 必须同时 mask 表达量和坐标吗？**  
A: 不需要，可以通过 `mask_expression` 和 `mask_position` 单独控制。

**Q: Mask ratio 设多少合适？**  
A: 表达量 30-50%，坐标 20-40%。建议从低开始。

**Q: 重建损失权重多少合适？**  
A: 初期 0.5，如果扩散损失收敛慢可降低到 0.3。

**Q: 为什么验证集损失没有重建损失？**  
A: 设计如此，验证/测试时不应用 masking 以获得公平评估。

---

## 📚 文件结构

```
diffusion_model/
├── __init__.py           # 导出所有模块
├── masking.py            # ← Masked Diffusion 核心实现
├── noise_model.py        # 噪声模型（已集成 masking）
├── loss.py               # 损失函数（已集成重建损失）
├── train.py              # 训练模块（已集成 masking）
├── sample.py             # 采样器
└── diffusion_utils.py    # 工具函数

docs/
└── MASKED_DIFFUSION.md   # 本文档
```

---

## 📈 预期效果

### 定量改进

- 缺失数据补全 RMSE 降低 10-20%
- 下游聚类 ARI 提升 5-15%
- 2D→3D 重建误差降低

### 定性改进

- 模型对测序深度变化更鲁棒
- 学习到的表示更具生物学意义
- 可解释性：分析基因依赖关系
