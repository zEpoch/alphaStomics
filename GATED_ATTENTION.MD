# Gated Attention for AlphaSTomics

## 🎯 概述

我们为 AlphaSTomics 引入了 **Gated Attention** 机制，完全替代原有的 Linear Attention。这一改进基于 Qwen3 的 Gated Attention 实现，专门针对空间转录组的双模态扩散建模优化。

### 为什么要替换 Linear Attention？

| 方面 | Linear Attention | Gated Attention | 说明 |
|------|------------------|-----------------|------|
| **复杂度** | O(n) | O(n²) | Batch 训练下 n 不大，O(n²) 可接受 |
| **精确性** | 近似 | 精确 | 空间关系建模需要精确注意力 |
| **动态性** | 静态 | 动态（query-dependent） | 扩散不同时间步需要不同关注点 |
| **双模态** | 无显式支持 | Gating 机制支持 | 表达量和位置需要动态平衡 |
| **Attention Sink** | 存在 | 避免 | 防止某些细胞被过度关注 |
| **训练稳定性** | 一般 | 优秀 | 支持更大学习率 |

---

## 🔧 核心实现

### 1. GatedMultiHeadAttention

标准的多头注意力 + 门控机制：

```python
class GatedMultiHeadAttention(nn.Module):
    """
    Softmax Attention + Query-dependent Gating
    
    gate_type:
        - 'headwise': 每个头一个标量 gate (推荐)
        - 'elementwise': 每个元素独立 gate
        - 'none': 标准 Softmax Attention
    """
```

**核心公式**：

$$\text{Output} = \sigma(g(Q)) \odot \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $g(Q)$: 门控函数（与 Q 一起从输入投影得到）
- $\sigma$: Sigmoid 激活
- $\odot$: 逐元素乘法（headwise 时对每个头）

### 2. GatedSelfAttentionModel

适配 AlphaSTomics 的自注意力模块：

```python
# 保持与原 SelfAttentionModel 相同的接口
output_expression, output_positions, output_time = model(
    expression_features,  # (B, N, expr_dim)
    diffusion_time,       # (B, time_dim)
    position_features     # (B, N, 3)
)
```

**特点**：
- 无缝替换原 `SelfAttentionModel`
- 保留位置编码（方向 + 距离）
- 支持时间步条件

### 3. 集成到 TransformerLayer

```python
TransformerLayer(
    ...,
    use_gated_attention=True,  # 启用 Gated Attention
    gate_type='headwise',       # 门控类型
    use_qk_norm=True            # QK 归一化
)
```

---

## 📝 使用方法

### 1. 配置文件设置

修改 `config.yaml`：

```yaml
TransformerLayer_setting:
  num_layers: 6
  num_heads: 8
  ...
  
  # 🆕 Gated Attention 配置
  use_gated_attention: true    # 启用（推荐）
  gate_type: "headwise"        # headwise / elementwise / none
  use_qk_norm: true            # QK 归一化（提升稳定性）
```

### 2. 训练

```bash
# 使用 Gated Attention 训练
python -m alphastomics.main train \
    --config config.yaml \
    --data_dir ./processed_data \
    --batch_size 1024

# 对比实验：关闭 Gated Attention
# 在 config.yaml 中设置 use_gated_attention: false
```

### 3. 测试实现

```bash
# 运行测试脚本
python test_gated_attention.py
```

测试将验证：
- ✓ `GatedMultiHeadAttention` 各模式工作正常
- ✓ `GatedSelfAttentionModel` 输出形状正确
- ✓ 完整 `Model` 可以正常前向传播
- ✓ `AlphaSTomicsModule` 可以正常训练

---

## 📊 预期效果

基于 Qwen3 论文的结论，我们预期在 AlphaSTomics 上看到以下改进：

### 1. 训练稳定性

- **更大的学习率**：Gating 提供更好的梯度流
- **更快的收敛**：动态调制减少冗余信息

### 2. 重建质量

| 指标 | 预期改进 | 原因 |
|------|---------|------|
| 表达量 MSE | -5% ~ -10% | 更精确的细胞间关系建模 |
| 位置 Distance MSE | -10% ~ -15% | 空间结构的精确注意力 |
| Masked 重建 | -15% ~ -20% | Gating + Masking 协同效果 |

### 3. 空间结构保持

- **kNN Preservation**：提升 5-10%
- **Moran's I**：空间自相关性更强
- **无 Attention Sink**：所有细胞被平等对待

### 4. 双模态对齐

- `joint` 模式下表达量-位置一致性提升
- 不同扩散时间步的动态调整能力

---

## 🔬 消融实验建议

### 实验设计

```yaml
# 1. Baseline: Linear Attention
use_gated_attention: false

# 2. Gated (headwise)
use_gated_attention: true
gate_type: "headwise"
use_qk_norm: true

# 3. Gated (elementwise)
use_gated_attention: true
gate_type: "elementwise"
use_qk_norm: true

# 4. Gated without QK norm
use_gated_attention: true
gate_type: "headwise"
use_qk_norm: false
```

### 评估指标

**必须报告**：
- 训练损失曲线
- 验证集 MSE（表达量 + 位置）
- 参数量和训练时间

**推荐报告**：
- kNN Preservation
- 距离矩阵相关性
- 细胞类型聚类 ARI/NMI
- 注意力权重可视化（检查 attention sink）

---

## 📐 参数量对比

以默认配置为例：

```python
# 假设: d_model=256, num_heads=8, num_layers=6

# Linear Attention (baseline)
# - 每层注意力: ~0.5M params

# Gated Attention (headwise)
# - 每层注意力: ~0.52M params (+4%)
# - 额外: num_heads 个 gate scores

# Gated Attention (elementwise)
# - 每层注意力: ~0.75M params (+50%)
# - 额外: d_model 个 gate scores
```

**推荐**：
- 小数据集（<10M 细胞）：headwise
- 大数据集：可尝试 elementwise
- 资源受限：headwise 几乎无额外开销

---

## 🐛 故障排除

### 问题 1: OOM (Out of Memory)

**原因**: O(n²) 注意力在大 batch 下内存消耗高

**解决**:
```yaml
training:
  batch_size: 512  # 降低 batch size
  accumulate_grad_batches: 2  # 梯度累积补偿
```

### 问题 2: 训练不稳定

**原因**: 门控梯度消失/爆炸

**解决**:
```yaml
TransformerLayer_setting:
  use_qk_norm: true  # 启用 QK 归一化
  dropout: 0.2       # 增加 dropout

training:
  gradient_clip_val: 0.5  # 更严格的梯度裁剪
```

### 问题 3: 性能下降

**可能原因**:
1. 数据集太小（<1000 细胞/样本）
2. 学习率不合适
3. Gate 全部接近 0 或 1

**调试**:
```python
# 在 forward 中添加日志
gate_scores = torch.sigmoid(self.gate(...))
print(f"Gate mean: {gate_scores.mean()}, std: {gate_scores.std()}")
# 健康范围: mean ~0.5, std ~0.2
```

---

## 📚 参考文献

1. **Qwen3 Gated Attention**  
   Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", NeurIPS 2025  
   https://arxiv.org/abs/2505.06708

2. **LUNA (原 AlphaSTomics 基础)**  
   Tissue reassembly with generative AI  
   https://github.com/mlbio-epfl/LUNA

3. **Attention is All You Need**  
   Vaswani et al., 2017

---

## ✅ Checklist

迁移到 Gated Attention 前请确认：

- [ ] 数据已预处理为 Parquet 格式
- [ ] `config.yaml` 中正确设置 `use_gated_attention: true`
- [ ] 运行 `test_gated_attention.py` 通过所有测试
- [ ] 准备对比实验（Gated vs Linear）
- [ ] 监控训练时的 GPU 内存使用
- [ ] 保存注意力权重用于可视化分析

---

## 🤝 贡献

如果你发现问题或有改进建议，欢迎提交 Issue 或 PR！

**特别关注的点**：
- 不同数据集上的效果对比
- 参数高效的 gating 变体
- 注意力权重的生物学解释

---

**License**: MIT  
**作者**: AlphaSTomics Team  
**更新日期**: 2025-12-08
