# Gated Attention + MoE for AlphaSTomics

## 📚 概述

本项目在 AlphaSTomics 中集成了两个关键技术改进：

1. **Gated Attention**: 替代 Linear Attention，提供更精确的空间建模能力
2. **Mixture of Experts (MoE)**: 可选的 FFN 替代方案，显著提升模型容量

---

## 🎯 Gated Attention

### 核心优势

| 特性 | Linear Attention | Gated Attention |
|------|-----------------|-----------------|
| **复杂度** | O(n) | O(n²) |
| **精确性** | 近似 | 精确 |
| **空间建模** | △ | ✓✓ |
| **训练稳定性** | △ | ✓✓ |
| **Attention Sink** | 存在 | 解决 |
| **适用场景** | 超长序列 | Batch 训练 |

### 使用方法

在 `config.yaml` 中启用：

```yaml
TransformerLayer_setting:
  use_gated_attention: true
  gate_type: "headwise"  # 或 "elementwise"
  use_qk_norm: true
```

### 门控类型

- **headwise** (推荐): 每个注意力头一个标量 gate
  - 参数少，稳定性好
  - 适合大多数场景
  
- **elementwise**: 每个元素一个 gate
  - 更灵活，表达能力更强
  - 参数量较大

---

## 🧠 Mixture of Experts (MoE)

### 核心原理

MoE 用多个"专家"网络替代标准的 FFN：

```
标准 FFN:  Input → Dense(d_ff) → ReLU → Dense(d_model) → Output

MoE:       Input → Router (选择 top-k 专家)
                 ↓
           Expert_1, Expert_2, ..., Expert_k
                 ↓
           加权组合 → Output
```

### 核心优势

1. **容量提升**: 8个专家 ≈ 8x FFN 的容量
2. **计算高效**: 每个 token 只激活 top-k 个专家（通常 k=2）
3. **专业化**: 不同专家学习不同模式
   - 可能某些专家专注表达量，某些专注位置
   - 或不同专家负责不同细胞类型/空间区域

### 使用方法

在 `config.yaml` 中启用：

```yaml
TransformerLayer_setting:
  use_moe: true
  num_experts: 8       # 专家数量
  moe_top_k: 2        # 每个 token 激活的专家数
  moe_load_balance_loss_weight: 0.01
```

### 参数建议

| 数据集规模 | num_experts | moe_top_k | 说明 |
|-----------|-------------|-----------|------|
| 小 (<10k 样本) | 4 | 1 | 避免过拟合 |
| 中 (10k-100k) | 8 | 2 | 平衡容量和效率 |
| 大 (>100k) | 16 | 2 | 充分利用数据 |

### 负载均衡

MoE 包含辅助损失，确保专家被均匀使用：

```python
# 训练时自动添加到总损失
total_loss = main_loss + moe_aux_loss
```

权重通过 `moe_load_balance_loss_weight` 控制（建议 0.01-0.1）。

---

## 🔧 配置示例

### 1. 仅 Gated Attention（推荐起点）

```yaml
TransformerLayer_setting:
  use_gated_attention: true
  gate_type: "headwise"
  use_qk_norm: true
  use_moe: false
```

### 2. Gated Attention + MoE（最大性能）

```yaml
TransformerLayer_setting:
  use_gated_attention: true
  gate_type: "headwise"
  use_qk_norm: true
  use_moe: true
  num_experts: 8
  moe_top_k: 2
  moe_load_balance_loss_weight: 0.01
```

### 3. 仅 MoE（对比实验）

```yaml
TransformerLayer_setting:
  use_gated_attention: false
  use_moe: true
  num_experts: 8
  moe_top_k: 2
```

---

## 📊 预期效果

### Gated Attention

| 指标 | 预期改进 |
|------|---------|
| 表达量重建 MSE | -5% ~ -10% |
| 位置 Distance MSE | -10% ~ -15% |
| 训练稳定性 | 支持更大学习率 |
| Masked Diffusion 效果 | ✓✓ |

### MoE

| 指标 | 预期改进 |
|------|---------|
| 模型容量 | +（num_experts - 1）× FFN |
| 表达量/位置重建 | -10% ~ -20% |
| 训练时间 | +20% ~ +30% |
| 参数量 | +（num_experts - 1）× FFN参数 |

### Gated Attention + MoE

组合使用时，效果叠加：
- 表达量/位置重建: -15% ~ -30%
- 在复杂任务上提升更明显

---

## 🧪 消融实验建议

1. **Baseline**: 标准配置（Linear Attention，无 MoE）
2. **+Gated**: 添加 Gated Attention
3. **+MoE**: 添加 MoE（4 experts, top-1）
4. **+Both**: Gated Attention + MoE (8 experts, top-2)

每个配置训练相同的 epochs，记录：
- 训练/验证损失曲线
- 表达量/位置重建误差
- 训练时间
- 参数量

---

## 📖 参考文献

1. **Gated Attention**: [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
   - NeurIPS 2025 Oral (Top 1.5%)
   - 已被 Qwen3-Next 采用

2. **Mixture of Experts**: 
   - [Switch Transformers](https://arxiv.org/abs/2101.03961)
   - [GShard](https://arxiv.org/abs/2006.16668)

---

## 🐛 故障排除

### MoE 训练不稳定

- 降低 `moe_load_balance_loss_weight`（如 0.001）
- 减少专家数量
- 增加 top-k

### 显存不足

- 减少 `num_experts`
- 使用 `gate_type: "headwise"` 而不是 "elementwise"
- 减小 batch_size

### 某些专家未被使用

- 增加 `moe_load_balance_loss_weight`
- 添加 noisy gating（已默认启用）

---

## ✅ 测试

运行测试脚本验证实现：

```bash
# 测试 MoE 模块
python test_moe.py

# 测试完整集成
python test_gated_attention.py
```

---

## 💡 使用建议

1. **新项目**: 从 `use_gated_attention=true, use_moe=false` 开始
2. **数据充足**: 尝试添加 MoE (8 experts, top-2)
3. **计算受限**: 仅使用 Gated Attention
4. **追求极致性能**: 两者都启用

**Happy Training! 🚀**
