# AlphaSTomics Demo 使用指南

本目录包含三个演示脚本，用于测试和对比不同的注意力机制和 MoE 配置。

## 📁 脚本说明

### 1. `demo_gated.py` - Gated Attention 完整测试
测试启用 Gated Attention 的完整训练流程。

**配置:**
- ✓ Gated Attention (headwise 门控)
- ✓ QK Normalization (RMSNorm)
- ✗ MoE

**测试内容:**
1. 前向传播测试
2. 噪声模型测试
3. Masked Diffusion 测试
4. 训练步骤测试
5. 完整训练流程测试
6. 扩散采样测试

**运行方式:**
```bash
python demo_gated.py
```

---

### 2. `demo_gated_moe.py` - Gated Attention + MoE 完整测试
测试同时启用 Gated Attention 和 MoE 的完整训练流程。

**配置:**
- ✓ Gated Attention (headwise 门控)
- ✓ QK Normalization (RMSNorm)
- ✓ MoE (4 个专家, top-2 激活)

**测试内容:**
与 `demo_gated.py` 相同，但使用 MoE 增强的 FFN。

**运行方式:**
```bash
python demo_gated_moe.py
```

**特点:**
- 总参数量增加约 19%
- 激活参数量减少约 18%（相比 baseline）
- 每个 token 只激活 2/4 的专家网络

---

### 3. `compare_configs.py` - 四种配置对比实验
对比四种不同配置的性能、参数量和训练时间。

**对比配置:**
1. **Baseline**: Linear Attention + 标准 FFN
2. **Gated Only**: Gated Attention + 标准 FFN
3. **MoE Only**: Linear Attention + MoE
4. **Gated + MoE**: Gated Attention + MoE (最强配置)

**运行方式:**
```bash
python compare_configs.py
```

**输出示例:**
```
配置                                       总参数          激活参数         训练损失         验证损失      
Baseline (Linear Attn + Standard FFN)    707,508      -            473.256      208.295
Gated Attention Only                     444,988      -            507.903      308.592
MoE Only (4 experts, top-2)              1,105,076    841,396      568.310      374.699
Gated + MoE (最强配置)                       842,556      578,876      498.796      305.334
```

---

## 🔧 配置修改

所有脚本使用类似的配置结构，可以通过修改 `DEMO_CONFIG` 或 `BASE_CONFIG` 来调整：

### 启用/禁用 Gated Attention
```python
"TransformerLayer_setting": {
    "use_gated_attention": True,  # True/False
    "gate_type": "headwise",      # "headwise" / "elementwise"
    "use_qk_norm": True,          # True/False
}
```

### 启用/禁用 MoE
```python
"TransformerLayer_setting": {
    "use_moe": True,                        # True/False
    "num_experts": 4,                       # 专家数量
    "moe_top_k": 2,                         # 每次激活的专家数
    "moe_load_balance_loss_weight": 0.01,   # 负载均衡损失权重
}
```

---

## 📊 关键发现

### 参数效率
- **Gated Attention**: 相比 baseline 参数减少 37%（移除 Linear Attention 依赖）
- **MoE**: 总参数增加 56%，但只激活 76% 的参数
- **Gated + MoE**: 总参数增加 19%，激活参数减少 18%

### 性能建议
1. **优先推荐**: Gated Attention Only
   - 参数少，训练快
   - 适合中小规模数据集
   
2. **进阶选择**: Gated + MoE
   - 最大模型容量
   - 适合大规模、复杂数据集
   - 需要更多调优

---

## 🚀 快速开始

1. 测试 Gated Attention:
```bash
python demo_gated.py
```

2. 测试 Gated Attention + MoE:
```bash
python demo_gated_moe.py
```

3. 对比所有配置:
```bash
python compare_configs.py
```

---

## 📝 注意事项

1. **数据集大小**: MoE 需要足够大的数据集才能充分利用专家网络
2. **负载均衡**: MoE 的 `load_balance_loss_weight` 需要根据数据集调优
3. **渐进式升级**: 建议先测试 Gated Attention，确认有效后再添加 MoE

---

## 📚 相关文档

- `GATED_ATTENTION.md`: Gated Attention 详细说明
- `GATED_MOE_GUIDE.md`: MoE 详细说明和使用指南
- `config.yaml`: 完整的模型配置示例
