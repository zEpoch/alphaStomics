# 默认 gate_type 更新说明

## 更新时间
2024-12-08

## 更新内容

已将默认的 `gate_type` 从 `headwise` 改为 `elementwise`，以获得更强的性能。

## 修改的文件

### 1. 配置文件
- ✅ `config.yaml`: gate_type 默认值改为 `"elementwise"`

### 2. 训练脚本
- ✅ `train.py`: --gate_type 默认值改为 `'elementwise'`

### 3. 文档
- ✅ `README.MD`: 更新示例和说明
- ✅ `USAGE_GUIDE.md`: 更新推荐配置说明
- ✅ `QUICKSTART.md`: 更新快速开始示例

### 4. 示例脚本
- ✅ `examples.sh`: 更新所有示例命令

## 使用方式

### 现在的默认行为（elementwise，性能最强）

```bash
# 直接使用 --use_gated_attention，默认为 elementwise
python train.py --config config.yaml --use_gated_attention ...

# 或者明确指定
python train.py --config config.yaml --use_gated_attention --gate_type elementwise ...
```

### 如果需要使用 headwise（参数更少）

```bash
python train.py --config config.yaml --use_gated_attention --gate_type headwise ...
```

## 性能对比

| gate_type | 参数量 | 性能 | 适用场景 |
|-----------|--------|------|---------|
| **elementwise** (默认) | 更多 | **最强** | 所有场景（推荐） |
| headwise | 较少 | 良好 | 需要减少参数时 |

## 主要优势

**elementwise（新默认值）：**
- ✅ 每个位置独立门控，表达能力最强
- ✅ 能更好地学习复杂的双模态关系
- ✅ 适合高性能需求的场景

**headwise（可选）：**
- ✅ 每个注意力头共享一个门控值
- ✅ 参数量更少（~13K vs 更多）
- ✅ 适合资源受限场景

## 向后兼容

- ✅ 现有的配置文件如果已经指定了 `gate_type`，不会受影响
- ✅ 命令行参数仍然可以覆盖配置文件
- ✅ 如果需要恢复 headwise，只需添加 `--gate_type headwise`

## 验证

所有文档和示例已更新，确保：
- ✅ config.yaml 默认为 elementwise
- ✅ train.py 默认为 elementwise
- ✅ 所有示例使用 elementwise（除非特别说明）
- ✅ 文档说明 elementwise 为推荐配置

## 建议

**推荐配置（性能优先）：**
```bash
python train.py \
    --config config.yaml \
    --use_gated_attention \
    --use_moe --num_experts 8 --moe_top_k 2 \
    ...
```

**资源受限配置（参数优先）：**
```bash
python train.py \
    --config config.yaml \
    --use_gated_attention --gate_type headwise \
    --use_moe --num_experts 4 --moe_top_k 2 \
    ...
```
