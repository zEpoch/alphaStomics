# AlphaSTomics - 使用指南

## 📋 目录

1. [快速开始](#快速开始)
2. [训练方式对比](#训练方式对比)
3. [高级特性](#高级特性)
4. [配置说明](#配置说明)
5. [工具脚本](#工具脚本)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository_url>
cd alphaStomics

# 安装依赖
pip install torch pytorch-lightning pyyaml wandb scanpy anndata pyarrow
pip install -e .
```

### 验证安装

```bash
python -c "from alphastomics.diffusion_model import MaskedDiffusionModule; print('✓ 安装成功')"
```

### 第一次训练

```bash
# 使用独立脚本
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --exp_name my_first_experiment

# 或使用模块化接口
python -m alphastomics.main train \
    --config config.yaml \
    --data_dir ./processed_data
```

---

## 📊 训练方式对比

AlphaSTomics 提供两种训练方式，功能完全相同，可根据使用场景选择：

### 方式 1: `train.py` （独立脚本）

**优势：**
- ✅ 简单直观，参数清晰
- ✅ 适合快速实验和原型开发
- ✅ 输出结构清晰（`outputs/<exp_name>/`）
- ✅ 需要指定实验名称，便于管理

**使用场景：**
- 研究和实验
- 参数调优
- 快速测试新想法

**示例：**
```bash
python train.py \
    --config config.yaml \
    --data_dir ./data \
    --num_genes 2000 \
    --use_gated_attention \
    --use_moe --num_experts 4 --moe_top_k 2 \
    --exp_name my_experiment
```

### 方式 2: `python -m alphastomics.main train` （模块化）

**优势：**
- ✅ 集成到完整流水线中
- ✅ 支持多个子命令（preprocess, train, test, sample）
- ✅ 适合生产环境
- ✅ 更好的代码组织

**使用场景：**
- 生产环境部署
- 自动化流水线
- 集成到更大的系统中

**示例：**
```bash
# 预处理
python -m alphastomics.main preprocess --raw_dir ./raw --output_dir ./processed

# 训练
python -m alphastomics.main train \
    --config config.yaml \
    --use_gated_attention \
    --use_moe --num_experts 4 --moe_top_k 2

# 测试
python -m alphastomics.main test --config config.yaml --checkpoint best.ckpt

# 采样
python -m alphastomics.main sample --checkpoint best.ckpt --num_samples 1000
```

### 参数对比

| 特性 | train.py | main.py train |
|------|----------|--------------|
| Gated Attention | ✅ | ✅ |
| MoE | ✅ | ✅ |
| Masked Diffusion | ✅ | ✅ |
| 自定义实验名 | 必需 `--exp_name` | 从配置推断 |
| 输出目录 | `outputs/<exp_name>/` | 配置文件指定 |
| 子命令支持 | ❌ 仅训练 | ✅ 完整流水线 |
| 参数覆盖 | ✅ | ✅ |

---

## 🎯 高级特性

### 1. Gated Attention

**作用：** 通过门控机制调节注意力强度，提升模型表达能力。

**参数：**
- `--use_gated_attention`: 启用 Gated Attention
- `--gate_type {headwise,elementwise}`: 门控类型
  - `headwise`: 每个注意力头共享一个门控值（参数更少）
  - `elementwise`: 每个位置独立门控值（表达能力更强）

**示例：**
```bash
# Elementwise gating（默认，性能最强）
python train.py --config config.yaml --use_gated_attention ...

# Headwise gating（参数少）
python train.py --config config.yaml --use_gated_attention --gate_type headwise ...
```

**参数影响：**
- Elementwise（默认）: 增加更多参数，但性能最强
- Headwise: 增加 ~13K 参数（约 1.9%），参数更少

### 2. Mixture of Experts (MoE)

**作用：** 使用多个专家网络，每次只激活部分专家，提升参数效率。

**参数：**
- `--use_moe`: 启用 MoE
- `--num_experts <int>`: 专家总数（推荐：4, 8, 16）
- `--moe_top_k <int>`: 每次激活的专家数（推荐：2）

**示例：**
```bash
# 4 选 2
python train.py --config config.yaml --use_moe --num_experts 4 --moe_top_k 2 ...

# 8 选 2
python train.py --config config.yaml --use_moe --num_experts 8 --moe_top_k 2 ...
```

**参数效率：**

| 配置 | 总参数 | 激活参数 | 激活率 |
|------|--------|---------|--------|
| Dense | 695K | 695K | 100% |
| MoE 4选2 | 710K | 644K | 90.7% |
| MoE 8选2 | 725K | 605K | 83.5% |

**关键见解：**
- 总参数量略增（~2%），因为需要路由网络
- 激活参数显著减少（4选2 约 50%）
- FFN 层激活率 = top_k / num_experts
- 整体模型激活率较高，因为其他层（embedding, attention）100% 激活

### 3. Masked Diffusion

**作用：** 在训练时随机遮罩部分特征，增强模型鲁棒性。

**参数：**
- `--enable_masking`: 启用遮罩
- `--expression_mask_ratio <float>`: 表达量遮罩比例（0.0-1.0）
- `--position_mask_ratio <float>`: 位置遮罩比例（0.0-1.0）

**示例：**
```bash
python train.py \
    --config config.yaml \
    --enable_masking \
    --expression_mask_ratio 0.3 \
    --position_mask_ratio 0.33 \
    ...
```

### 4. 组合使用

**最强配置：** Gated Attention + MoE + Masked Diffusion

```bash
python train.py \
    --config config.yaml \
    --data_dir ./data \
    --num_genes 2000 \
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
    --lr 1e-4 \
    --exp_name full_featured
```

---

## ⚙️ 配置说明

### 配置文件优先级

**命令行参数 > config.yaml**

这意味着你可以：
1. 在 `config.yaml` 中设置默认值
2. 用命令行参数快速实验不同配置
3. 不需要修改配置文件

### config.yaml 结构

```yaml
# 模型架构
TransformerLayer_setting:
  d_model: 256
  num_heads: 8
  d_ff: 2048
  use_gated_attention: false  # 可被 --use_gated_attention 覆盖
  gate_type: headwise          # 可被 --gate_type 覆盖
  use_moe: false               # 可被 --use_moe 覆盖
  num_experts: 8               # 可被 --num_experts 覆盖
  moe_top_k: 2                 # 可被 --moe_top_k 覆盖

# 扩散设置
diffusion:
  diffusion_steps: 1000

# 训练设置
training:
  batch_size: 64               # 可被 --batch_size 覆盖
  epochs: 100                  # 可被 --epochs 覆盖
  learning_rate: 1e-4          # 可被 --lr 覆盖

# 遮罩设置
masking:
  enable: false                # 可被 --enable_masking 覆盖
  expression_mask_ratio: 0.3   # 可被 --expression_mask_ratio 覆盖
  position_mask_ratio: 0.33    # 可被 --position_mask_ratio 覆盖

# 训练模式
training_mode: joint           # 可被 --training_mode 覆盖
```

### 常用参数组合

```bash
# 研究配置（小模型，快速迭代）
python train.py --config config.yaml --batch_size 128 --epochs 50 --lr 5e-4

# 生产配置（大模型，充分训练）
python train.py --config config.yaml --batch_size 32 --epochs 200 --lr 1e-4

# 参数高效配置（MoE）
python train.py --config config.yaml --use_moe --num_experts 8 --moe_top_k 2

# 表达能力优先（Gated + Dense）
python train.py --config config.yaml --use_gated_attention --gate_type elementwise
```

---

## 🛠️ 工具脚本

### 1. compare_configs.py

比较不同配置的参数量：

```bash
python compare_configs.py
```

输出：
- 4 种配置（Baseline, Gated, MoE, Gated+MoE）
- 总参数量、激活参数量、激活率
- FFN 层激活率 vs 整体模型激活率

### 2. examples.sh

交互式示例脚本：

```bash
# 交互模式
./examples.sh

# 直接运行某个示例
./examples.sh 1          # 基础训练
./examples.sh gated      # Gated Attention
./examples.sh moe        # MoE
./examples.sh combined   # Gated + MoE
./examples.sh full       # 完整配置
./examples.sh compare    # 比较配置

# 运行所有示例
./examples.sh all
```

---

## ❓ 常见问题

### Q1: 什么时候用 train.py，什么时候用 main.py？

**A:** 
- **研究/实验**: 用 `train.py`，参数直观，输出清晰
- **生产/流水线**: 用 `python -m alphastomics.main`，支持完整流程

### Q2: MoE 真的能减少计算吗？

**A:** 
是的！虽然总参数量略增（~2%），但每次前向传播只激活 top_k 个专家：
- 4 选 2: FFN 计算量减少 50%
- 8 选 2: FFN 计算量减少 75%
- 整体模型计算量减少约 10-20%（取决于 FFN 在模型中的占比）

### Q3: Gated Attention 和普通 Attention 有什么区别？

**A:**
- **普通 Attention**: `output = attention(Q, K, V)`
- **Gated Attention**: `output = gate * attention(Q, K, V)`
- 门控值是可学习的，可以动态调节注意力强度
- Headwise: 每个头一个门（参数少）
- Elementwise: 每个位置一个门（表达能力强）

### Q4: 如何选择 num_experts 和 moe_top_k？

**A:**
推荐配置：
- **4 选 2**: 轻量级，适合小数据集
- **8 选 2**: 标准配置，平衡性能和效率
- **16 选 4**: 大规模模型，需要更多数据

原则：
- `top_k` 通常是 `num_experts` 的 1/4 到 1/2
- 更多专家 = 更高表达能力，但需要更多数据避免过拟合

### Q5: 命令行参数是否会被保存？

**A:**
使用 `train.py` 时，所有有效配置会被保存到 `outputs/<exp_name>/config.yaml`，包括：
- 配置文件的原始值
- 命令行覆盖的值
- 方便后续复现实验

### Q6: 如何恢复训练？

**A:**
```bash
# train.py
python train.py --config config.yaml --resume outputs/my_exp/checkpoints/last.ckpt ...

# main.py
python -m alphastomics.main train --config config.yaml --resume path/to/checkpoint.ckpt
```

### Q7: 如何监控训练？

**A:**
AlphaSTomics 使用 Weights & Biases (wandb) 进行实验跟踪：
```bash
# 登录 wandb
wandb login

# 训练时自动上传日志
python train.py --config config.yaml ...

# 访问 https://wandb.ai 查看实验
```

### Q8: 不同训练模式的区别？

**A:**
- `joint`: 同时训练表达量→位置 和 位置→表达量
- `expr_to_pos`: 只训练 表达量→位置
- `pos_to_expr`: 只训练 位置→表达量

推荐使用 `joint`，除非有特定需求。

---

## 📚 更多资源

- **详细文档**: `README.MD`
- **快速入门**: `QUICKSTART.md`
- **更新日志**: `CHANGELOG_MoE_Fix.md`
- **配置说明**: `config.yaml`（内含详细注释）
- **参数比较**: 运行 `python compare_configs.py`
- **示例脚本**: `./examples.sh`

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

请参考 `LICENSE` 文件。
