# AlphaSTomics 快速开始指南

## 📦 安装

```bash
# 1. 克隆仓库
git clone https://github.com/zEpoch/alphaStomics.git
cd alphaStomics

# 2. 创建环境
conda create -n alphastomics python=3.12
conda activate alphastomics

# 3. 安装依赖
pip install torch pytorch-lightning pyyaml wandb scanpy anndata pyarrow
pip install -e .
```

## 🚀 5 分钟快速测试

### 1. 测试 Gated Attention

```bash
python demo_gated.py
```

预期输出：
```
✅ 前向传播成功!
✅ 噪声模型测试成功!
✅ Masked Diffusion 测试成功!
✅ 训练步骤成功!
✅ 完整训练流程成功!
✅ 扩散采样成功!

🎉 所有测试通过!
```

### 2. 测试 Gated Attention + MoE

```bash
python demo_gated_moe.py
```

### 3. 对比四种配置

```bash
python compare_configs.py
```

输出示例：
```
配置                                       总参数          激活参数         训练损失         验证损失      
Baseline (Linear Attn + Standard FFN)    707,508      -            473.256      208.295
Gated Attention Only                     444,988      -            507.903      308.592
MoE Only (4 experts, top-2)              710,324      644,020      516.622      306.972
Gated + MoE (最强配置)                       447,804      381,500      519.657      321.556
```

## 📚 准备真实数据

### 方案 A：从 h5ad 文件开始

```bash
# 假设你有以下数据
data/
├── slice_01.h5ad
├── slice_02.h5ad
└── slice_03.h5ad

# TODO: 实现数据预处理脚本
# python preprocess_data.py --input_dir data/ --output_dir processed/
```

### 方案 B：使用 Demo 数据测试

Demo 脚本会自动生成模拟数据，你可以直接运行训练。

## 🎯 开始训练

### 基础训练（推荐配置）

```bash
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --use_gated_attention \
    --batch_size 32 \
    --epochs 50 \
    --exp_name my_first_experiment
```

### 高级训练（Gated + MoE）

```bash
python train.py \
    --config config.yaml \
    --data_dir ./processed_data \
    --num_genes 2000 \
    --use_gated_attention \
    --use_moe \
    --num_experts 4 \
    --moe_top_k 2 \
    --enable_masking \
    --batch_size 32 \
    --epochs 100 \
    --gpus 2 \
    --fp16 \
    --exp_name gated_moe_experiment
```

## 📊 监控训练

训练输出会保存到 `./outputs/<exp_name>/`：

```
outputs/my_first_experiment/
├── checkpoints/
│   ├── alphastomics-epoch=00-val_loss=0.5000.ckpt
│   ├── alphastomics-epoch=01-val_loss=0.4500.ckpt
│   └── last.ckpt
├── logs/
│   └── events.out.tfevents...
└── config.yaml
```

使用 TensorBoard 查看：
```bash
tensorboard --logdir outputs/my_first_experiment/logs
```

## 🔧 配置选择建议

### 小型数据集（< 1M 细胞）
```bash
--use_gated_attention  # elementwise 默认，性能最强
```
- 性能最强
- 表达能力最好
- 适合各种数据集大小

### 中型数据集（1M - 10M 细胞）
```bash
--use_gated_attention --use_moe --num_experts 4 --moe_top_k 2
```
- 参数适中
- 模型容量大
- 性能最优

### 大型数据集（> 10M 细胞）
```bash
--use_gated_attention --use_moe --num_experts 8 --moe_top_k 2
```
- 最大模型容量
- 稀疏激活（25%）
- 适合复杂数据

## ❓ 常见问题

### Q1: 如何选择 num_experts 和 moe_top_k？

**推荐配置：**
- 小模型：4 专家，激活 2 个（50% 激活）
- 中模型：8 专家，激活 2 个（25% 激活）
- 大模型：16 专家，激活 2 个（12.5% 激活）

**原则：**
- `num_experts` 越大，模型容量越大，但需要更多数据
- `moe_top_k` 通常设为 1 或 2
- 激活比例 = top_k / num_experts

### Q2: Gated Attention 和 MoE 可以单独使用吗？

可以！

- **仅 Gated Attention**（推荐新手）：`--use_gated_attention`
- **仅 MoE**：`--use_moe --num_experts 4 --moe_top_k 2`
- **两者结合**（最强）：同时启用

### Q3: 训练时显存不足怎么办？

```bash
# 1. 减小 batch size
--batch_size 16

# 2. 启用梯度累积（TODO: 需要在 train.py 中添加）
# --accumulate_grad_batches 2

# 3. 使用 FP16 混合精度
--fp16

# 4. 减少专家数量
--num_experts 4  # 而不是 8
```

### Q4: 如何恢复训练？

```bash
python train.py \
    --config config.yaml \
    --resume_from outputs/my_experiment/checkpoints/last.ckpt \
    ...其他参数...
```

## 📖 下一步

1. 阅读详细文档：
   - [GATED_ATTENTION.md](GATED_ATTENTION.md)
   - [GATED_MOE_GUIDE.md](GATED_MOE_GUIDE.md)
   - [DEMO_USAGE.md](DEMO_USAGE.md)

2. 准备自己的数据

3. 调优超参数

4. 评估模型性能

## 💬 获取帮助

- 查看 [README.MD](README.MD) 了解完整功能
- 运行 `python train.py --help` 查看所有参数
- 提交 Issue：https://github.com/zEpoch/alphaStomics/issues
