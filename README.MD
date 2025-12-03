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

AlphaSTomics 是一个先进的空间转录组学数据建模框架，基于扩散模型和 Transformer 架构。支持**双模态扩散**，可以同时对基因表达量和空间坐标进行生成与重建。

## 核心特性（保留）
- 双模态生成：表达量 + 3D 坐标
- Masked Diffusion（特征级 masking）
- 训练模式：`expr_to_pos` / `pos_to_expr` / `joint`
- 损失：表达量 MSE + 位置距离矩阵 MSE（旋转/平移不变）
- 高效注意力：Linear Attention，近 O(n)
- 分布式训练：支持 DDP / FSDP / DeepSpeed



---

## **功能一：双模态生成**
- **说明**: 同时建模基因表达（expression, (B,N,G)）和 3D 坐标（position, (B,N,3)）。

- **示例**:

```python
from diffusion_model import DiffusionSampler
# sampler.sample 返回 (expression, positions)
expression, positions = sampler.sample(expr, pos, mask, mode='joint')
```

---

## **功能二：Masked Diffusion (MDT)**
- **说明**: 对每个细胞内部的特征维度（基因或 x/y/z）进行 mask，重建损失仅在 mask 上计算。

- **示例（启用）**:

```python
# config['masking'] 控制 MDT
cfg['masking'] = {'enable': True, 'expression_mask_ratio':0.4}
model = AlphaSTomicsModule(cfg, num_genes)
```

---

## **功能三：训练模式**
- **说明**: 支持 `expr_to_pos`, `pos_to_expr`, `joint` 三种模式。

- **示例**:

```bash
python -m alphastomics.main train --config config.yaml --data_mode slice
```

或在代码中：

```python
cfg['training_mode'] = 'expr_to_pos'
```

---

## **功能四：损失设计**
- **说明**: 表达量使用 MSE；位置使用距离矩阵 MSE（旋转/平移不变）；可选 masked reconstruction 加权。

- **示例**:

```python
loss = DualModalLoss(lambda_expression=1.0, lambda_position=1.0, lambda_masked_reconstruction=0.5)
L = loss(pred_expr, pred_pos, true_expr, true_pos, mask_info)
```

---

## **功能五：高效注意力与可扩展性**
- **说明**: 使用 Linear Attention，时空复杂度近 O(n)，便于处理大序列或高细胞数。

---

## **功能六：采样器接口**
- **说明**: `diffusion_model/sample.py` 提供 `expr_to_pos` / `pos_to_expr` / `joint`。

- **示例**:

```python
pos = DiffusionSampler(model, noise_model).sample(expr, pos, mask, mode='expr_to_pos')[1]
```

---

## **功能七：数据与部署**
- **说明**: `utils/dataholder.py` 提供统一数据容器；支持 `slice` 和 `cell` 两种加载模式；基于 PyTorch Lightning 支持 DDP/FSDP/DeepSpeed。

---

## **快速安装与本地验证**

```bash
# macOS / zsh (项目根目录)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytorch-lightning pyyaml wandb
pip install -e .

# 快速导入测试
python -c "from diffusion_model import MaskedDiffusionModule; print('OK')"
```

---

更多配置细节见 `config.yaml`，MDT 使用说明见 `docs/MASKED_DIFFUSION.md`。

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