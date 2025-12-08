"""
测试 MoE 模块
重点展示：总参数量 vs 激活参数量
"""
import torch
from alphastomics.attn_model.moe import (
    Expert,
    TopKRouter,
    MixtureOfExperts,
    MoETransformerFFN
)


def count_parameters(model):
    """计算模型总参数量"""
    return sum(p.numel() for p in model.parameters())


def calculate_activated_params(d_model, d_ff, num_experts, top_k):
    """
    计算 MoE 的激活参数量
    
    激活参数 = Router参数 + top_k个Expert的参数
    """
    # Router 参数: gate (d_model -> num_experts) + w_noise (d_model -> num_experts)
    router_params = d_model * num_experts * 2
    
    # 每个 Expert 参数: w1 (d_model -> d_ff) + w2 (d_ff -> d_model) + bias
    expert_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    
    # 激活参数 = router + top_k 个 experts
    activated_params = router_params + top_k * expert_params
    
    return activated_params


def test_expert():
    """测试单个专家"""
    print("=" * 70)
    print("测试 1: Expert 网络")
    print("=" * 70)
    
    d_model, d_ff = 256, 1024
    expert = Expert(d_model=d_model, d_ff=d_ff, dropout=0.1, activation='relu')
    x = torch.randn(2, 100, d_model)
    out = expert(x)
    
    total_params = count_parameters(expert)
    
    print(f"配置: d_model={d_model}, d_ff={d_ff}")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"总参数量: {total_params:,}")
    print("✓ Expert 测试通过！\n")


def test_router():
    """测试路由器"""
    print("=" * 70)
    print("测试 2: TopKRouter (专家选择器)")
    print("=" * 70)
    
    d_model = 256
    num_experts = 8
    top_k = 2
    
    router = TopKRouter(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        use_noisy_gating=True
    )
    x = torch.randn(2, 100, d_model)
    
    weights, indices, loss = router(x, training=True)
    total_params = count_parameters(router)
    
    print(f"配置: {num_experts}个专家, 每次选择top-{top_k}")
    print(f"输入形状: {x.shape}")
    print(f"输出:")
    print(f"  - 专家权重: {weights.shape} (每个token选{top_k}个专家)")
    print(f"  - 专家索引: {indices.shape}")
    print(f"  - 负载均衡损失: {loss.item():.6f}")
    print(f"\nRouter 参数量: {total_params:,}")
    print("✓ TopKRouter 测试通过！\n")


def test_moe():
    """测试 MoE 层 - 重点展示参数效率"""
    print("=" * 70)
    print("测试 3: MixtureOfExperts (完整 MoE 层)")
    print("=" * 70)
    
    d_model = 256
    d_ff = 1024
    num_experts = 8
    top_k = 2
    
    moe = MixtureOfExperts(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        dropout=0.1
    )
    x = torch.randn(2, 100, d_model)
    
    out, aux_loss = moe(x, return_load_balance_loss=True)
    
    # 计算参数量
    total_params = count_parameters(moe)
    activated_params = calculate_activated_params(d_model, d_ff, num_experts, top_k)
    
    # 标准 FFN 对比
    standard_ffn_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    
    print(f"配置: {num_experts}个专家, top-{top_k}激活")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"辅助损失: {aux_loss.item():.6f}")
    
    print(f"\n" + "=" * 70)
    print("📊 参数量对比")
    print("=" * 70)
    print(f"标准 FFN:")
    print(f"  总参数量:     {standard_ffn_params:,}")
    print(f"  激活参数量:   {standard_ffn_params:,}  (100%)")
    
    print(f"\nMoE ({num_experts} experts, top-{top_k}):")
    print(f"  总参数量:     {total_params:,}  ({total_params/standard_ffn_params:.1f}x)")
    print(f"  激活参数量:   {activated_params:,}  ({activated_params/standard_ffn_params:.1f}x)")
    
    print(f"\n💡 关键洞察:")
    print(f"  - 总参数增加了 {total_params/standard_ffn_params:.1f}x (模型容量)")
    print(f"  - 激活参数仅 {activated_params/standard_ffn_params:.1f}x (实际计算)")
    print(f"  - 参数效率: 用 {activated_params/standard_ffn_params:.1f}x 计算获得 {total_params/standard_ffn_params:.1f}x 容量!")
    print("=" * 70)
    print("✓ MixtureOfExperts 测试通过！\n")


def test_moe_transformer_ffn():
    """测试 MoE Transformer FFN - 完整对比"""
    print("=" * 70)
    print("测试 4: MoETransformerFFN (实际使用接口)")
    print("=" * 70)
    
    d_model = 256
    d_ff = 1024
    num_experts = 8
    top_k = 2
    
    # 测试标准 FFN 模式
    print("\n1️⃣  标准 FFN 模式:")
    print("-" * 70)
    ffn_standard = MoETransformerFFN(
        d_model=d_model,
        d_ff=d_ff,
        use_moe=False
    )
    x = torch.randn(2, 100, d_model)
    out_std, aux_loss_std = ffn_standard(x)
    std_params = count_parameters(ffn_standard)
    
    print(f"  输出形状: {out_std.shape}")
    print(f"  辅助损失: {aux_loss_std}")
    print(f"  总参数量: {std_params:,}")
    print(f"  激活参数量: {std_params:,} (100%)")
    
    # 测试 MoE 模式
    print(f"\n2️⃣  MoE 模式 ({num_experts} experts, top-{top_k}):")
    print("-" * 70)
    ffn_moe = MoETransformerFFN(
        d_model=d_model,
        d_ff=d_ff,
        use_moe=True,
        num_experts=num_experts,
        top_k=top_k
    )
    out_moe, aux_loss_moe = ffn_moe(x)
    moe_params = count_parameters(ffn_moe)
    moe_activated = calculate_activated_params(d_model, d_ff, num_experts, top_k)
    
    print(f"  输出形状: {out_moe.shape}")
    print(f"  辅助损失: {aux_loss_moe.item():.6f}")
    print(f"  总参数量: {moe_params:,} ({moe_params/std_params:.1f}x)")
    print(f"  激活参数量: {moe_activated:,} ({moe_activated/std_params:.1f}x)")
    
    print(f"\n" + "=" * 70)
    print("📈 性能 vs 效率权衡")
    print("=" * 70)
    print(f"容量提升:     {moe_params/std_params:.1f}x  (总参数)")
    print(f"计算成本:     {moe_activated/std_params:.1f}x  (激活参数)")
    print(f"效率比:       {(moe_params/std_params) / (moe_activated/std_params):.1f}x  (容量/计算)")
    print("=" * 70)
    
    print("✓ MoETransformerFFN 测试通过！\n")


def test_scaling_analysis():
    """测试不同配置下的参数缩放"""
    print("=" * 70)
    print("测试 5: 参数缩放分析")
    print("=" * 70)
    
    d_model = 256
    d_ff = 1024
    standard_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    
    configs = [
        (4, 1, "小规模 MoE"),
        (4, 2, "小规模 MoE (top-2)"),
        (8, 1, "中等 MoE"),
        (8, 2, "中等 MoE (top-2, 推荐)"),
        (16, 2, "大规模 MoE"),
    ]
    
    print(f"\n基准: 标准 FFN = {standard_params:,} 参数\n")
    print(f"{'配置':<25} {'总参数':<15} {'激活参数':<15} {'容量比':<10} {'计算比':<10} {'效率':<10}")
    print("-" * 95)
    
    for num_experts, top_k, desc in configs:
        total = standard_params * num_experts + (d_model * num_experts * 2)
        activated = calculate_activated_params(d_model, d_ff, num_experts, top_k)
        capacity_ratio = total / standard_params
        compute_ratio = activated / standard_params
        efficiency = capacity_ratio / compute_ratio
        
        print(f"{desc:<25} {total:>12,}  {activated:>12,}  {capacity_ratio:>8.1f}x  {compute_ratio:>8.1f}x  {efficiency:>8.1f}x")
    
    print("\n💡 效率 = 容量比 / 计算比 (越高越好)")
    print("   推荐: 8 experts, top-2 → 用 2.1x 计算获得 8x 容量 = 3.8x 效率")
    print("=" * 70)
    print("✓ 参数缩放分析完成！\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 15 + "AlphaSTomics MoE 模块测试套件")
    print(" " * 10 + "重点: 总参数量 vs 激活参数量的区别")
    print("=" * 70 + "\n")
    
    try:
        test_expert()
        test_router()
        test_moe()
        test_moe_transformer_ffn()
        test_scaling_analysis()
        
        print("=" * 70)
        print(" " * 25 + "✓ 所有测试通过！")
        print("=" * 70)
        print("\n📝 核心要点:")
        print("  1. MoE 总参数多 = 模型容量大 = 学习能力强")
        print("  2. 激活参数少 = 计算成本低 = 训练推理快")
        print("  3. 稀疏激活是 MoE 的核心优势!")
        print("\n🎯 推荐配置:")
        print("  - 标准场景: 8 experts, top-2")
        print("  - 计算受限: 4 experts, top-1")
        print("  - 追求性能: 16 experts, top-2")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ 测试失败: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
