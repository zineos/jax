#!/usr/bin/env python3
"""
方案2：构建2D NPU Mesh 完整指南

展示如何在虚拟NPU上构建和使用2D设备网格（Mesh）

核心思路：
- 虚拟NPU = JAX现有设备
- 使用JAX原生Mesh API
- 2D分片 = 数据并行 × 模型并行

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np


print("=" * 80)
print("方案2：2D NPU Mesh 构建指南")
print("=" * 80)


# ============================================================================
# 准备：获取虚拟NPU设备
# ============================================================================

def setup_virtual_npus(num_npus=8):
    """设置虚拟NPU设备"""
    devices = jax.devices()
    
    if len(devices) < num_npus:
        print(f"⚠️  系统只有 {len(devices)} 个设备，将使用全部设备")
        num_npus = len(devices)
    else:
        devices = devices[:num_npus]
    
    print(f"\n✓ 虚拟NPU设备 ({num_npus} 个):")
    for i, dev in enumerate(devices):
        print(f"  虚拟NPU-{i}: {dev}")
    
    return devices


# ============================================================================
# 方法1: 基础2D Mesh构建
# ============================================================================

def demo_basic_2d_mesh():
    """基础2D Mesh构建"""
    print("\n" + "=" * 80)
    print("【方法1】基础2D Mesh构建")
    print("=" * 80)
    
    # 获取8个虚拟NPU
    devices = setup_virtual_npus(8)
    
    if len(devices) < 4:
        print("需要至少4个设备，跳过此演示")
        return
    
    # 构建 2×2 mesh (4个设备)
    print("\n1️⃣  构建 2×2 Mesh (4个NPU)")
    print("-" * 80)
    
    # 关键：reshape设备数组为2D
    device_array = np.array(devices[:4]).reshape(2, 2)
    
    # 创建Mesh，指定轴名称
    mesh_2x2 = Mesh(
        devices=device_array,
        axis_names=('data', 'model')  # 第一维=数据并行，第二维=模型并行
    )
    
    print(f"Mesh形状: {mesh_2x2.shape}")
    print(f"轴名称: {mesh_2x2.axis_names}")
    print(f"设备布局:")
    for i in range(2):
        row = [f"NPU{j*2+i}" for j in range(2)]
        print(f"  {'  '.join(row)}")
    
    # 可视化mesh
    print(f"\nMesh详情:")
    print(f"  设备总数: {mesh_2x2.size}")
    print(f"  数据并行度: {mesh_2x2.shape['data']}")
    print(f"  模型并行度: {mesh_2x2.shape['model']}")
    
    return mesh_2x2


def demo_4x2_mesh():
    """构建 4×2 Mesh (8个设备)"""
    print("\n" + "=" * 80)
    print("【方法2】4×2 Mesh构建 (数据并行×模型并行)")
    print("=" * 80)
    
    devices = setup_virtual_npus(8)
    
    if len(devices) < 8:
        print(f"需要8个设备，当前只有{len(devices)}个，跳过")
        return
    
    # 构建 4×2 mesh
    print("\n2️⃣  构建 4×2 Mesh (8个NPU)")
    print("-" * 80)
    
    device_array = np.array(devices[:8]).reshape(4, 2)
    
    mesh_4x2 = Mesh(
        devices=device_array,
        axis_names=('data', 'model')
    )
    
    print(f"Mesh形状: {mesh_2x2.shape}")
    print(f"设备布局 (4行×2列):")
    print("  模型0   模型1")
    for i in range(4):
        print(f"  NPU{i*2}    NPU{i*2+1}  ← 数据切片{i}")
    
    print(f"\n并行配置:")
    print(f"  数据并行: {mesh_4x2.shape['data']} 路")
    print(f"  模型并行: {mesh_4x2.shape['model']} 路")
    print(f"  总并行度: {mesh_4x2.size}")
    
    return mesh_4x2


# ============================================================================
# 方法3: 使用2D Mesh进行分片
# ============================================================================

def demo_2d_sharding():
    """使用2D Mesh进行数据分片"""
    print("\n" + "=" * 80)
    print("【方法3】2D Mesh 数据分片")
    print("=" * 80)
    
    devices = setup_virtual_npus(4)
    if len(devices) < 4:
        return
    
    # 创建2×2 mesh
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('data', 'model'))
    
    print("\n3️⃣  数据分片策略")
    print("-" * 80)
    
    # 准备数据：大矩阵 (batch, features)
    batch_size, features = 256, 1024
    key = random.PRNGKey(0)
    X = random.normal(key, (batch_size, features))
    
    print(f"原始数据: X.shape = {X.shape}")
    print(f"  batch维度: {batch_size}")
    print(f"  feature维度: {features}")
    
    # 定义不同的分片策略
    shardings = {
        "无分片": NamedSharding(mesh, P(None, None)),
        "仅数据分片": NamedSharding(mesh, P('data', None)),
        "仅模型分片": NamedSharding(mesh, P(None, 'model')),
        "2D分片": NamedSharding(mesh, P('data', 'model')),
    }
    
    print("\n不同分片策略:")
    for name, sharding in shardings.items():
        X_sharded = jax.device_put(X, sharding)
        print(f"\n  {name}:")
        print(f"    PartitionSpec: {sharding.spec}")
        print(f"    设备分布: {sharding}")
        
        # 计算每个设备的数据量
        if name == "无分片":
            print(f"    每设备数据: 全部 {X.shape}")
        elif name == "仅数据分片":
            print(f"    每设备数据: ({batch_size//2}, {features})")
        elif name == "仅模型分片":
            print(f"    每设备数据: ({batch_size}, {features//2})")
        elif name == "2D分片":
            print(f"    每设备数据: ({batch_size//2}, {features//2})")


# ============================================================================
# 方法4: 2D Mesh矩阵乘法
# ============================================================================

def demo_2d_mesh_matmul():
    """使用2D Mesh进行高效矩阵乘法"""
    print("\n" + "=" * 80)
    print("【方法4】2D Mesh 矩阵乘法")
    print("=" * 80)
    
    devices = setup_virtual_npus(4)
    if len(devices) < 4:
        return
    
    # 创建2×2 mesh
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y'))
    
    print("\n4️⃣  2D分片矩阵乘法: C = A @ B")
    print("-" * 80)
    
    # 矩阵: A(m,k) @ B(k,n) = C(m,n)
    m, k, n = 512, 256, 128
    
    key = random.PRNGKey(42)
    k1, k2 = random.split(key)
    
    A = random.normal(k1, (m, k))
    B = random.normal(k2, (k, n))
    
    print(f"矩阵大小:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  C: ({m}, {n})")
    
    # 2D分片策略
    print(f"\n2D分片策略:")
    print(f"  A: 按(x, None)分片  → 每设备: ({m//2}, {k})")
    print(f"  B: 按(None, y)分片  → 每设备: ({k}, {n//2})")
    print(f"  C: 按(x, y)分片     → 每设备: ({m//2}, {n//2})")
    
    # 定义分片
    sharding_A = NamedSharding(mesh, P('x', None))
    sharding_B = NamedSharding(mesh, P(None, 'y'))
    sharding_C = NamedSharding(mesh, P('x', 'y'))
    
    # 分片数据
    A_sharded = jax.device_put(A, sharding_A)
    B_sharded = jax.device_put(B, sharding_B)
    
    # 定义计算
    @jit
    def matmul_2d(a, b):
        return jnp.matmul(a, b)
    
    # 执行
    C_sharded = matmul_2d(A_sharded, B_sharded)
    
    print(f"\n结果:")
    print(f"  C形状: {C_sharded.shape}")
    print(f"  C分片: {C_sharded.sharding}")
    
    # 验证正确性
    C_expected = jnp.matmul(A, B)
    error = jnp.max(jnp.abs(C_sharded - C_expected))
    print(f"  验证误差: {error:.2e}")
    
    print(f"\n✓ 2D分片矩阵乘法完成！")


# ============================================================================
# 方法5: 数据并行 × 模型并行 混合策略
# ============================================================================

def demo_hybrid_parallelism():
    """数据并行 × 模型并行 混合策略"""
    print("\n" + "=" * 80)
    print("【方法5】数据并行 × 模型并行（混合策略）")
    print("=" * 80)
    
    devices = setup_virtual_npus(4)
    if len(devices) < 4:
        return
    
    # 创建2×2 mesh: 2路数据并行 × 2路模型并行
    mesh = Mesh(
        np.array(devices[:4]).reshape(2, 2),
        ('data', 'model')
    )
    
    print("\n5️⃣  神经网络层: 数据并行 × 模型并行")
    print("-" * 80)
    
    # 网络配置
    batch_size = 64
    input_dim = 512
    hidden_dim = 1024
    
    print(f"网络配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    
    # 准备数据和参数
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)
    
    # 输入: (batch, input_dim)
    x = random.normal(k1, (batch_size, input_dim))
    # 权重: (input_dim, hidden_dim)
    w = random.normal(k2, (input_dim, hidden_dim))
    # 偏置: (hidden_dim,)
    b = random.normal(k3, (hidden_dim,))
    
    # 混合并行策略
    print(f"\n混合并行策略:")
    print(f"  x: P('data', None)  → 数据按batch分片")
    print(f"  w: P(None, 'model') → 权重按hidden分片")
    print(f"  b: P('model',)      → 偏置按hidden分片")
    print(f"  输出: P('data', 'model') → 2D分片")
    
    # 定义分片
    x_sharding = NamedSharding(mesh, P('data', None))
    w_sharding = NamedSharding(mesh, P(None, 'model'))
    b_sharding = NamedSharding(mesh, P('model',))
    out_sharding = NamedSharding(mesh, P('data', 'model'))
    
    # 分片数据
    x_sharded = jax.device_put(x, x_sharding)
    w_sharded = jax.device_put(w, w_sharding)
    b_sharded = jax.device_put(b, b_sharding)
    
    # 定义前向传播
    @jit
    def forward(x, w, b):
        h = jnp.matmul(x, w) + b
        return jnp.maximum(0, h)  # ReLU
    
    # 执行
    output = forward(x_sharded, w_sharded, b_sharded)
    
    print(f"\n执行结果:")
    print(f"  输出形状: {output.shape}")
    print(f"  输出分片: {output.sharding}")
    print(f"  每设备计算: ({batch_size//2}, {hidden_dim//2})")
    
    print(f"\n并行效率:")
    print(f"  数据并行: 2路 → 有效batch = {batch_size//2}/NPU")
    print(f"  模型并行: 2路 → 每NPU只需 {hidden_dim//2} 个神经元")
    print(f"  总加速比: 理论4x (2×2网格)")


# ============================================================================
# 方法6: 更复杂的Mesh配置
# ============================================================================

def demo_advanced_mesh():
    """更复杂的Mesh配置"""
    print("\n" + "=" * 80)
    print("【方法6】高级Mesh配置")
    print("=" * 80)
    
    devices = setup_virtual_npus(8)
    if len(devices) < 8:
        print("需要8个设备，跳过")
        return
    
    print("\n6️⃣  多种Mesh配置")
    print("-" * 80)
    
    # 配置1: 1×8 (纯模型并行)
    mesh_1x8 = Mesh(
        np.array(devices[:8]).reshape(1, 8),
        ('data', 'model')
    )
    print(f"\n配置1: 1×8 Mesh (纯模型并行)")
    print(f"  形状: {mesh_1x8.shape}")
    print(f"  用途: 大模型，单样本训练")
    
    # 配置2: 8×1 (纯数据并行)
    mesh_8x1 = Mesh(
        np.array(devices[:8]).reshape(8, 1),
        ('data', 'model')
    )
    print(f"\n配置2: 8×1 Mesh (纯数据并行)")
    print(f"  形状: {mesh_8x1.shape}")
    print(f"  用途: 小模型，大批次训练")
    
    # 配置3: 4×2 (平衡)
    mesh_4x2 = Mesh(
        np.array(devices[:8]).reshape(4, 2),
        ('data', 'model')
    )
    print(f"\n配置3: 4×2 Mesh (平衡)")
    print(f"  形状: {mesh_4x2.shape}")
    print(f"  用途: 中等模型，中等批次")
    
    # 配置4: 2×4 (模型为主)
    mesh_2x4 = Mesh(
        np.array(devices[:8]).reshape(2, 4),
        ('data', 'model')
    )
    print(f"\n配置4: 2×4 Mesh (模型并行为主)")
    print(f"  形状: {mesh_2x4.shape}")
    print(f"  用途: 大模型，小批次")
    
    # 配置5: 3D Mesh (高级)
    if len(devices) >= 8:
        mesh_3d = Mesh(
            np.array(devices[:8]).reshape(2, 2, 2),
            ('data', 'model', 'pipeline')
        )
        print(f"\n配置5: 2×2×2 3D Mesh (高级)")
        print(f"  形状: {mesh_3d.shape}")
        print(f"  轴: data={mesh_3d.shape['data']}, "
              f"model={mesh_3d.shape['model']}, "
              f"pipeline={mesh_3d.shape['pipeline']}")
        print(f"  用途: 超大模型，流水线并行")


# ============================================================================
# NPU特性 + 2D Mesh
# ============================================================================

def demo_npu_features_with_2d_mesh():
    """NPU特性 + 2D Mesh组合"""
    print("\n" + "=" * 80)
    print("【方法7】NPU特性 + 2D Mesh")
    print("=" * 80)
    
    devices = setup_virtual_npus(4)
    if len(devices) < 4:
        return
    
    print("\n7️⃣  组合NPU量化 + 2D分片")
    print("-" * 80)
    
    # NPU量化模拟器
    class NPUSimulator:
        @staticmethod
        def quantize_int8(x):
            scale = jnp.max(jnp.abs(x)) / 127.0
            quantized = jnp.round(x / scale).astype(jnp.int8)
            return quantized.astype(jnp.float32) * scale
        
        @staticmethod
        def npu_matmul(a, b):
            # NPU特性：量化计算
            a_q = NPUSimulator.quantize_int8(a)
            b_q = NPUSimulator.quantize_int8(b)
            return jnp.matmul(a_q, b_q)
    
    # 创建2×2 mesh
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y'))
    
    # 准备数据
    m, k, n = 512, 256, 128
    key = random.PRNGKey(99)
    k1, k2 = random.split(key)
    
    A = random.normal(k1, (m, k))
    B = random.normal(k2, (k, n))
    
    # 2D分片
    A_sharded = jax.device_put(A, NamedSharding(mesh, P('x', None)))
    B_sharded = jax.device_put(B, NamedSharding(mesh, P(None, 'y')))
    
    # NPU计算 + 2D分片
    @jit
    def npu_matmul_2d(a, b):
        return NPUSimulator.npu_matmul(a, b)
    
    result = npu_matmul_2d(A_sharded, B_sharded)
    
    print(f"NPU量化 + 2D分片:")
    print(f"  输入A: {A.shape} → 分片到 {mesh.shape['x']} 个NPU")
    print(f"  输入B: {B.shape} → 分片到 {mesh.shape['y']} 个NPU")
    print(f"  NPU特性: INT8量化")
    print(f"  分片策略: 2D并行")
    print(f"  输出: {result.shape}")
    print(f"\n✓ NPU特性与2D分片完美结合！")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    print("\n💡 核心思路:")
    print("  虚拟NPU = JAX设备")
    print("  2D Mesh = 数据并行 × 模型并行")
    print("  使用JAX原生Mesh API，无需包装！\n")
    
    try:
        # 基础演示
        demo_basic_2d_mesh()
        demo_4x2_mesh()
        demo_2d_sharding()
        
        # 计算演示
        demo_2d_mesh_matmul()
        demo_hybrid_parallelism()
        
        # 高级演示
        demo_advanced_mesh()
        demo_npu_features_with_2d_mesh()
        
        print("\n" + "=" * 80)
        print("✅ 所有演示完成！")
        print("=" * 80)
        
        print("\n📚 总结:")
        print("  1. 2×2 Mesh → 4个NPU，2D分片")
        print("  2. 4×2 Mesh → 8个NPU，数据4×模型2")
        print("  3. 数据分片 → P('data', None)")
        print("  4. 模型分片 → P(None, 'model')")
        print("  5. 2D分片 → P('data', 'model')")
        print("  6. 完全使用JAX原生API！")
        
        print("\n🎯 关键代码:")
        print("  # 构建2D mesh")
        print("  mesh = Mesh(")
        print("      np.array(devices).reshape(2, 2),")
        print("      ('data', 'model')")
        print("  )")
        print("  ")
        print("  # 2D分片")
        print("  sharding = NamedSharding(mesh, P('data', 'model'))")
        print("  data_sharded = jax.device_put(data, sharding)")
        
    except Exception as e:
        print(f"\n⚠️  需要JAX环境")
        print(f"错误: {e}")
        print("\n但原理已经清楚：")
        print("使用 np.array(devices).reshape(rows, cols) 构建2D mesh！")


if __name__ == "__main__":
    main()
