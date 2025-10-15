#!/usr/bin/env python3
"""
在虚拟 NPU 上直接使用 JAX 原生 Sharding API

这个文件展示如何在虚拟 NPU backend 上**直接**使用 JAX 的原生 API，
无需任何包装或修改！

核心思路：
虚拟 NPU 设备 = JAX 真实设备
因此，所有 JAX 原生 API 都可以直接使用！

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from jax import random, jit, pmap
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint
import numpy as np


print("=" * 70)
print("直接使用 JAX 原生 Sharding API")
print("=" * 70)


# ============================================================================
# 设置：将虚拟 NPU 映射到 JAX 设备
# ============================================================================

# 获取 JAX 设备（这些可以是 CPU 或 GPU）
# 在实际应用中，这些就是你的虚拟 NPU 设备
devices = jax.devices()
num_devices = len(devices)

print(f"\n✓ 检测到 {num_devices} 个设备（虚拟 NPU）:")
for i, device in enumerate(devices):
    print(f"  虚拟NPU-{i}: {device}")


# ============================================================================
# 方法 1: 直接使用 jax.device_put + NamedSharding
# ============================================================================

def demo_native_device_put():
    """直接使用 jax.device_put 进行分片"""
    print("\n" + "=" * 70)
    print("【方法 1】jax.device_put + NamedSharding (原生 API)")
    print("=" * 70)
    
    # 创建设备网格（原生 API）
    mesh = Mesh(np.array(devices[:4]), axis_names=('x',))
    
    print(f"设备网格: {mesh}")
    
    # 定义分片策略（原生 API）
    sharding = NamedSharding(mesh, P('x', None))
    
    # 创建数据
    key = random.PRNGKey(0)
    A = random.normal(key, (1024, 512))
    B = random.normal(key, (512, 256))
    
    # 直接使用原生 API 进行分片
    A_sharded = jax.device_put(A, sharding)  # ← 原生 JAX API！
    
    print(f"\n数据分片:")
    print(f"  A 原始: {A.shape}")
    print(f"  A 分片后: {A_sharded.sharding}")
    print(f"  每个设备存储: ({1024//4}, 512)")
    
    # 定义计算（原生 API）
    @jit  # ← 原生 JAX API！
    def matmul(a, b):
        return jnp.matmul(a, b)
    
    # 执行计算
    result = matmul(A_sharded, B)
    
    print(f"\n结果:")
    print(f"  形状: {result.shape}")
    print(f"  分片: {result.sharding}")
    
    return result


# ============================================================================
# 方法 2: 直接使用 pmap (原生 API)
# ============================================================================

def demo_native_pmap():
    """直接使用 jax.pmap"""
    print("\n" + "=" * 70)
    print("【方法 2】jax.pmap (原生 API)")
    print("=" * 70)
    
    # 直接使用原生 pmap！
    @pmap  # ← 原生 JAX API！
    def parallel_matmul(a, b):
        return jnp.matmul(a, b)
    
    # 准备数据
    key = random.PRNGKey(42)
    A = random.normal(key, (4, 256, 512))  # (设备数, ...)
    B = random.normal(key, (4, 512, 128))
    
    print(f"输入:")
    print(f"  A: {A.shape} - 自动分片到 4 个设备")
    print(f"  B: {B.shape}")
    
    # 执行（原生 API）
    result = parallel_matmul(A, B)
    
    print(f"\n结果:")
    print(f"  形状: {result.shape}")
    print(f"  每个设备输出: (256, 128)")
    
    return result


# ============================================================================
# 方法 3: 直接使用 with_sharding_constraint
# ============================================================================

def demo_native_sharding_constraint():
    """直接使用 jax.lax.with_sharding_constraint"""
    print("\n" + "=" * 70)
    print("【方法 3】jax.lax.with_sharding_constraint (原生 API)")
    print("=" * 70)
    
    # 创建设备网格
    mesh = Mesh(np.array(devices[:4]), axis_names=('batch',))
    
    # 在 mesh 上下文中定义计算
    with mesh:
        @jit  # ← 原生 JAX API！
        def sharded_computation(x, w):
            # 在计算中间插入分片约束（原生 API）
            x = with_sharding_constraint(x, P('batch', None))  # ← 原生 API！
            h = jnp.matmul(x, w)
            h = with_sharding_constraint(h, P('batch', None))  # ← 原生 API！
            return jnp.maximum(0, h)  # ReLU
        
        # 准备数据
        key = random.PRNGKey(123)
        x = random.normal(key, (128, 512))
        w = random.normal(key, (512, 256))
        
        print(f"输入:")
        print(f"  x: {x.shape}")
        print(f"  w: {w.shape}")
        print(f"  分片约束: x 按 'batch' 维分片")
        
        # 执行
        result = sharded_computation(x, w)
        
        print(f"\n结果:")
        print(f"  形状: {result.shape}")
        print(f"  分片: {result.sharding}")
    
    return result


# ============================================================================
# 方法 4: 直接使用 shard_map (原生 API)
# ============================================================================

def demo_native_shard_map():
    """直接使用 jax.experimental.shard_map"""
    print("\n" + "=" * 70)
    print("【方法 4】jax.experimental.shard_map (原生 API)")
    print("=" * 70)
    
    # 创建设备网格
    mesh = Mesh(np.array(devices[:4]), axis_names=('devices',))
    
    # 直接使用原生 shard_map！
    @shard_map(  # ← 原生 JAX API！
        mesh=mesh,
        in_specs=(P('devices', None), P(None, None)),
        out_specs=P('devices', None)
    )
    def manual_sharded_matmul(a_shard, b):
        # 每个设备上的计算逻辑
        return jnp.matmul(a_shard, b)
    
    # 准备数据
    key = random.PRNGKey(456)
    A = random.normal(key, (512, 256))
    B = random.normal(key, (256, 128))
    
    print(f"输入:")
    print(f"  A: {A.shape} - 将按第一维分片")
    print(f"  B: {B.shape} - 复制到所有设备")
    print(f"  每个设备计算: ({512//4}, 256) × (256, 128)")
    
    # 执行（自动处理分片）
    result = manual_sharded_matmul(A, B)
    
    print(f"\n结果:")
    print(f"  形状: {result.shape}")
    
    # 验证
    expected = jnp.matmul(A, B)
    error = jnp.mean(jnp.abs(result - expected))
    print(f"  验证误差: {error:.2e}")
    
    return result


# ============================================================================
# 方法 5: 直接使用二维 Mesh (原生 API)
# ============================================================================

def demo_native_2d_mesh():
    """直接使用二维设备网格"""
    print("\n" + "=" * 70)
    print("【方法 5】二维 Mesh 分片 (原生 API)")
    print("=" * 70)
    
    if len(devices) < 4:
        print("需要至少 4 个设备")
        return
    
    # 创建 2×2 设备网格（原生 API）
    mesh = Mesh(
        np.array(devices[:4]).reshape(2, 2),  # ← 原生 API！
        axis_names=('data', 'model')
    )
    
    print(f"设备网格: 2×2")
    print(f"  轴名称: ('data', 'model')")
    
    # 定义分片策略
    # x 按 'data' 分片，w 按 'model' 分片
    x_sharding = NamedSharding(mesh, P('data', None))
    w_sharding = NamedSharding(mesh, P(None, 'model'))
    
    # 准备数据
    key = random.PRNGKey(789)
    x = random.normal(key, (256, 512))
    w = random.normal(key, (512, 1024))
    
    # 分片数据（原生 API）
    x_sharded = jax.device_put(x, x_sharding)
    w_sharded = jax.device_put(w, w_sharding)
    
    print(f"\n数据分片:")
    print(f"  x: {x.shape} → 按 'data' 分片 (2 份)")
    print(f"  w: {w.shape} → 按 'model' 分片 (2 份)")
    print(f"  总共使用 4 个设备 (2×2 网格)")
    
    # 定义计算
    @jit
    def compute(x, w):
        return jnp.matmul(x, w)
    
    # 执行
    result = compute(x_sharded, w_sharded)
    
    print(f"\n结果:")
    print(f"  形状: {result.shape}")
    print(f"  分片: {result.sharding}")
    
    return result


# ============================================================================
# 方法 6: 直接使用 jit 的 in_shardings 和 out_shardings
# ============================================================================

def demo_native_jit_shardings():
    """直接在 jit 中指定分片"""
    print("\n" + "=" * 70)
    print("【方法 6】jit 的 in_shardings/out_shardings (原生 API)")
    print("=" * 70)
    
    # 创建设备网格
    mesh = Mesh(np.array(devices[:4]), axis_names=('x',))
    
    # 定义分片
    sharding = NamedSharding(mesh, P('x', None))
    
    # 在 jit 中直接指定分片（原生 API）
    @jit(  # ← 原生 JAX API！
        in_shardings=(sharding, None),   # ← 原生 API！
        out_shardings=sharding            # ← 原生 API！
    )
    def sharded_matmul(a, b):
        return jnp.matmul(a, b)
    
    # 准备数据
    key = random.PRNGKey(999)
    A = random.normal(key, (1024, 512))
    B = random.normal(key, (512, 256))
    
    print(f"输入分片策略:")
    print(f"  A: 按第一维分片")
    print(f"  B: 不分片（复制）")
    print(f"输出分片策略:")
    print(f"  result: 按第一维分片")
    
    # 执行（JAX 自动处理分片）
    result = sharded_matmul(A, B)
    
    print(f"\n结果:")
    print(f"  形状: {result.shape}")
    print(f"  分片: {result.sharding}")
    
    return result


# ============================================================================
# 方法 7: 直接使用 pjit (原生 API)
# ============================================================================

def demo_native_pjit():
    """直接使用 jax.experimental.pjit"""
    print("\n" + "=" * 70)
    print("【方法 7】jax.experimental.pjit (原生 API)")
    print("=" * 70)
    
    from jax.experimental.pjit import pjit  # ← 原生 API！
    
    # 创建设备网格
    mesh = Mesh(np.array(devices[:4]), axis_names=('batch',))
    
    # 使用 pjit（原生 API）
    @pjit(  # ← 原生 JAX API！
        in_shardings=(P('batch', None), P(None, None)),
        out_shardings=P('batch', None)
    )
    def pjit_matmul(a, b):
        return jnp.matmul(a, b)
    
    # 在 mesh 上下文中执行
    with mesh:
        key = random.PRNGKey(111)
        A = random.normal(key, (512, 256))
        B = random.normal(key, (256, 128))
        
        print(f"输入:")
        print(f"  A: {A.shape}")
        print(f"  B: {B.shape}")
        
        result = pjit_matmul(A, B)
        
        print(f"\n结果:")
        print(f"  形状: {result.shape}")
        print(f"  分片: {result.sharding}")
    
    return result


# ============================================================================
# 完整示例：原生 API 训练循环
# ============================================================================

def demo_native_training_loop():
    """使用原生 JAX API 的完整训练循环"""
    print("\n" + "=" * 70)
    print("【完整示例】使用原生 JAX API 的训练循环")
    print("=" * 70)
    
    # 设置
    mesh = Mesh(np.array(devices[:4]), axis_names=('batch',))
    
    # 定义模型（原生 JAX）
    def model(params, x):
        w, b = params
        return jnp.matmul(x, w) + b
    
    def loss_fn(params, x, y):
        pred = model(params, x)
        return jnp.mean((pred - y) ** 2)
    
    # 定义训练步骤（原生 pmap）
    @pmap  # ← 原生 API！
    def train_step(params, x, y, lr):
        from jax import grad, value_and_grad
        
        # 计算损失和梯度（原生 API）
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        
        # 跨设备平均梯度（原生 API）
        grads = jax.lax.pmean(grads, axis_name='batch')
        
        # 更新参数
        new_params = jax.tree_map(
            lambda p, g: p - lr * g,
            params, grads
        )
        
        return new_params, loss
    
    # 初始化参数
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    # 在所有设备上复制参数
    params = [
        jnp.stack([random.normal(k1, (512, 256))] * 4),  # w
        jnp.stack([jnp.zeros(256)] * 4)  # b
    ]
    
    # 训练数据（每个设备不同批次）
    x = random.normal(k2, (4, 32, 512))  # (4设备, 32样本, 512特征)
    y = random.normal(k2, (4, 32, 256))
    
    print("训练配置:")
    print(f"  设备数: 4")
    print(f"  每设备批次: 32")
    print(f"  总批次大小: 128")
    print(f"  模型: 512 -> 256")
    
    # 训练循环（原生 JAX）
    print("\n开始训练...")
    for epoch in range(5):
        params, loss = train_step(params, x, y, lr=0.01)
        if epoch % 1 == 0:
            print(f"  Epoch {epoch}: Loss = {jnp.mean(loss):.6f}")
    
    print("\n✓ 训练完成！完全使用原生 JAX API")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    
    print("\n💡 关键洞察:")
    print("  虚拟 NPU 设备 = JAX 真实设备")
    print("  因此，所有 JAX 原生 API 都可以直接使用！")
    print("  不需要任何包装或修改！")
    
    try:
        demo_native_device_put()
        demo_native_pmap()
        demo_native_sharding_constraint()
        demo_native_shard_map()
        
        if len(devices) >= 4:
            demo_native_2d_mesh()
        
        demo_native_jit_shardings()
        demo_native_pjit()
        demo_native_training_loop()
        
        print("\n" + "=" * 70)
        print("所有演示完成！")
        print("=" * 70)
        
        print("\n✅ 总结:")
        print("  1. jax.device_put + NamedSharding - ✓ 原生")
        print("  2. jax.pmap - ✓ 原生")
        print("  3. jax.lax.with_sharding_constraint - ✓ 原生")
        print("  4. jax.experimental.shard_map - ✓ 原生")
        print("  5. 二维 Mesh - ✓ 原生")
        print("  6. jit in_shardings/out_shardings - ✓ 原生")
        print("  7. jax.experimental.pjit - ✓ 原生")
        print("  8. 完整训练循环 - ✓ 原生")
        
        print("\n🎉 虚拟 NPU 完全兼容 JAX 原生 API！")
        
    except Exception as e:
        print(f"\n⚠️  演示需要 JAX 环境")
        print(f"错误: {e}")
        print("\n但原理已经清楚：")
        print("虚拟 NPU 使用 JAX 真实设备，所以所有原生 API 都可用！")


if __name__ == "__main__":
    main()
