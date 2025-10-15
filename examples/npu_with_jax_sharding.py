#!/usr/bin/env python3
"""
在虚拟 NPU Backend 上使用 JAX 分布式计算

这个文件展示如何在虚拟 NPU backend 中复用 JAX 的所有分布式能力：
- pmap (数据并行)
- shard_map (灵活分片)
- Sharding API (自动分片)
- 分布式 matmul

核心思路：
虚拟 NPU backend 使用 JAX CPU/GPU backend 作为计算引擎，
所以可以直接使用 JAX 的所有分布式 API！

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from jax import random, jit, pmap, vmap
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import numpy as np
from typing import Tuple


# ============================================================================
# 虚拟 NPU 设备管理
# ============================================================================

class VirtualNPUDeviceManager:
    """虚拟 NPU 设备管理器
    
    管理多个虚拟 NPU 设备，并将它们映射到 JAX 的真实设备上。
    """
    
    def __init__(self, num_virtual_npus: int = 4, use_real_devices: bool = True):
        """
        Args:
            num_virtual_npus: 虚拟 NPU 数量
            use_real_devices: 是否使用真实的 JAX 设备（CPU/GPU）
        """
        self.num_virtual_npus = num_virtual_npus
        
        if use_real_devices:
            # 使用 JAX 的真实设备
            self.devices = jax.devices()[:num_virtual_npus]
            if len(self.devices) < num_virtual_npus:
                print(f"⚠️  只有 {len(self.devices)} 个真实设备，"
                      f"但请求了 {num_virtual_npus} 个虚拟 NPU")
                self.num_virtual_npus = len(self.devices)
        else:
            # 使用 CPU 设备模拟
            self.devices = jax.devices()[:1] * num_virtual_npus
        
        print(f"✓ 初始化 {self.num_virtual_npus} 个虚拟 NPU 设备")
        for i, dev in enumerate(self.devices):
            print(f"  虚拟NPU-{i} -> {dev}")
    
    def get_mesh(self, mesh_shape: Tuple[int, ...], axis_names: Tuple[str, ...]):
        """创建设备网格
        
        Args:
            mesh_shape: 网格形状，如 (2, 2) 表示 2x2 网格
            axis_names: 轴名称，如 ('data', 'model')
        """
        # 创建设备网格
        device_array = np.array(self.devices).reshape(mesh_shape)
        return Mesh(device_array, axis_names)


# ============================================================================
# NPU 分布式计算包装器
# ============================================================================

class NPUDistributedCompute:
    """NPU 分布式计算
    
    在虚拟 NPU 上使用 JAX 的分布式计算能力。
    """
    
    def __init__(self, device_manager: VirtualNPUDeviceManager):
        self.device_manager = device_manager
        self.devices = device_manager.devices
        self.num_devices = len(self.devices)
    
    # ------------------------------------------------------------------------
    # 方法 1: 使用 pmap (数据并行)
    # ------------------------------------------------------------------------
    
    def pmap_matmul(self, A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """使用 pmap 进行数据并行的矩阵乘法
        
        将数据在多个 NPU 设备上并行处理。
        
        Args:
            A: 形状 (num_devices, batch_per_device, n, k)
            B: 形状 (k, m) - 在所有设备上复制
        """
        @pmap
        def matmul_fn(a, b):
            return jnp.matmul(a, b)
        
        # B 需要在第一个维度上复制
        B_replicated = jnp.stack([B] * self.num_devices)
        
        result = matmul_fn(A, B_replicated)
        return result
    
    def pmap_layer(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray
    ) -> jnp.ndarray:
        """使用 pmap 的神经网络层
        
        数据并行：每个设备处理不同的批次。
        """
        @pmap
        def layer_fn(x_shard, w, b):
            # 线性变换 + ReLU
            out = jnp.matmul(x_shard, w) + b
            return jnp.maximum(0, out)
        
        # 复制权重和偏置到所有设备
        weight_rep = jnp.stack([weight] * self.num_devices)
        bias_rep = jnp.stack([bias] * self.num_devices)
        
        return layer_fn(x, weight_rep, bias_rep)
    
    # ------------------------------------------------------------------------
    # 方法 2: 使用 Sharding API (自动分片)
    # ------------------------------------------------------------------------
    
    def sharded_matmul(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        mesh_shape: Tuple[int, ...] = None
    ) -> jnp.ndarray:
        """使用 Sharding API 的矩阵乘法
        
        自动在设备间分片数据。
        
        Args:
            A: 形状 (m, k)
            B: 形状 (k, n)
            mesh_shape: 设备网格形状
        """
        if mesh_shape is None:
            mesh_shape = (self.num_devices,)
        
        # 创建设备网格
        mesh = self.device_manager.get_mesh(mesh_shape, ('data',))
        
        # 定义分片策略
        # A 按第一维分片，B 不分片
        sharding_A = NamedSharding(mesh, P('data', None))
        sharding_B = NamedSharding(mesh, P(None, None))
        sharding_out = NamedSharding(mesh, P('data', None))
        
        @jit
        def matmul_fn(a, b):
            return jnp.matmul(a, b)
        
        # 将数据放置到设备上
        A_sharded = jax.device_put(A, sharding_A)
        B_sharded = jax.device_put(B, sharding_B)
        
        # 执行计算（自动处理分片）
        result = matmul_fn(A_sharded, B_sharded)
        
        return result
    
    # ------------------------------------------------------------------------
    # 方法 3: 使用 shard_map (更灵活的分片)
    # ------------------------------------------------------------------------
    
    def shard_map_matmul(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray
    ) -> jnp.ndarray:
        """使用 shard_map 的矩阵乘法
        
        手动指定每个设备上的计算逻辑。
        """
        mesh = self.device_manager.get_mesh((self.num_devices,), ('devices',))
        
        @jit
        @shard_map(
            mesh=mesh,
            in_specs=(P('devices', None), P(None, None)),
            out_specs=P('devices', None)
        )
        def sharded_matmul(a_shard, b):
            # 每个设备上的计算
            # a_shard: (rows_per_device, k)
            # b: (k, n)
            return jnp.matmul(a_shard, b)
        
        return sharded_matmul(A, B)
    
    # ------------------------------------------------------------------------
    # 方法 4: 二维分片（高级）
    # ------------------------------------------------------------------------
    
    def sharded_matmul_2d(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        mesh_shape: Tuple[int, int] = (2, 2)
    ) -> jnp.ndarray:
        """二维分片的矩阵乘法
        
        将 A 和 B 都进行二维分片，实现更高效的并行。
        
        Args:
            A: 形状 (m, k)
            B: 形状 (k, n)
            mesh_shape: (rows, cols) 设备网格形状
        """
        mesh = self.device_manager.get_mesh(mesh_shape, ('rows', 'cols'))
        
        # A: 按 (rows, k) 分片
        # B: 按 (k, cols) 分片
        # C: 按 (rows, cols) 分片
        sharding_A = NamedSharding(mesh, P('rows', None))
        sharding_B = NamedSharding(mesh, P(None, 'cols'))
        sharding_C = NamedSharding(mesh, P('rows', 'cols'))
        
        @jit
        def matmul_2d(a, b):
            return jnp.matmul(a, b)
        
        A_sharded = jax.device_put(A, sharding_A)
        B_sharded = jax.device_put(B, sharding_B)
        
        result = matmul_2d(A_sharded, B_sharded)
        
        return result


# ============================================================================
# 演示和测试
# ============================================================================

def demo_pmap_data_parallel():
    """示例 1: 使用 pmap 进行数据并行"""
    print("\n【示例 1】pmap 数据并行矩阵乘法")
    print("=" * 70)
    
    # 创建虚拟 NPU 设备
    npu_mgr = VirtualNPUDeviceManager(num_virtual_npus=4)
    compute = NPUDistributedCompute(npu_mgr)
    
    # 准备数据：4 个设备，每个设备处理 32 个样本
    batch_per_device = 32
    n, k, m = 128, 256, 64
    
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    # A: (4, 32, 128, 256) - 每个设备一个分片
    A = random.normal(k1, (4, batch_per_device, n, k))
    # B: (256, 64) - 所有设备共享
    B = random.normal(k2, (k, m))
    
    print(f"输入:")
    print(f"  A: {A.shape} (分片到 4 个设备)")
    print(f"  B: {B.shape} (复制到所有设备)")
    
    # 执行数据并行计算
    result = compute.pmap_matmul(A, B)
    
    print(f"\n输出:")
    print(f"  Result: {result.shape}")
    print(f"  每个设备输出形状: ({batch_per_device}, {n}, {m})")
    
    # 验证结果
    expected = jnp.stack([jnp.matmul(A[i], B) for i in range(4)])
    error = jnp.mean(jnp.abs(result - expected))
    print(f"  验证误差: {error:.2e}")


def demo_pmap_neural_network():
    """示例 2: 使用 pmap 训练神经网络层"""
    print("\n【示例 2】pmap 数据并行训练")
    print("=" * 70)
    
    npu_mgr = VirtualNPUDeviceManager(num_virtual_npus=4)
    compute = NPUDistributedCompute(npu_mgr)
    
    # 数据：4 个设备，每个 16 个样本
    batch_per_device = 16
    input_dim, output_dim = 512, 256
    
    key = random.PRNGKey(42)
    k1, k2, k3 = random.split(key, 3)
    
    # 输入数据分片
    x = random.normal(k1, (4, batch_per_device, input_dim))
    # 权重和偏置（所有设备共享）
    weight = random.normal(k2, (input_dim, output_dim))
    bias = random.normal(k3, (output_dim,))
    
    print(f"配置:")
    print(f"  总批次大小: {4 * batch_per_device}")
    print(f"  每设备批次: {batch_per_device}")
    print(f"  层: {input_dim} -> {output_dim}")
    
    # 前向传播
    output = compute.pmap_layer(x, weight, bias)
    
    print(f"\n结果:")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  非零元素比例: {(output > 0).mean():.2%}")


def demo_sharded_matmul():
    """示例 3: 使用 Sharding API 自动分片"""
    print("\n【示例 3】Sharding API 自动分片")
    print("=" * 70)
    
    npu_mgr = VirtualNPUDeviceManager(num_virtual_npus=4)
    compute = NPUDistributedCompute(npu_mgr)
    
    # 大矩阵
    m, k, n = 1024, 512, 256
    
    key = random.PRNGKey(123)
    k1, k2 = random.split(key)
    
    A = random.normal(k1, (m, k))
    B = random.normal(k2, (k, n))
    
    print(f"矩阵:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  自动分片策略: A 按第一维分片，B 不分片")
    
    # 使用自动分片
    result = compute.sharded_matmul(A, B)
    
    print(f"\n结果:")
    print(f"  输出形状: {result.shape}")
    print(f"  输出范围: [{result.min():.4f}, {result.max():.4f}]")
    
    # 验证
    expected = jnp.matmul(A, B)
    error = jnp.mean(jnp.abs(result - expected))
    print(f"  验证误差: {error:.2e}")


def demo_shard_map():
    """示例 4: 使用 shard_map 手动分片"""
    print("\n【示例 4】shard_map 手动分片控制")
    print("=" * 70)
    
    npu_mgr = VirtualNPUDeviceManager(num_virtual_npus=4)
    compute = NPUDistributedCompute(npu_mgr)
    
    # 矩阵
    m, k, n = 512, 256, 128
    
    key = random.PRNGKey(456)
    k1, k2 = random.split(key)
    
    A = random.normal(k1, (m, k))
    B = random.normal(k2, (k, n))
    
    print(f"矩阵:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  分片策略: A 按设备分片，B 复制")
    print(f"  每设备计算: ({m//4}, {k}) × ({k}, {n})")
    
    # 使用 shard_map
    result = compute.shard_map_matmul(A, B)
    
    print(f"\n结果:")
    print(f"  输出形状: {result.shape}")
    
    # 验证
    expected = jnp.matmul(A, B)
    error = jnp.mean(jnp.abs(result - expected))
    print(f"  验证误差: {error:.2e}")


def demo_2d_sharding():
    """示例 5: 二维分片（高级）"""
    print("\n【示例 5】二维分片矩阵乘法")
    print("=" * 70)
    
    npu_mgr = VirtualNPUDeviceManager(num_virtual_npus=4)
    compute = NPUDistributedCompute(npu_mgr)
    
    # 大矩阵
    m, k, n = 1024, 1024, 1024
    
    key = random.PRNGKey(789)
    k1, k2 = random.split(key)
    
    A = random.normal(k1, (m, k))
    B = random.normal(k2, (k, n))
    
    print(f"矩阵:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  设备网格: 2×2 (4 个设备)")
    print(f"  A 分片: 按行分片")
    print(f"  B 分片: 按列分片")
    print(f"  结果: 2D 分片 (行×列)")
    
    # 二维分片
    result = compute.sharded_matmul_2d(A, B, mesh_shape=(2, 2))
    
    print(f"\n结果:")
    print(f"  输出形状: {result.shape}")
    
    # 验证
    expected = jnp.matmul(A, B)
    error = jnp.mean(jnp.abs(result - expected))
    print(f"  验证误差: {error:.2e}")


def demo_performance_comparison():
    """示例 6: 性能对比"""
    print("\n【示例 6】不同分片策略性能对比")
    print("=" * 70)
    
    import time
    
    npu_mgr = VirtualNPUDeviceManager(num_virtual_npus=4)
    compute = NPUDistributedCompute(npu_mgr)
    
    # 测试矩阵大小
    sizes = [512, 1024, 2048]
    
    print(f"\n{'大小':<10} {'串行 (ms)':<12} {'pmap (ms)':<12} {'sharding (ms)':<12} {'加速比':<10}")
    print("-" * 70)
    
    for size in sizes:
        m = k = n = size
        
        key = random.PRNGKey(size)
        k1, k2 = random.split(key)
        
        # 1. 串行计算
        A_serial = random.normal(k1, (m, k))
        B_serial = random.normal(k2, (k, n))
        
        @jit
        def serial_matmul(a, b):
            return jnp.matmul(a, b)
        
        # 预热
        _ = serial_matmul(A_serial, B_serial).block_until_ready()
        
        start = time.time()
        _ = serial_matmul(A_serial, B_serial).block_until_ready()
        time_serial = (time.time() - start) * 1000
        
        # 2. pmap 计算
        A_pmap = random.normal(k1, (4, m//4, k))
        B_pmap = B_serial
        
        # 预热
        _ = compute.pmap_matmul(A_pmap, B_pmap)
        
        start = time.time()
        _ = compute.pmap_matmul(A_pmap, B_pmap)
        time_pmap = (time.time() - start) * 1000
        
        # 3. Sharding 计算
        # 预热
        _ = compute.sharded_matmul(A_serial, B_serial)
        
        start = time.time()
        _ = compute.sharded_matmul(A_serial, B_serial)
        time_sharding = (time.time() - start) * 1000
        
        speedup = time_serial / min(time_pmap, time_sharding)
        
        print(f"{size}x{size:<5} {time_serial:<12.2f} {time_pmap:<12.2f} "
              f"{time_sharding:<12.2f} {speedup:<10.2f}x")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    print("=" * 70)
    print("虚拟 NPU 上的 JAX 分布式计算")
    print("=" * 70)
    print("\n✓ 完全复用 JAX 的分布式能力！")
    print("\n支持的分布式模式:")
    print("  1. pmap - 数据并行（SPMD）")
    print("  2. Sharding API - 自动分片")
    print("  3. shard_map - 手动分片控制")
    print("  4. 二维分片 - 高级并行")
    
    try:
        demo_pmap_data_parallel()
        demo_pmap_neural_network()
        demo_sharded_matmul()
        demo_shard_map()
        demo_2d_sharding()
        demo_performance_comparison()
        
        print("\n" + "=" * 70)
        print("所有演示完成！")
        print("=" * 70)
        
        print("\n💡 核心优势:")
        print("  ✓ 直接使用 JAX 的所有分布式 API")
        print("  ✓ 自动处理数据分片和通信")
        print("  ✓ JIT 编译优化")
        print("  ✓ 支持多种并行模式")
        print("  ✓ 可扩展到任意数量的虚拟 NPU")
        
    except Exception as e:
        print(f"\n⚠️  演示需要 JAX 环境")
        print(f"错误: {e}")
        print("\n这些示例展示了如何在虚拟 NPU 上使用 JAX 分布式能力")


if __name__ == "__main__":
    main()
