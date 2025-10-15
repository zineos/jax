#!/usr/bin/env python3
"""
完整的可计算 NPU Backend

这个文件展示如何创建一个真正能执行计算的 NPU backend，
将 NPU 模拟器集成到 JAX backend 系统中。

核心思路：
1. 虚拟 NPU backend 负责设备管理和接口
2. NPU 模拟器负责实际计算（使用 JAX CPU/GPU）
3. 可以模拟 NPU 的各种特性

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Callable, Dict, List, Any
import time

# 导入 NPU 模拟器
from examples.npu_simulator_with_compute import NPUSimulator
from examples.simulate_npu_cluster import VirtualNPU, NPUClusterBackend


# ============================================================================
# 可计算的 NPU Backend
# ============================================================================

class ComputableNPUBackend(NPUClusterBackend):
    """可执行计算的 NPU Backend
    
    继承自 NPUClusterBackend，添加计算能力。
    每个设备都有一个对应的 NPU 模拟器。
    """
    
    def __init__(self, *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 为每个本地设备创建 NPU 模拟器
        self._simulators = {}
        for device in self._local_devices:
            simulator = NPUSimulator(
                npu_name=f"{self.platform}:{device.id}",
                simulate_quantization=True,
                backend="cpu"  # 使用 CPU 作为计算后端
            )
            self._simulators[device.id] = simulator
        
        print(f"\n✓ 可计算 NPU Backend 初始化完成")
        print(f"  创建了 {len(self._simulators)} 个 NPU 模拟器")
    
    # ------------------------------------------------------------------------
    # 计算接口
    # ------------------------------------------------------------------------
    
    def execute_on_device(
        self,
        device_id: int,
        operation: str,
        *args,
        precision: str = "int8",
        **kwargs
    ):
        """在指定设备上执行操作
        
        Args:
            device_id: 设备 ID
            operation: 操作名称 ("matmul", "conv2d", etc.)
            *args: 操作参数
            precision: 计算精度
            **kwargs: 其他参数
        """
        if device_id not in self._simulators:
            raise ValueError(f"设备 {device_id} 不存在或不在本地")
        
        simulator = self._simulators[device_id]
        
        # 根据操作类型调用相应的模拟器方法
        if operation == "matmul":
            return simulator.npu_matmul(*args, precision=precision)
        elif operation == "conv2d":
            return simulator.npu_conv2d(*args, precision=precision, **kwargs)
        elif operation == "relu":
            return simulator.npu_relu(*args, precision=precision)
        elif operation == "fused_linear_relu":
            return simulator.npu_fused_linear_relu(*args, precision=precision)
        else:
            raise ValueError(f"未知操作: {operation}")
    
    def get_device_stats(self, device_id: int) -> Dict:
        """获取设备统计信息"""
        if device_id not in self._simulators:
            raise ValueError(f"设备 {device_id} 不存在")
        return self._simulators[device_id].get_stats()
    
    def reset_device_stats(self, device_id: int = None):
        """重置设备统计信息
        
        Args:
            device_id: 设备 ID，如果为 None 则重置所有设备
        """
        if device_id is None:
            for simulator in self._simulators.values():
                simulator.reset_stats()
        else:
            if device_id in self._simulators:
                self._simulators[device_id].reset_stats()
    
    def print_all_device_stats(self):
        """打印所有设备的统计信息"""
        print("\n" + "=" * 70)
        print("所有设备统计信息")
        print("=" * 70)
        
        for device_id, simulator in self._simulators.items():
            stats = simulator.get_stats()
            print(f"\n设备 {device_id} ({self.platform}:{device_id}):")
            print(f"  总操作数: {stats['operations_count']}")
            print(f"  INT8: {stats['int8_operations']}, "
                  f"FP16: {stats['fp16_operations']}, "
                  f"FP32: {stats['fp32_operations']}")
            print(f"  总时间: {stats['total_compute_time']:.4f}s, "
                  f"平均: {stats['avg_op_time']*1000:.4f}ms")
        
        print("=" * 70)


# ============================================================================
# 分布式计算示例
# ============================================================================

class DistributedNPUCompute:
    """分布式 NPU 计算
    
    模拟在多个 NPU 设备上的分布式计算。
    """
    
    def __init__(self, backend: ComputableNPUBackend):
        self.backend = backend
        self.local_device_ids = [d.id for d in backend.local_devices()]
    
    def distributed_matmul(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        precision: str = "int8"
    ) -> jnp.ndarray:
        """分布式矩阵乘法
        
        将矩阵 A 按行切分到多个设备，每个设备计算一部分。
        """
        num_devices = len(self.local_device_ids)
        
        # 按行切分矩阵 A
        rows_per_device = A.shape[0] // num_devices
        results = []
        
        print(f"\n分布式矩阵乘法:")
        print(f"  设备数: {num_devices}")
        print(f"  矩阵 A: {A.shape}, 矩阵 B: {B.shape}")
        print(f"  每设备处理: {rows_per_device} 行")
        
        for i, device_id in enumerate(self.local_device_ids):
            start_row = i * rows_per_device
            end_row = start_row + rows_per_device if i < num_devices - 1 else A.shape[0]
            
            A_slice = A[start_row:end_row]
            
            print(f"  设备 {device_id}: 行 {start_row}-{end_row}")
            
            # 在设备上执行计算
            result_slice = self.backend.execute_on_device(
                device_id, "matmul", A_slice, B, precision=precision
            )
            results.append(result_slice)
        
        # 合并结果
        result = jnp.concatenate(results, axis=0)
        print(f"  最终结果: {result.shape}")
        
        return result
    
    def data_parallel_training_step(
        self,
        data_batches: List[jnp.ndarray],
        weights: jnp.ndarray,
        bias: jnp.ndarray,
        precision: str = "int8"
    ):
        """数据并行训练步骤
        
        每个设备处理不同的数据批次。
        """
        results = []
        
        print(f"\n数据并行训练:")
        print(f"  设备数: {len(self.local_device_ids)}")
        print(f"  每批次大小: {data_batches[0].shape}")
        
        for device_id, batch in zip(self.local_device_ids, data_batches):
            # 在每个设备上执行前向传播
            output = self.backend.execute_on_device(
                device_id, "fused_linear_relu",
                batch, weights, bias,
                precision=precision
            )
            results.append(output)
            
            print(f"  设备 {device_id}: 处理批次 {batch.shape} -> {output.shape}")
        
        return results


# ============================================================================
# 使用示例
# ============================================================================

def demo_basic_compute():
    """示例 1: 基础计算"""
    print("\n【示例 1】基础 NPU 计算")
    print("=" * 70)
    
    # 创建可计算的 NPU backend
    backend = ComputableNPUBackend(
        platform="ascend",
        chip_model="Ascend910",
        num_nodes=1,
        npus_per_node=4,
        memory_gb=32
    )
    
    # 生成测试数据
    key = random.PRNGKey(0)
    A = random.normal(key, (128, 128))
    B = random.normal(key, (128, 128))
    
    # 在设备 0 上执行矩阵乘法
    print("\n在设备 0 上执行 INT8 矩阵乘法...")
    result = backend.execute_on_device(0, "matmul", A, B, precision="int8")
    
    print(f"结果形状: {result.shape}")
    print(f"结果范围: [{result.min():.4f}, {result.max():.4f}]")
    
    # 查看统计
    backend.print_all_device_stats()


def demo_multi_device_compute():
    """示例 2: 多设备计算"""
    print("\n【示例 2】多设备并行计算")
    print("=" * 70)
    
    # 创建 4 设备 backend
    backend = ComputableNPUBackend(
        platform="ascend",
        chip_model="Ascend910",
        num_nodes=1,
        npus_per_node=4
    )
    
    # 在不同设备上执行不同计算
    key = random.PRNGKey(42)
    data = random.normal(key, (64, 256))
    
    print("\n在多个设备上并行执行计算...")
    for i, device in enumerate(backend.local_devices()):
        # 每个设备使用不同的权重
        k = random.PRNGKey(i)
        weight = random.normal(k, (256, 128))
        bias = random.normal(k, (128,))
        
        result = backend.execute_on_device(
            device.id, "fused_linear_relu",
            data, weight, bias,
            precision="int8"
        )
        
        print(f"设备 {device.id}: {data.shape} -> {result.shape}")
    
    backend.print_all_device_stats()


def demo_distributed_computation():
    """示例 3: 分布式计算"""
    print("\n【示例 3】分布式矩阵乘法")
    print("=" * 70)
    
    # 创建 backend
    backend = ComputableNPUBackend(
        platform="ascend",
        chip_model="Ascend910",
        num_nodes=1,
        npus_per_node=4
    )
    
    # 创建分布式计算对象
    dist_compute = DistributedNPUCompute(backend)
    
    # 大矩阵乘法
    key = random.PRNGKey(123)
    A = random.normal(key, (512, 256))
    B = random.normal(key, (256, 128))
    
    # 分布式计算
    result = dist_compute.distributed_matmul(A, B, precision="int8")
    
    print(f"\n分布式计算完成！")
    print(f"最终结果: {result.shape}")
    
    backend.print_all_device_stats()


def demo_data_parallel_training():
    """示例 4: 数据并行训练"""
    print("\n【示例 4】数据并行训练模拟")
    print("=" * 70)
    
    backend = ComputableNPUBackend(
        platform="ascend",
        chip_model="Ascend910",
        num_nodes=1,
        npus_per_node=4
    )
    
    dist_compute = DistributedNPUCompute(backend)
    
    # 准备数据：4 个设备，每个处理不同的批次
    batch_size = 32
    input_dim = 512
    output_dim = 256
    
    key = random.PRNGKey(999)
    
    # 每个设备的数据批次
    data_batches = [
        random.normal(random.PRNGKey(i), (batch_size, input_dim))
        for i in range(4)
    ]
    
    # 共享的权重和偏置
    k1, k2 = random.split(key)
    weights = random.normal(k1, (input_dim, output_dim))
    bias = random.normal(k2, (output_dim,))
    
    # 执行一个训练步骤
    outputs = dist_compute.data_parallel_training_step(
        data_batches, weights, bias, precision="int8"
    )
    
    print(f"\n训练步骤完成！")
    print(f"每个设备的输出形状: {outputs[0].shape}")
    
    backend.print_all_device_stats()


def demo_precision_comparison():
    """示例 5: 不同精度对比"""
    print("\n【示例 5】不同精度计算对比")
    print("=" * 70)
    
    backend = ComputableNPUBackend(
        platform="ascend",
        chip_model="Ascend910",
        num_nodes=1,
        npus_per_node=1
    )
    
    # 测试数据
    key = random.PRNGKey(777)
    A = random.normal(key, (256, 256))
    B = random.normal(key, (256, 256))
    
    # 不同精度计算
    precisions = ["int8", "fp16", "fp32"]
    results = {}
    
    print("\n在同一设备上使用不同精度:")
    for precision in precisions:
        backend.reset_device_stats(0)
        
        start = time.time()
        result = backend.execute_on_device(0, "matmul", A, B, precision=precision)
        exec_time = (time.time() - start) * 1000
        
        results[precision] = result
        print(f"\n{precision.upper()} 精度:")
        print(f"  执行时间: {exec_time:.4f} ms")
        print(f"  结果范围: [{result.min():.4f}, {result.max():.4f}]")
    
    # 精度对比
    print("\n精度误差对比 (相对于 FP32):")
    ref = results["fp32"]
    for precision in ["int8", "fp16"]:
        error = jnp.mean(jnp.abs(results[precision] - ref))
        rel_error = error / (jnp.mean(jnp.abs(ref)) + 1e-8)
        print(f"  {precision.upper()}: 绝对误差={error:.6f}, 相对误差={rel_error:.6f}")


def demo_performance_scaling():
    """示例 6: 性能扩展性测试"""
    print("\n【示例 6】多设备性能扩展性")
    print("=" * 70)
    
    # 测试不同数量的设备
    device_counts = [1, 2, 4]
    
    print(f"\n{'设备数':<10} {'总时间 (ms)':<15} {'加速比':<10}")
    print("-" * 40)
    
    baseline_time = None
    
    for num_devices in device_counts:
        backend = ComputableNPUBackend(
            platform="ascend",
            chip_model="Ascend910",
            num_nodes=1,
            npus_per_node=num_devices
        )
        
        dist_compute = DistributedNPUCompute(backend)
        
        # 固定大小的问题
        key = random.PRNGKey(888)
        A = random.normal(key, (1024, 512))
        B = random.normal(key, (512, 256))
        
        # 计时
        start = time.time()
        _ = dist_compute.distributed_matmul(A, B, precision="int8")
        total_time = (time.time() - start) * 1000
        
        if baseline_time is None:
            baseline_time = total_time
            speedup = 1.0
        else:
            speedup = baseline_time / total_time
        
        print(f"{num_devices:<10} {total_time:<15.4f} {speedup:<10.2f}x")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    print("=" * 70)
    print("可计算 NPU Backend 完整演示")
    print("=" * 70)
    print("\n这个 Backend 真正支持计算！")
    print("功能:")
    print("  ✓ 多设备并行计算")
    print("  ✓ 分布式矩阵运算")
    print("  ✓ 数据并行训练")
    print("  ✓ INT8/FP16/FP32 多精度")
    print("  ✓ 性能统计和分析")
    
    demo_basic_compute()
    demo_multi_device_compute()
    demo_distributed_computation()
    demo_data_parallel_training()
    demo_precision_comparison()
    demo_performance_scaling()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print("\n💡 你现在可以:")
    print("  1. 用这个框架开发 NPU 专用算子")
    print("  2. 测试分布式训练逻辑")
    print("  3. 验证量化训练效果")
    print("  4. 分析不同精度的性能权衡")


if __name__ == "__main__":
    main()
