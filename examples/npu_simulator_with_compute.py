#!/usr/bin/env python3
"""
带计算能力的 NPU 模拟器

这个文件展示如何创建一个真正能执行计算的虚拟 NPU backend。
主要思路：
1. 使用 JAX 的 CPU backend 作为计算引擎
2. 模拟 NPU 的特性（量化、特殊算子、内存限制等）
3. 可以用来开发和测试 NPU 专用算子

示例用途：
- 模拟 NPU 量化计算（INT8/FP16）
- 测试 NPU 特定优化
- 验证算子融合逻辑
- 模拟 NPU 内存限制

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from typing import Callable, Any, Dict, List
import numpy as np
from functools import wraps
import time


# ============================================================================
# NPU 模拟器核心
# ============================================================================

class NPUSimulator:
    """NPU 计算模拟器
    
    这个类使用 JAX 的 CPU backend 来模拟 NPU 的计算行为。
    可以模拟：
    - INT8/FP16 量化计算
    - NPU 特定算子
    - 内存限制
    - 算子融合
    """
    
    def __init__(
        self,
        npu_name: str = "VirtualNPU",
        simulate_quantization: bool = True,
        simulate_memory_limit: bool = False,
        memory_limit_gb: int = 32,
        backend: str = "cpu"  # 底层计算 backend
    ):
        self.npu_name = npu_name
        self.simulate_quantization = simulate_quantization
        self.simulate_memory_limit = simulate_memory_limit
        self.memory_limit_gb = memory_limit_gb
        self.backend = backend
        
        # 统计信息
        self.stats = {
            "operations_count": 0,
            "int8_operations": 0,
            "fp16_operations": 0,
            "fp32_operations": 0,
            "total_compute_time": 0.0
        }
        
        print(f"✓ NPU 模拟器初始化: {npu_name}")
        print(f"  量化模拟: {simulate_quantization}")
        print(f"  内存限制: {memory_limit_gb} GB" if simulate_memory_limit else "  内存限制: 关闭")
        print(f"  计算后端: {backend.upper()}")
    
    # ------------------------------------------------------------------------
    # 量化模拟
    # ------------------------------------------------------------------------
    
    def quantize_int8(self, x: jnp.ndarray) -> jnp.ndarray:
        """模拟 INT8 量化"""
        if not self.simulate_quantization:
            return x
        
        # 简单的对称量化
        scale = jnp.max(jnp.abs(x)) / 127.0
        quantized = jnp.round(x / scale).astype(jnp.int8)
        return quantized.astype(jnp.float32) * scale
    
    def quantize_fp16(self, x: jnp.ndarray) -> jnp.ndarray:
        """模拟 FP16 量化"""
        if not self.simulate_quantization:
            return x
        
        # 转换为 FP16 再转回 FP32（模拟精度损失）
        return x.astype(jnp.float16).astype(jnp.float32)
    
    # ------------------------------------------------------------------------
    # NPU 算子模拟
    # ------------------------------------------------------------------------
    
    def npu_matmul(
        self, 
        a: jnp.ndarray, 
        b: jnp.ndarray,
        precision: str = "int8"
    ) -> jnp.ndarray:
        """NPU 矩阵乘法
        
        模拟 NPU 的矩阵乘法，支持量化计算。
        
        Args:
            a, b: 输入矩阵
            precision: 精度 ("int8", "fp16", "fp32")
        """
        start_time = time.time()
        
        # 根据精度进行量化
        if precision == "int8":
            a_q = self.quantize_int8(a)
            b_q = self.quantize_int8(b)
            result = jnp.matmul(a_q, b_q)
            self.stats["int8_operations"] += 1
        elif precision == "fp16":
            a_q = self.quantize_fp16(a)
            b_q = self.quantize_fp16(b)
            result = jnp.matmul(a_q, b_q)
            self.stats["fp16_operations"] += 1
        else:  # fp32
            result = jnp.matmul(a, b)
            self.stats["fp32_operations"] += 1
        
        self.stats["operations_count"] += 1
        self.stats["total_compute_time"] += time.time() - start_time
        
        return result
    
    def npu_conv2d(
        self,
        input: jnp.ndarray,
        kernel: jnp.ndarray,
        stride: int = 1,
        precision: str = "int8"
    ) -> jnp.ndarray:
        """NPU 卷积操作
        
        模拟 NPU 的 2D 卷积。
        """
        # 简化的卷积实现（实际应该用 lax.conv）
        if precision == "int8":
            input = self.quantize_int8(input)
            kernel = self.quantize_int8(kernel)
            self.stats["int8_operations"] += 1
        elif precision == "fp16":
            input = self.quantize_fp16(input)
            kernel = self.quantize_fp16(kernel)
            self.stats["fp16_operations"] += 1
        else:
            self.stats["fp32_operations"] += 1
        
        # 使用 JAX 的卷积
        from jax import lax
        dn = lax.conv_dimension_numbers(
            input.shape, kernel.shape,
            ('NCHW', 'OIHW', 'NCHW')
        )
        result = lax.conv_general_dilated(
            input, kernel, (stride, stride),
            'SAME', dimension_numbers=dn
        )
        
        self.stats["operations_count"] += 1
        return result
    
    def npu_relu(self, x: jnp.ndarray, precision: str = "fp16") -> jnp.ndarray:
        """NPU ReLU 激活函数"""
        if precision == "int8":
            x = self.quantize_int8(x)
        elif precision == "fp16":
            x = self.quantize_fp16(x)
        
        return jnp.maximum(0, x)
    
    def npu_batch_norm(
        self,
        x: jnp.ndarray,
        scale: jnp.ndarray,
        bias: jnp.ndarray,
        precision: str = "fp16"
    ) -> jnp.ndarray:
        """NPU Batch Normalization"""
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
        
        normalized = (x - mean) / jnp.sqrt(var + 1e-5)
        result = scale.reshape(1, -1, 1, 1) * normalized + bias.reshape(1, -1, 1, 1)
        
        if precision == "fp16":
            result = self.quantize_fp16(result)
        
        return result
    
    # ------------------------------------------------------------------------
    # 算子融合模拟
    # ------------------------------------------------------------------------
    
    def npu_fused_linear_relu(
        self,
        x: jnp.ndarray,
        weight: jnp.ndarray,
        bias: jnp.ndarray,
        precision: str = "int8"
    ) -> jnp.ndarray:
        """融合的线性层 + ReLU
        
        模拟 NPU 的算子融合优化。
        """
        # 矩阵乘法
        output = self.npu_matmul(x, weight, precision)
        
        # 加偏置
        output = output + bias
        
        # ReLU 激活
        output = self.npu_relu(output, precision)
        
        return output
    
    # ------------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------------
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "avg_op_time": (
                self.stats["total_compute_time"] / self.stats["operations_count"]
                if self.stats["operations_count"] > 0 else 0
            )
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "operations_count": 0,
            "int8_operations": 0,
            "fp16_operations": 0,
            "fp32_operations": 0,
            "total_compute_time": 0.0
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print(f"NPU 模拟器统计 ({self.npu_name})")
        print("=" * 60)
        print(f"总操作数: {stats['operations_count']}")
        print(f"  INT8 操作: {stats['int8_operations']}")
        print(f"  FP16 操作: {stats['fp16_operations']}")
        print(f"  FP32 操作: {stats['fp32_operations']}")
        print(f"总计算时间: {stats['total_compute_time']:.4f} 秒")
        print(f"平均操作时间: {stats['avg_op_time']*1000:.4f} 毫秒")
        print("=" * 60)


# ============================================================================
# 装饰器：NPU 算子
# ============================================================================

def npu_op(precision: str = "int8"):
    """装饰器：将函数标记为 NPU 算子
    
    使用方法:
        @npu_op(precision="int8")
        def my_npu_function(x, y):
            return x + y
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 在这里可以添加 NPU 特定的预处理
            result = func(*args, **kwargs)
            # 在这里可以添加 NPU 特定的后处理
            return result
        
        wrapper._npu_op = True
        wrapper._precision = precision
        return wrapper
    
    return decorator


# ============================================================================
# 示例：使用 NPU 模拟器
# ============================================================================

def demo_basic_operations():
    """示例 1: 基础算子测试"""
    print("\n【示例 1】基础 NPU 算子测试")
    print("=" * 60)
    
    # 创建 NPU 模拟器
    npu = NPUSimulator(npu_name="Ascend910-Sim")
    
    # 生成测试数据
    key = random.PRNGKey(0)
    A = random.normal(key, (256, 256))
    B = random.normal(key, (256, 256))
    
    print("\n测试 1: INT8 矩阵乘法")
    result_int8 = npu.npu_matmul(A, B, precision="int8")
    print(f"  结果形状: {result_int8.shape}")
    print(f"  结果范围: [{result_int8.min():.2f}, {result_int8.max():.2f}]")
    
    print("\n测试 2: FP16 矩阵乘法")
    result_fp16 = npu.npu_matmul(A, B, precision="fp16")
    print(f"  结果形状: {result_fp16.shape}")
    print(f"  结果范围: [{result_fp16.min():.2f}, {result_fp16.max():.2f}]")
    
    print("\n测试 3: FP32 矩阵乘法（参考）")
    result_fp32 = npu.npu_matmul(A, B, precision="fp32")
    print(f"  结果形状: {result_fp32.shape}")
    print(f"  结果范围: [{result_fp32.min():.2f}, {result_fp32.max():.2f}]")
    
    # 比较精度
    print("\n精度对比:")
    error_int8 = jnp.mean(jnp.abs(result_int8 - result_fp32))
    error_fp16 = jnp.mean(jnp.abs(result_fp16 - result_fp32))
    print(f"  INT8 vs FP32 平均误差: {error_int8:.6f}")
    print(f"  FP16 vs FP32 平均误差: {error_fp16:.6f}")
    
    npu.print_stats()


def demo_neural_network_layer():
    """示例 2: 模拟神经网络层"""
    print("\n【示例 2】NPU 神经网络层模拟")
    print("=" * 60)
    
    npu = NPUSimulator(npu_name="Ascend910-Sim", simulate_quantization=True)
    
    # 模拟一个全连接层
    batch_size = 32
    input_dim = 512
    output_dim = 256
    
    key = random.PRNGKey(42)
    k1, k2, k3 = random.split(key, 3)
    
    # 输入数据
    x = random.normal(k1, (batch_size, input_dim))
    weight = random.normal(k2, (input_dim, output_dim))
    bias = random.normal(k3, (output_dim,))
    
    print(f"\n层配置: {input_dim} -> {output_dim}")
    print(f"批次大小: {batch_size}")
    
    # 使用融合算子
    print("\n使用融合算子 (Linear + ReLU)...")
    output = npu.npu_fused_linear_relu(x, weight, bias, precision="int8")
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"非零元素比例: {(output > 0).mean():.2%}")
    
    npu.print_stats()


def demo_conv_network():
    """示例 3: 卷积网络"""
    print("\n【示例 3】NPU 卷积网络模拟")
    print("=" * 60)
    
    npu = NPUSimulator(npu_name="Ascend910-Sim")
    
    # 模拟卷积层输入 (NCHW 格式)
    batch_size = 8
    channels = 64
    height, width = 32, 32
    
    key = random.PRNGKey(123)
    k1, k2 = random.split(key)
    
    # 输入特征图
    input_map = random.normal(k1, (batch_size, channels, height, width))
    
    # 卷积核 (out_channels, in_channels, kernel_h, kernel_w)
    kernel = random.normal(k2, (128, channels, 3, 3))
    
    print(f"\n卷积配置:")
    print(f"  输入: {input_map.shape}")
    print(f"  卷积核: {kernel.shape}")
    
    # 执行卷积
    output = npu.npu_conv2d(input_map, kernel, stride=1, precision="int8")
    
    print(f"  输出: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    npu.print_stats()


def demo_performance_comparison():
    """示例 4: 性能对比"""
    print("\n【示例 4】不同精度性能对比")
    print("=" * 60)
    
    npu = NPUSimulator(npu_name="Ascend910-Sim")
    
    # 测试不同大小的矩阵
    sizes = [128, 256, 512, 1024]
    
    print(f"\n{'矩阵大小':<12} {'INT8 (ms)':<12} {'FP16 (ms)':<12} {'FP32 (ms)':<12}")
    print("-" * 60)
    
    for size in sizes:
        key = random.PRNGKey(size)
        A = random.normal(key, (size, size))
        B = random.normal(key, (size, size))
        
        # 预热
        _ = npu.npu_matmul(A, B, precision="int8")
        npu.reset_stats()
        
        # INT8
        start = time.time()
        _ = npu.npu_matmul(A, B, precision="int8")
        time_int8 = (time.time() - start) * 1000
        
        # FP16
        start = time.time()
        _ = npu.npu_matmul(A, B, precision="fp16")
        time_fp16 = (time.time() - start) * 1000
        
        # FP32
        start = time.time()
        _ = npu.npu_matmul(A, B, precision="fp32")
        time_fp32 = (time.time() - start) * 1000
        
        print(f"{size}x{size:<7} {time_int8:<12.4f} {time_fp16:<12.4f} {time_fp32:<12.4f}")


def demo_jit_compilation():
    """示例 5: JIT 编译优化"""
    print("\n【示例 5】JIT 编译优化")
    print("=" * 60)
    
    npu = NPUSimulator(npu_name="Ascend910-Sim")
    
    # 定义一个复杂的 NPU 计算图
    @jit
    def npu_mlp(x, w1, b1, w2, b2):
        """两层 MLP"""
        # 第一层
        h = npu.npu_matmul(x, w1, precision="int8")
        h = h + b1
        h = npu.npu_relu(h, precision="int8")
        
        # 第二层
        out = npu.npu_matmul(h, w2, precision="int8")
        out = out + b2
        
        return out
    
    # 准备数据
    key = random.PRNGKey(999)
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    x = random.normal(k1, (64, 512))
    w1 = random.normal(k2, (512, 256))
    b1 = random.normal(k3, (256,))
    w2 = random.normal(k4, (256, 128))
    b2 = random.normal(k5, (128,))
    
    print("\n编译 NPU 计算图...")
    # 第一次调用会触发编译
    start = time.time()
    result = npu_mlp(x, w1, b1, w2, b2)
    compile_time = (time.time() - start) * 1000
    
    print(f"编译 + 首次执行: {compile_time:.2f} ms")
    
    # 后续调用使用编译后的代码
    start = time.time()
    for _ in range(10):
        result = npu_mlp(x, w1, b1, w2, b2)
    exec_time = (time.time() - start) * 1000 / 10
    
    print(f"编译后平均执行: {exec_time:.2f} ms")
    print(f"加速比: {compile_time / exec_time:.2f}x")
    print(f"\n输出形状: {result.shape}")


def demo_custom_npu_operator():
    """示例 6: 自定义 NPU 算子"""
    print("\n【示例 6】自定义 NPU 算子")
    print("=" * 60)
    
    npu = NPUSimulator(npu_name="CustomNPU-Sim")
    
    # 定义自定义 NPU 算子
    @npu_op(precision="int8")
    def npu_gelu(x):
        """GELU 激活函数 (NPU 优化版本)"""
        # 先量化
        x_q = npu.quantize_int8(x)
        # GELU 近似: x * sigmoid(1.702 * x)
        return x_q * jax.nn.sigmoid(1.702 * x_q)
    
    @npu_op(precision="int8")
    def npu_layer_norm(x, eps=1e-5):
        """Layer Normalization (NPU 优化版本)"""
        x_q = npu.quantize_int8(x)
        mean = jnp.mean(x_q, axis=-1, keepdims=True)
        var = jnp.var(x_q, axis=-1, keepdims=True)
        return (x_q - mean) / jnp.sqrt(var + eps)
    
    # 测试自定义算子
    key = random.PRNGKey(777)
    x = random.normal(key, (32, 512))
    
    print("\n测试自定义 GELU 算子:")
    result_gelu = npu_gelu(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {result_gelu.shape}")
    print(f"  输出范围: [{result_gelu.min():.4f}, {result_gelu.max():.4f}]")
    
    print("\n测试自定义 LayerNorm 算子:")
    result_ln = npu_layer_norm(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {result_ln.shape}")
    print(f"  输出均值: {result_ln.mean():.6f} (应接近 0)")
    print(f"  输出标准差: {result_ln.std():.6f} (应接近 1)")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    print("=" * 60)
    print("NPU 模拟器 - 带计算能力")
    print("=" * 60)
    print("\n这个模拟器使用 JAX 来模拟 NPU 的计算行为")
    print("支持:")
    print("  - INT8/FP16/FP32 量化")
    print("  - 矩阵乘法、卷积等基础算子")
    print("  - 算子融合")
    print("  - JIT 编译优化")
    print("  - 自定义 NPU 算子")
    
    # 运行所有演示
    demo_basic_operations()
    demo_neural_network_layer()
    demo_conv_network()
    demo_performance_comparison()
    demo_jit_compilation()
    demo_custom_npu_operator()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n💡 提示：你可以基于这个框架开发:")
    print("  1. NPU 专用算子库")
    print("  2. NPU 量化训练模拟")
    print("  3. NPU 性能分析工具")
    print("  4. NPU 算子融合优化器")


if __name__ == "__main__":
    main()
