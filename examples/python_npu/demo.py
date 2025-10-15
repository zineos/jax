#!/usr/bin/env python3
"""
NPU模拟器演示脚本

展示纯Python NPU实现的完整功能，包括：
1. 基本矩阵乘法操作
2. 多设备并行计算
3. 梯度计算和自动微分
4. 性能基准测试
"""

import jax
import jax.numpy as jnp
import numpy as np
from npu_simulator import (
    initialize_npu, npu_matmul, npu_batch_matmul,
    get_npu_device_count, set_npu_device, get_npu_device_info,
    shutdown_npu
)
from parallel_strategies import (
    NPUParallelCompute, create_npu_distributed_array,
    benchmark_npu_parallel_strategies
)

def demo_basic_operations():
    """演示基本NPU操作"""
    print("\n" + "=" * 50)
    print("🎯 DEMO 1: Basic NPU Operations")
    print("=" * 50)
    
    # 创建测试矩阵
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)
    
    a = jax.random.normal(key1, (512, 256))
    b = jax.random.normal(key2, (256, 128))
    
    print(f"Matrix A: {a.shape} {a.dtype}")
    print(f"Matrix B: {b.shape} {b.dtype}")
    
    # NPU矩阵乘法
    print("\n🔥 Executing NPU MatMul...")
    result = npu_matmul(a, b, device_id=0)
    
    print(f"Result: {result.shape} {result.dtype}")
    print(f"Result sample: {result[0, :5]}")
    
    # 验证正确性
    expected = jnp.dot(a, b)
    error = jnp.max(jnp.abs(result - expected))
    print(f"✅ Verification: max error = {error:.2e}")
    
    return result

def demo_automatic_differentiation():
    """演示自动微分功能"""
    print("\n" + "=" * 50)
    print("🎯 DEMO 2: Automatic Differentiation")
    print("=" * 50)
    
    # 定义一个使用NPU操作的函数
    def npu_neural_layer(x, w, b):
        """简单的神经网络层"""
        linear = npu_matmul(x, w, device_id=0)
        return linear + b
    
    def loss_function(params, x, y_target):
        """损失函数"""
        w, b = params
        y_pred = npu_neural_layer(x, w, b)
        return jnp.mean((y_pred - y_target) ** 2)
    
    # 创建测试数据
    key = jax.random.key(123)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    x = jax.random.normal(key1, (32, 64))  # batch_size=32, input_dim=64
    w = jax.random.normal(key2, (64, 16))  # input_dim=64, output_dim=16
    b = jax.random.normal(key3, (16,))     # output_dim=16
    y_target = jax.random.normal(key4, (32, 16))
    
    params = (w, b)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Target shape: {y_target.shape}")
    
    # 计算损失
    loss_value = loss_function(params, x, y_target)
    print(f"\n📊 Initial loss: {loss_value:.6f}")
    
    # 计算梯度
    print("🔄 Computing gradients using NPU operations...")
    grad_fn = jax.grad(loss_function)
    gradients = grad_fn(params, x, y_target)
    
    grad_w, grad_b = gradients
    print(f"✅ Gradient w shape: {grad_w.shape}, norm: {jnp.linalg.norm(grad_w):.4f}")
    print(f"✅ Gradient b shape: {grad_b.shape}, norm: {jnp.linalg.norm(grad_b):.4f}")
    
    # 简单的梯度下降步骤
    learning_rate = 0.01
    new_w = w - learning_rate * grad_w
    new_b = b - learning_rate * grad_b
    new_params = (new_w, new_b)
    
    new_loss = loss_function(new_params, x, y_target)
    print(f"📈 Loss after gradient step: {new_loss:.6f} (reduction: {loss_value - new_loss:.6f})")

def demo_multi_device_parallel():
    """演示多设备并行计算"""
    print("\n" + "=" * 50)
    print("🎯 DEMO 3: Multi-Device Parallel Computing")
    print("=" * 50)
    
    device_count = get_npu_device_count()
    print(f"Available NPU devices: {device_count}")
    
    # 显示设备信息
    for i in range(device_count):
        info = get_npu_device_info(i)
        print(f"NPU-{i}: {info['memory_total_gb']} GB, {info['peak_gflops']} GFLOPS, "
              f"Util: {info['memory_utilization']:.1%}")
    
    # 创建批量数据
    batch_size = 16
    matrix_size = 256
    
    key = jax.random.key(456)
    key1, key2 = jax.random.split(key)
    
    batch_a = jax.random.normal(key1, (batch_size, matrix_size, matrix_size))
    batch_b = jax.random.normal(key2, (matrix_size, matrix_size))
    
    print(f"\n📦 Batch data: {batch_a.shape} × {batch_b.shape}")
    
    # 数据并行计算
    parallel_compute = NPUParallelCompute()
    
    print("🔄 Data parallel computation...")
    result_parallel = parallel_compute.data_parallel_matmul(batch_a, batch_b)
    print(f"✅ Parallel result shape: {result_parallel.shape}")
    
    # 模型并行计算
    print("\n🔄 Model parallel computation (output split)...")
    single_a = batch_a[0]  # 取第一个样本
    result_model_parallel = parallel_compute.model_parallel_matmul(
        single_a, batch_b, split_dim="output"
    )
    print(f"✅ Model parallel result shape: {result_model_parallel.shape}")
    
    # 流水线并行计算
    print("\n🔄 Pipeline parallel computation...")
    weights = [
        jax.random.normal(jax.random.key(i), (matrix_size, matrix_size))
        for i in range(min(3, device_count))
    ]
    
    result_pipeline = parallel_compute.pipeline_parallel_matmul(single_a, weights)
    print(f"✅ Pipeline result shape: {result_pipeline.shape}")

def demo_batch_operations():
    """演示批量操作"""
    print("\n" + "=" * 50)
    print("🎯 DEMO 4: Batch Operations")
    print("=" * 50)
    
    # 批量矩阵乘法
    batch_size = 8
    m, k, n = 128, 64, 32
    
    key = jax.random.key(789)
    key1, key2 = jax.random.split(key)
    
    batch_a = jax.random.normal(key1, (batch_size, m, k))
    batch_b = jax.random.normal(key2, (batch_size, k, n))
    
    print(f"Batch A: {batch_a.shape}")
    print(f"Batch B: {batch_b.shape}")
    
    print("🔥 Executing NPU Batch MatMul...")
    batch_result = npu_batch_matmul(batch_a, batch_b, device_id=0)
    
    print(f"✅ Batch result: {batch_result.shape}")
    
    # 验证正确性
    expected_batch = jax.vmap(jnp.dot)(batch_a, batch_b)
    error = jnp.max(jnp.abs(batch_result - expected_batch))
    print(f"✅ Verification: max error = {error:.2e}")

def demo_jit_compilation():
    """演示JIT编译"""
    print("\n" + "=" * 50)
    print("🎯 DEMO 5: JIT Compilation with NPU")
    print("=" * 50)
    
    # 定义一个复杂的NPU函数
    @jax.jit
    def npu_mlp(x, w1, w2, w3):
        """三层MLP使用NPU操作"""
        h1 = jax.nn.relu(npu_matmul(x, w1, device_id=0))
        h2 = jax.nn.relu(npu_matmul(h1, w2, device_id=1 % get_npu_device_count()))
        output = npu_matmul(h2, w3, device_id=2 % get_npu_device_count())
        return output
    
    # 创建测试数据
    key = jax.random.key(999)
    keys = jax.random.split(key, 4)
    
    x = jax.random.normal(keys[0], (64, 128))   # 输入
    w1 = jax.random.normal(keys[1], (128, 256)) # 第一层权重
    w2 = jax.random.normal(keys[2], (256, 128)) # 第二层权重 
    w3 = jax.random.normal(keys[3], (128, 10))  # 输出层权重
    
    print(f"Input: {x.shape}")
    print(f"Weights: {w1.shape}, {w2.shape}, {w3.shape}")
    
    print("🔥 First call (JIT compilation + execution)...")
    import time
    start_time = time.time()
    result1 = npu_mlp(x, w1, w2, w3)
    first_call_time = time.time() - start_time
    
    print("🔥 Second call (cached execution)...")
    start_time = time.time()
    result2 = npu_mlp(x, w1, w2, w3)
    second_call_time = time.time() - start_time
    
    print(f"✅ Output shape: {result1.shape}")
    print(f"⏱️  First call: {first_call_time:.4f}s (includes compilation)")
    print(f"⏱️  Second call: {second_call_time:.4f}s (cached)")
    print(f"🚀 Speedup: {first_call_time / second_call_time:.2f}x")
    
    # 验证结果一致性
    error = jnp.max(jnp.abs(result1 - result2))
    print(f"✅ Results consistency: max error = {error:.2e}")

def main():
    """主演示函数"""
    print("🚀 NPU Simulator Demo - Pure Python Implementation")
    print("=" * 70)
    
    # 初始化NPU运行时
    print("Initializing NPU Runtime...")
    initialize_npu(num_devices=4, memory_per_device=8.0)
    
    try:
        # 运行各种演示
        demo_basic_operations()
        demo_automatic_differentiation() 
        demo_multi_device_parallel()
        demo_batch_operations()
        demo_jit_compilation()
        
        # 性能基准测试
        print("\n" + "=" * 50)
        print("🎯 BENCHMARK: Performance Comparison")
        print("=" * 50)
        benchmark_results = benchmark_npu_parallel_strategies(
            matrix_size=512,
            batch_size=16
        )
        
        print("\n🎉 All demos completed successfully!")
        
    finally:
        # 清理资源
        print("\n🔄 Shutting down NPU Runtime...")
        shutdown_npu()

if __name__ == "__main__":
    main()