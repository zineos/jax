# NPU Simulator for JAX - Pure Python Implementation

这是一个**完全基于Python**的NPU（神经处理单元）模拟器，专为JAX设计，无需C++扩展。

## 🌟 特性

### ✅ 完整的JAX集成
- **JIT编译支持**: NPU操作可以被JAX JIT编译，获得优化性能
- **自动微分**: 支持`jax.grad`，可以计算NPU操作的梯度
- **向量化**: 支持`jax.vmap`，批量执行NPU操作
- **并行化**: 支持多设备并行计算策略
- **优化器兼容**: 与JAX优化器完美配合

### 🔥 NPU核心功能
- 矩阵乘法操作（标准版和优化版）
- 批量矩阵乘法
- 多设备管理和负载均衡
- 内存管理模拟
- 性能基准测试

### 🚀 并行策略
- **数据并行**: 将批次数据分布到多个NPU设备
- **模型并行**: 将模型参数分割到多个设备
- **流水线并行**: 多层网络的流水线执行

## 🛠 安装使用

### 环境要求
```bash
pip install jax numpy
```

### 快速开始
```python
import jax
import jax.numpy as jnp
from npu_simulator import initialize_npu, npu_matmul

# 1. 初始化NPU运行时
initialize_npu(num_devices=4)

# 2. 创建测试数据
a = jax.random.normal(jax.random.key(0), (512, 256))
b = jax.random.normal(jax.random.key(1), (256, 128))

# 3. NPU矩阵乘法
result = npu_matmul(a, b, device_id=0)
print(f"Result: {result.shape}")

# 4. 支持JAX的所有变换
@jax.jit
def npu_computation(x, w):
    return jax.nn.relu(npu_matmul(x, w))

# 5. 自动微分
grad_fn = jax.grad(lambda w: jnp.sum(npu_matmul(a, w)))
gradients = grad_fn(b)
```

## 📋 完整示例

### 基本NPU操作
```python
from npu_simulator import *

# 初始化NPU
initialize_npu(num_devices=4, memory_per_device=8.0)

# 基本矩阵乘法
a = jax.random.normal(jax.random.key(0), (1024, 512))
b = jax.random.normal(jax.random.key(1), (512, 256))

result = npu_matmul(a, b, device_id=0)
print(f"NPU MatMul result: {result.shape}")

# 批量矩阵乘法
batch_a = jax.random.normal(jax.random.key(2), (16, 256, 128))
batch_b = jax.random.normal(jax.random.key(3), (16, 128, 64))

batch_result = npu_batch_matmul(batch_a, batch_b, device_id=1)
print(f"Batch MatMul result: {batch_result.shape}")
```

### JAX JIT编译
```python
@jax.jit
def npu_neural_network(x, w1, w2, w3):
    """三层神经网络，使用NPU加速"""
    h1 = jax.nn.relu(npu_matmul(x, w1, device_id=0))
    h2 = jax.nn.relu(npu_matmul(h1, w2, device_id=1))
    output = npu_matmul(h2, w3, device_id=2)
    return output

# 创建网络参数
x = jax.random.normal(jax.random.key(0), (64, 784))
w1 = jax.random.normal(jax.random.key(1), (784, 256))
w2 = jax.random.normal(jax.random.key(2), (256, 128))
w3 = jax.random.normal(jax.random.key(3), (128, 10))

# JIT编译并执行
result = npu_neural_network(x, w1, w2, w3)  # 第一次调用：编译+执行
result = npu_neural_network(x, w1, w2, w3)  # 后续调用：直接执行缓存版本
```

### 自动微分训练
```python
def npu_loss_function(params, x, y_true):
    """使用NPU的损失函数"""
    w1, b1, w2, b2 = params
    
    # 前向传播使用NPU
    hidden = jax.nn.relu(npu_matmul(x, w1, device_id=0) + b1)
    logits = npu_matmul(hidden, w2, device_id=1) + b2
    
    # 计算损失
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y_true * log_probs, axis=1))

# 参数初始化
key = jax.random.key(42)
keys = jax.random.split(key, 4)

w1 = jax.random.normal(keys[0], (784, 128)) * 0.1
b1 = jnp.zeros(128)
w2 = jax.random.normal(keys[1], (128, 10)) * 0.1
b2 = jnp.zeros(10)

params = (w1, b1, w2, b2)

# 训练数据
x = jax.random.normal(keys[2], (64, 784))
y = jax.nn.one_hot(jax.random.randint(keys[3], (64,), 0, 10), 10)

# 计算梯度（通过NPU操作）
grad_fn = jax.grad(npu_loss_function)
gradients = grad_fn(params, x, y)

print("✅ 梯度计算成功，支持通过NPU操作进行反向传播！")
```

### 并行策略
```python
from parallel_strategies import NPUParallelCompute

# 创建并行计算管理器
parallel_compute = NPUParallelCompute()

# 数据并行
batch_a = jax.random.normal(jax.random.key(0), (32, 512, 256))
batch_b = jax.random.normal(jax.random.key(1), (256, 128))

data_parallel_result = parallel_compute.data_parallel_matmul(
    batch_a, batch_b, devices=[0, 1, 2, 3]
)

# 模型并行
model_parallel_result = parallel_compute.model_parallel_matmul(
    batch_a[0], batch_b, split_dim="output", devices=[0, 1]
)

# 流水线并行
weights = [
    jax.random.normal(jax.random.key(i), (512, 512)) 
    for i in range(3)
]

pipeline_result = parallel_compute.pipeline_parallel_matmul(
    batch_a[0], weights, devices=[0, 1, 2]
)
```

## 🎯 运行演示

### 基础演示
```bash
cd examples/python_npu
python demo.py
```

### JAX集成演示
```bash
python jax_integration_demo.py
```

输出示例：
```
🚀 NPU Simulator Demo - Pure Python Implementation
======================================================================
Initializing NPU Runtime...
🚀 Initializing NPU Runtime with 4 devices...
✅ NPU-0 initialized: 8.0GB memory, 128 compute units, 100.0 GFLOPS
✅ NPU-1 initialized: 8.0GB memory, 128 compute units, 110.0 GFLOPS
✅ NPU-2 initialized: 8.0GB memory, 128 compute units, 120.0 GFLOPS
✅ NPU-3 initialized: 8.0GB memory, 128 compute units, 130.0 GFLOPS
✅ NPU Runtime initialized successfully!

🔥 JAX JIT + NPU Integration Demo
==================================================
Input: (64, 128), Weights: (128, 256), (256, 10)
🔄 First call (JIT compilation + NPU execution)...
🔥 NPU-0 executing MatMul: (64×128) × (128×256) = 4.19 GFLOPS, estimated 41.94ms
🔥 NPU-1 executing MatMul: (64×256) × (256×10) = 0.33 GFLOPS, estimated 3.01ms
⚡ Second call (cached execution)...
🔥 NPU-0 executing MatMul: (64×128) × (128×256) = 4.19 GFLOPS, estimated 41.94ms
🔥 NPU-1 executing MatMul: (64×256) × (256×10) = 0.33 GFLOPS, estimated 3.01ms
✅ Output shape: (64, 10)
⏱️  Compile + execute: 0.0523s
⏱️  Cached execute: 0.0451s
🚀 Speedup: 1.2x
✅ JIT consistency verified!
```

## 🏗 架构设计

### 核心组件

1. **NPUDevice**: 单个NPU设备模拟器
   - 内存管理
   - 计算能力模拟
   - 状态监控

2. **NPURuntime**: NPU运行时管理器
   - 多设备管理
   - 任务调度
   - 线程池管理

3. **JAX Integration**: JAX集成层
   - `pure_callback`实现自定义计算
   - `custom_vjp`实现梯度计算
   - 完整支持JAX变换

### 关键技术

- **pure_callback**: 让JAX调用Python函数
- **custom_vjp**: 定义前向和反向传播
- **JIT兼容**: 所有操作都支持JIT编译
- **设备抽象**: 模拟真实NPU的行为特性

## 🔍 为什么选择Python实现？

### ✅ 优势
- **开发速度快**: 无需C++编译，快速迭代
- **调试方便**: Python调试工具丰富
- **可读性强**: 代码逻辑清晰，易于理解和修改
- **集成简单**: 与JAX生态系统无缝集成
- **跨平台**: 任何支持Python的平台都可运行

### ⚡ 性能
- 通过JAX JIT编译获得接近原生性能
- 支持并行计算，充分利用多核CPU
- 可以作为C++实现的原型和验证

### 🎯 适用场景
- **算法验证**: 快速验证NPU算法的正确性
- **架构探索**: 尝试不同的NPU架构设计
- **教学演示**: 理解NPU工作原理
- **原型开发**: 在投入大量C++开发前验证概念

## 📊 性能基准

```python
# 运行性能基准测试
from parallel_strategies import benchmark_npu_parallel_strategies

results = benchmark_npu_parallel_strategies(
    matrix_size=512, 
    batch_size=16,
    num_devices=4
)

# 输出示例：
# data_parallel         : 0.156s, (16, 512, 512)  Throughput: 17.35 GFLOPS
# model_parallel_output : 0.089s, (512, 512)      Throughput: 19.23 GFLOPS  
# pipeline_parallel     : 0.201s, (512, 512)      Layers: 3
```

## 🚀 扩展方向

这个Python NPU模拟器可以轻松扩展：

1. **添加新操作**: 卷积、池化、注意力机制等
2. **优化算法**: 实现特定的NPU优化策略
3. **硬件特性**: 模拟更精确的硬件行为
4. **分布式计算**: 跨机器的NPU集群
5. **可视化工具**: 实时监控NPU状态和性能

## 🎉 总结

这个**纯Python的NPU模拟器完美集成了JAX的所有核心功能**：

- ✅ JIT编译加速
- ✅ 自动微分支持  
- ✅ 向量化操作
- ✅ 并行计算策略
- ✅ 神经网络训练
- ✅ 高级控制流

**无需任何C++代码**，就能实现功能完整的自定义NPU，并在JAX生态系统中无缝使用！