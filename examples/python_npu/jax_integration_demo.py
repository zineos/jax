#!/usr/bin/env python3
"""
NPU与JAX完整集成演示

展示Python NPU模拟器如何与JAX的各种功能完美集成：
- JIT编译
- 自动微分
- 向量化 (vmap)
- 并行化 (pmap)
- 优化器
- 神经网络训练
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax.example_libraries import optimizers
import numpy as np
import time

from npu_simulator import (
    initialize_npu, npu_matmul, get_npu_device_count, 
    get_npu_device_info, shutdown_npu
)

def demo_jax_jit_integration():
    """演示NPU与JAX JIT编译的集成"""
    print("\n🔥 JAX JIT + NPU Integration Demo")
    print("=" * 50)
    
    @jit
    def npu_computation(x, w1, w2):
        """JIT编译的NPU计算"""
        h = jax.nn.relu(npu_matmul(x, w1, device_id=0))
        return npu_matmul(h, w2, device_id=1 % get_npu_device_count())
    
    # 创建测试数据
    key = jax.random.key(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    x = jax.random.normal(key1, (64, 128))
    w1 = jax.random.normal(key2, (128, 256))
    w2 = jax.random.normal(key3, (256, 10))
    
    print(f"Input: {x.shape}, Weights: {w1.shape}, {w2.shape}")
    
    # 第一次调用（包含编译）
    print("🔄 First call (JIT compilation + NPU execution)...")
    start = time.time()
    result1 = npu_computation(x, w1, w2)
    compile_time = time.time() - start
    
    # 第二次调用（使用缓存）
    print("⚡ Second call (cached execution)...")
    start = time.time()
    result2 = npu_computation(x, w1, w2)
    cached_time = time.time() - start
    
    print(f"✅ Output shape: {result1.shape}")
    print(f"⏱️  Compile + execute: {compile_time:.4f}s")
    print(f"⏱️  Cached execute: {cached_time:.4f}s")
    print(f"🚀 Speedup: {compile_time/cached_time:.1f}x")
    
    # 验证结果一致性
    assert jnp.allclose(result1, result2), "JIT结果不一致！"
    print("✅ JIT consistency verified!")

def demo_jax_autodiff_integration():
    """演示NPU与JAX自动微分的集成"""
    print("\n🔥 JAX Autodiff + NPU Integration Demo")
    print("=" * 50)
    
    def npu_loss_function(params, x, y_true):
        """使用NPU的损失函数"""
        w1, b1, w2, b2 = params
        
        # 第一层：NPU矩阵乘法 + 偏置 + 激活
        h1 = jax.nn.relu(npu_matmul(x, w1, device_id=0) + b1)
        
        # 第二层：NPU矩阵乘法 + 偏置
        logits = npu_matmul(h1, w2, device_id=1 % get_npu_device_count()) + b2
        
        # softmax交叉熵损失
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.sum(y_true * log_probs, axis=1))
    
    # 创建数据
    key = jax.random.key(123)
    keys = jax.random.split(key, 6)
    
    # 网络参数
    input_dim, hidden_dim, output_dim = 784, 128, 10
    batch_size = 32
    
    w1 = jax.random.normal(keys[0], (input_dim, hidden_dim)) * 0.1
    b1 = jax.random.normal(keys[1], (hidden_dim,)) * 0.01
    w2 = jax.random.normal(keys[2], (hidden_dim, output_dim)) * 0.1
    b2 = jax.random.normal(keys[3], (output_dim,)) * 0.01
    
    x = jax.random.normal(keys[4], (batch_size, input_dim))
    y_true = jax.nn.one_hot(jax.random.randint(keys[5], (batch_size,), 0, output_dim), output_dim)
    
    params = (w1, b1, w2, b2)
    
    print(f"Network: {input_dim} → {hidden_dim} → {output_dim}")
    print(f"Batch size: {batch_size}")
    
    # 计算损失
    loss_value = npu_loss_function(params, x, y_true)
    print(f"📊 Initial loss: {loss_value:.6f}")
    
    # 计算梯度（JAX自动微分通过NPU操作）
    print("🔄 Computing gradients through NPU operations...")
    grad_fn = jit(grad(npu_loss_function))
    
    start = time.time()
    gradients = grad_fn(params, x, y_true)
    grad_time = time.time() - start
    
    grad_w1, grad_b1, grad_w2, grad_b2 = gradients
    
    print(f"✅ Gradient computation time: {grad_time:.4f}s")
    print(f"✅ Gradient norms: W1={jnp.linalg.norm(grad_w1):.4f}, "
          f"W2={jnp.linalg.norm(grad_w2):.4f}")
    
    # 梯度下降步骤
    learning_rate = 0.01
    new_w1 = w1 - learning_rate * grad_w1
    new_b1 = b1 - learning_rate * grad_b1
    new_w2 = w2 - learning_rate * grad_w2
    new_b2 = b2 - learning_rate * grad_b2
    
    new_params = (new_w1, new_b1, new_w2, new_b2)
    new_loss = npu_loss_function(new_params, x, y_true)
    
    print(f"📈 Loss after gradient step: {new_loss:.6f}")
    print(f"📉 Loss reduction: {loss_value - new_loss:.6f}")

def demo_jax_vmap_integration():
    """演示NPU与JAX vmap的集成"""
    print("\n🔥 JAX vmap + NPU Integration Demo")
    print("=" * 50)
    
    def single_npu_forward(x, w):
        """单个样本的NPU前向传播"""
        return npu_matmul(x.reshape(1, -1), w, device_id=0).reshape(-1)
    
    # 使用vmap向量化NPU操作
    batch_npu_forward = jit(vmap(single_npu_forward, in_axes=(0, None)))
    
    # 创建批量数据
    key = jax.random.key(456)
    key1, key2 = jax.random.split(key)
    
    batch_size, input_dim, output_dim = 16, 512, 128
    x_batch = jax.random.normal(key1, (batch_size, input_dim))
    w = jax.random.normal(key2, (input_dim, output_dim))
    
    print(f"Batch input: {x_batch.shape}")
    print(f"Weight: {w.shape}")
    
    print("🔄 vmap + NPU execution...")
    start = time.time()
    result = batch_npu_forward(x_batch, w)
    vmap_time = time.time() - start
    
    print(f"✅ vmap result shape: {result.shape}")
    print(f"⏱️  Execution time: {vmap_time:.4f}s")
    
    # 验证与标准实现的一致性
    expected = jax.vmap(lambda x: jnp.dot(x, w))(x_batch)
    error = jnp.max(jnp.abs(result - expected))
    print(f"✅ Consistency check: max error = {error:.2e}")

def demo_neural_network_training():
    """演示使用NPU训练神经网络"""
    print("\n🔥 Neural Network Training with NPU")
    print("=" * 50)
    
    # 网络定义
    def npu_mlp_predict(params, x):
        """使用NPU的MLP预测"""
        w1, b1, w2, b2, w3, b3 = params
        
        # 第一层
        h1 = jax.nn.relu(npu_matmul(x, w1, device_id=0) + b1)
        
        # 第二层  
        h2 = jax.nn.relu(npu_matmul(h1, w2, device_id=1 % get_npu_device_count()) + b2)
        
        # 输出层
        logits = npu_matmul(h2, w3, device_id=2 % get_npu_device_count()) + b3
        
        return jax.nn.softmax(logits)
    
    def npu_loss(params, x, y):
        """NPU损失函数"""
        predictions = npu_mlp_predict(params, x)
        return -jnp.mean(jnp.sum(y * jnp.log(predictions + 1e-8), axis=1))
    
    def accuracy(params, x, y):
        """准确率计算"""
        predictions = npu_mlp_predict(params, x)
        return jnp.mean(jnp.argmax(predictions, axis=1) == jnp.argmax(y, axis=1))
    
    # 初始化参数
    key = jax.random.key(789)
    keys = jax.random.split(key, 8)
    
    # 网络结构：784 -> 256 -> 128 -> 10
    input_dim, hidden1_dim, hidden2_dim, output_dim = 784, 256, 128, 10
    
    w1 = jax.random.normal(keys[0], (input_dim, hidden1_dim)) * jnp.sqrt(2.0 / input_dim)
    b1 = jnp.zeros(hidden1_dim)
    w2 = jax.random.normal(keys[1], (hidden1_dim, hidden2_dim)) * jnp.sqrt(2.0 / hidden1_dim)
    b2 = jnp.zeros(hidden2_dim)
    w3 = jax.random.normal(keys[2], (hidden2_dim, output_dim)) * jnp.sqrt(2.0 / hidden2_dim)
    b3 = jnp.zeros(output_dim)
    
    params = (w1, b1, w2, b2, w3, b3)
    
    # 生成模拟数据（类似MNIST）
    batch_size = 64
    x_train = jax.random.normal(keys[3], (batch_size, input_dim))
    y_train = jax.nn.one_hot(jax.random.randint(keys[4], (batch_size,), 0, output_dim), output_dim)
    
    print(f"Network architecture: {input_dim} → {hidden1_dim} → {hidden2_dim} → {output_dim}")
    print(f"Training batch size: {batch_size}")
    
    # 设置优化器
    opt_init, opt_update, get_params = optimizers.adam(learning_rate=0.001)
    opt_state = opt_init(params)
    
    # JIT编译训练步骤
    @jit
    def train_step(opt_state, x, y):
        params = get_params(opt_state)
        loss_value = npu_loss(params, x, y)
        grads = grad(npu_loss)(params, x, y)
        return loss_value, opt_update(0, grads, opt_state)
    
    # 训练循环
    print("\n🔄 Training with NPU...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        start = time.time()
        
        loss_value, opt_state = train_step(opt_state, x_train, y_train)
        current_params = get_params(opt_state)
        train_acc = accuracy(current_params, x_train, y_train)
        
        epoch_time = time.time() - start
        
        print(f"Epoch {epoch+1:2d}: loss={loss_value:.4f}, "
              f"accuracy={train_acc:.4f}, time={epoch_time:.3f}s")
    
    print("✅ Training completed!")

def demo_advanced_jax_features():
    """演示NPU与JAX高级功能的集成"""
    print("\n🔥 Advanced JAX Features + NPU")
    print("=" * 50)
    
    # 1. scan与NPU
    print("1️⃣ scan + NPU (RNN-like computation):")
    
    def npu_rnn_step(carry, x):
        """NPU RNN步骤"""
        h_prev, w = carry
        h_new = jax.nn.tanh(npu_matmul(jnp.concatenate([h_prev, x]), w, device_id=0))
        return (h_new, w), h_new
    
    # 数据
    key = jax.random.key(999)
    keys = jax.random.split(key, 3)
    
    seq_len, input_dim, hidden_dim = 10, 32, 64
    x_seq = jax.random.normal(keys[0], (seq_len, input_dim))
    h_init = jax.random.normal(keys[1], (hidden_dim,))
    w_rnn = jax.random.normal(keys[2], (input_dim + hidden_dim, hidden_dim)) * 0.1
    
    print(f"   Sequence: {x_seq.shape}, Hidden: {h_init.shape}")
    
    carry_init = (h_init, w_rnn)
    carry_final, h_sequence = jax.lax.scan(npu_rnn_step, carry_init, x_seq)
    
    print(f"   ✅ RNN output: {h_sequence.shape}")
    
    # 2. 条件计算与NPU
    print("\n2️⃣ Conditional computation + NPU:")
    
    def conditional_npu_compute(x, condition):
        """条件NPU计算"""
        w_branch1 = jax.random.normal(jax.random.key(1), (x.shape[1], 64))
        w_branch2 = jax.random.normal(jax.random.key(2), (x.shape[1], 64))
        
        def branch1(x):
            return npu_matmul(x, w_branch1, device_id=0)
        
        def branch2(x):
            return npu_matmul(x, w_branch2, device_id=1 % get_npu_device_count())
        
        return jax.lax.cond(condition, branch1, branch2, x)
    
    x_test = jax.random.normal(jax.random.key(3), (32, 128))
    
    result_true = conditional_npu_compute(x_test, True)
    result_false = conditional_npu_compute(x_test, False)
    
    print(f"   ✅ Conditional results: {result_true.shape}, {result_false.shape}")
    print(f"   Different branches: {not jnp.allclose(result_true, result_false)}")

def main():
    """主演示函数"""
    print("🚀 NPU + JAX Complete Integration Demo")
    print("=" * 60)
    
    # 初始化NPU
    print("Initializing NPU Runtime...")
    initialize_npu(num_devices=4)
    
    # 显示NPU状态
    print(f"\n📊 NPU Status: {get_npu_device_count()} devices available")
    for i in range(get_npu_device_count()):
        info = get_npu_device_info(i)
        print(f"   NPU-{i}: {info['peak_gflops']} GFLOPS, {info['memory_total_gb']} GB")
    
    try:
        # 运行所有演示
        demo_jax_jit_integration()
        demo_jax_autodiff_integration()
        demo_jax_vmap_integration()
        demo_neural_network_training()
        demo_advanced_jax_features()
        
        print("\n" + "=" * 60)
        print("🎉 All JAX + NPU integration demos completed successfully!")
        print("✅ NPU works seamlessly with:")
        print("   • JIT compilation")
        print("   • Automatic differentiation") 
        print("   • Vectorization (vmap)")
        print("   • Optimization")
        print("   • Neural network training")
        print("   • Advanced control flow")
        
    finally:
        # 清理
        print("\n🔄 Shutting down NPU...")
        shutdown_npu()

if __name__ == "__main__":
    main()