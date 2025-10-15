#!/usr/bin/env python3
"""
创建虚拟NPU Backend - 直接替代CPU

核心思路：
1. 使用JAX的CPU backend作为底层计算引擎
2. 将其包装成"虚拟NPU" backend
3. 所有JAX原生API（pmap, Sharding, Mesh等）直接可用
4. 可以构建2D mesh等任意拓扑

这样做的优势：
- 虚拟NPU = 完整的JAX backend
- 复用所有JAX API，零修改
- 可以添加NPU特性（量化等）

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from jax import jit, pmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src import xla_bridge
import numpy as np


print("=" * 80)
print("虚拟NPU Backend - 直接替代CPU")
print("=" * 80)


# ============================================================================
# 方法：使用JAX的CPU backend作为虚拟NPU
# ============================================================================

def setup_virtual_npu_backend():
    """
    设置虚拟NPU backend
    
    关键思路：
    1. JAX已经有CPU backend（xla_client.Client）
    2. 我们直接使用它，但在逻辑上当作"虚拟NPU"
    3. 所有JAX API都能直接用
    """
    
    print("\n1️⃣  虚拟NPU Backend设置")
    print("-" * 80)
    
    # 获取CPU backend（这就是我们的虚拟NPU的底层）
    cpu_backend = jax.devices('cpu')
    
    print(f"底层backend: CPU")
    print(f"虚拟NPU设备数: {len(cpu_backend)}")
    print(f"\n虚拟NPU设备列表（概念上）:")
    
    # 将CPU设备"重命名"为虚拟NPU
    virtual_npu_devices = []
    for i, cpu_dev in enumerate(cpu_backend):
        print(f"  虚拟NPU-{i} (底层: {cpu_dev})")
        virtual_npu_devices.append(cpu_dev)
    
    return virtual_npu_devices


# ============================================================================
# 虚拟NPU管理器
# ============================================================================

class VirtualNPUBackend:
    """
    虚拟NPU Backend管理器
    
    这个类：
    - 管理虚拟NPU设备
    - 提供NPU特性（量化等）
    - 但底层使用JAX CPU backend
    """
    
    def __init__(self, num_virtual_npus: int = None):
        """
        初始化虚拟NPU backend
        
        Args:
            num_virtual_npus: 虚拟NPU数量（默认=CPU核心数）
        """
        # 获取CPU设备作为虚拟NPU
        cpu_devices = jax.devices('cpu')
        
        if num_virtual_npus is None:
            num_virtual_npus = len(cpu_devices)
        
        # 虚拟NPU设备（底层是CPU）
        self.devices = cpu_devices[:num_virtual_npus]
        self.num_devices = len(self.devices)
        
        print(f"\n✓ 虚拟NPU Backend初始化完成")
        print(f"  虚拟NPU数量: {self.num_devices}")
        print(f"  底层实现: JAX CPU backend")
    
    def get_devices(self):
        """获取虚拟NPU设备（返回JAX设备，可直接用于所有JAX API）"""
        return self.devices
    
    def create_2d_mesh(self, rows: int, cols: int, axis_names=('x', 'y')):
        """
        创建2D NPU mesh
        
        这个mesh可以直接用于：
        - NamedSharding
        - shard_map
        - pjit
        等所有JAX分布式API
        """
        if rows * cols > self.num_devices:
            raise ValueError(
                f"需要{rows}×{cols}={rows*cols}个NPU，"
                f"但只有{self.num_devices}个"
            )
        
        # 创建JAX Mesh（使用虚拟NPU设备）
        device_array = np.array(self.devices[:rows*cols]).reshape(rows, cols)
        mesh = Mesh(device_array, axis_names)
        
        print(f"\n✓ 创建2D NPU Mesh: {rows}×{cols}")
        print(f"  轴名称: {axis_names}")
        print(f"  总设备数: {mesh.size}")
        
        return mesh
    
    # NPU特性：量化
    @staticmethod
    def quantize_int8(x):
        """INT8量化（NPU特性）"""
        scale = jnp.max(jnp.abs(x)) / 127.0
        quantized = jnp.round(x / scale).astype(jnp.int8)
        return quantized.astype(jnp.float32) * scale
    
    @staticmethod
    def quantize_fp16(x):
        """FP16量化（NPU特性）"""
        return x.astype(jnp.float16).astype(jnp.float32)


# ============================================================================
# 示例1: 基础使用 - 所有JAX API直接可用
# ============================================================================

def demo_basic_usage():
    """基础使用：所有JAX API直接可用"""
    print("\n" + "=" * 80)
    print("【示例1】基础使用 - 所有JAX API直接可用")
    print("=" * 80)
    
    # 创建虚拟NPU backend
    npu = VirtualNPUBackend(num_virtual_npus=4)
    devices = npu.get_devices()
    
    print("\n测试1: jit编译")
    @jit
    def compute(x):
        return x * 2 + 1
    
    x = jnp.ones(100)
    result = compute(x)
    print(f"  ✅ jit可用: {result.shape}")
    
    print("\n测试2: pmap数据并行")
    @pmap
    def parallel_compute(x):
        return x ** 2
    
    x = jnp.ones((4, 100))
    result = parallel_compute(x)
    print(f"  ✅ pmap可用: {result.shape}")
    print(f"  数据自动分配到4个虚拟NPU")
    
    print("\n测试3: 自动微分")
    from jax import grad
    
    def loss(x):
        return jnp.sum(x ** 2)
    
    grad_fn = grad(loss)
    x = jnp.array([1.0, 2.0, 3.0])
    grads = grad_fn(x)
    print(f"  ✅ 自动微分可用: {grads}")
    
    print("\n✓ 所有基础JAX API都能直接使用！")


# ============================================================================
# 示例2: 2D Mesh（完全原生）
# ============================================================================

def demo_2d_mesh():
    """2D Mesh - 完全使用JAX原生API"""
    print("\n" + "=" * 80)
    print("【示例2】2D NPU Mesh - 完全原生JAX API")
    print("=" * 80)
    
    # 创建虚拟NPU backend
    npu = VirtualNPUBackend(num_virtual_npus=4)
    
    # 创建2×2 mesh
    mesh = npu.create_2d_mesh(rows=2, cols=2, axis_names=('data', 'model'))
    
    print("\n使用这个mesh进行2D分片:")
    
    # 准备数据
    from jax import random
    key = random.PRNGKey(0)
    A = random.normal(key, (512, 256))
    B = random.normal(key, (256, 128))
    
    # 2D分片 - 完全原生API！
    sharding_A = NamedSharding(mesh, P('data', None))
    sharding_B = NamedSharding(mesh, P(None, 'model'))
    
    A_sharded = jax.device_put(A, sharding_A)
    B_sharded = jax.device_put(B, sharding_B)
    
    print(f"  A: {A.shape} → 按'data'维分片")
    print(f"  B: {B.shape} → 按'model'维分片")
    
    # 矩阵乘法 - 自动处理2D分片
    @jit
    def matmul(a, b):
        return jnp.matmul(a, b)
    
    C = matmul(A_sharded, B_sharded)
    
    print(f"  C: {C.shape} → 自动2D分片")
    print(f"  C的分片: {C.sharding}")
    
    print("\n✓ 完全使用JAX原生Sharding API！")


# ============================================================================
# 示例3: shard_map（高级）
# ============================================================================

def demo_shard_map():
    """shard_map - 手动控制分片"""
    print("\n" + "=" * 80)
    print("【示例3】shard_map - 手动分片控制")
    print("=" * 80)
    
    from jax.experimental.shard_map import shard_map
    
    # 创建虚拟NPU backend
    npu = VirtualNPUBackend(num_virtual_npus=4)
    mesh = npu.create_2d_mesh(2, 2, ('x', 'y'))
    
    print("\n使用shard_map:")
    
    # 定义分片计算 - 完全原生API！
    @shard_map(
        mesh=mesh,
        in_specs=(P('x', None), P(None, 'y')),
        out_specs=P('x', 'y')
    )
    def sharded_matmul(a_shard, b_shard):
        # 每个虚拟NPU上的局部计算
        return jnp.matmul(a_shard, b_shard)
    
    # 准备数据
    from jax import random
    key = random.PRNGKey(42)
    A = random.normal(key, (400, 200))
    B = random.normal(key, (200, 100))
    
    # 执行
    C = sharded_matmul(A, B)
    
    print(f"  输入A: {A.shape}")
    print(f"  输入B: {B.shape}")
    print(f"  输出C: {C.shape}")
    print(f"  ✅ shard_map完全可用！")


# ============================================================================
# 示例4: 混合并行（数据 × 模型）
# ============================================================================

def demo_hybrid_parallelism():
    """混合并行：数据并行 × 模型并行"""
    print("\n" + "=" * 80)
    print("【示例4】混合并行（数据 × 模型）")
    print("=" * 80)
    
    # 创建虚拟NPU backend
    npu = VirtualNPUBackend(num_virtual_npus=4)
    mesh = npu.create_2d_mesh(2, 2, ('data', 'model'))
    
    print("\n神经网络层：混合并行")
    
    # 网络参数
    batch_size = 64
    input_dim = 512
    hidden_dim = 1024
    
    # 准备数据
    from jax import random
    key = random.PRNGKey(0)
    k1, k2, k3 = random.split(key, 3)
    
    x = random.normal(k1, (batch_size, input_dim))
    w = random.normal(k2, (input_dim, hidden_dim))
    b = random.normal(k3, (hidden_dim,))
    
    # 混合分片策略 - 完全原生API！
    x_sharding = NamedSharding(mesh, P('data', None))
    w_sharding = NamedSharding(mesh, P(None, 'model'))
    b_sharding = NamedSharding(mesh, P('model',))
    
    x_sharded = jax.device_put(x, x_sharding)
    w_sharded = jax.device_put(w, w_sharding)
    b_sharded = jax.device_put(b, b_sharding)
    
    # 前向传播
    @jit
    def forward(x, w, b):
        h = jnp.matmul(x, w) + b
        return jnp.maximum(0, h)  # ReLU
    
    output = forward(x_sharded, w_sharded, b_sharded)
    
    print(f"  批次大小: {batch_size}")
    print(f"  输入维度: {input_dim}")
    print(f"  隐藏维度: {hidden_dim}")
    print(f"\n  分片策略:")
    print(f"    x: P('data', None) → 数据并行")
    print(f"    w: P(None, 'model') → 模型并行")
    print(f"    输出: 自动2D分片")
    print(f"\n  输出形状: {output.shape}")
    print(f"  输出分片: {output.sharding}")
    print(f"\n✓ 混合并行完全可用！")


# ============================================================================
# 示例5: NPU特性 + JAX API
# ============================================================================

def demo_npu_features():
    """NPU特性（量化）+ JAX原生API"""
    print("\n" + "=" * 80)
    print("【示例5】NPU特性（量化）+ JAX原生API")
    print("=" * 80)
    
    # 创建虚拟NPU backend
    npu = VirtualNPUBackend(num_virtual_npus=4)
    mesh = npu.create_2d_mesh(2, 2, ('x', 'y'))
    
    print("\nNPU量化 + 2D分片:")
    
    # 准备数据
    from jax import random
    key = random.PRNGKey(99)
    k1, k2 = random.split(key)
    
    A = random.normal(k1, (512, 256))
    B = random.normal(k2, (256, 128))
    
    # 2D分片
    A_sharded = jax.device_put(A, NamedSharding(mesh, P('x', None)))
    B_sharded = jax.device_put(B, NamedSharding(mesh, P(None, 'y')))
    
    # NPU计算：量化 + 矩阵乘法
    @jit
    def npu_matmul(a, b):
        # NPU特性：INT8量化
        a_q = npu.quantize_int8(a)
        b_q = npu.quantize_int8(b)
        return jnp.matmul(a_q, b_q)
    
    result = npu_matmul(A_sharded, B_sharded)
    
    print(f"  输入A: {A.shape} → 2D分片")
    print(f"  输入B: {B.shape} → 2D分片")
    print(f"  NPU特性: INT8量化")
    print(f"  输出: {result.shape}")
    print(f"  输出分片: {result.sharding}")
    print(f"\n✓ NPU特性与JAX原生API完美结合！")


# ============================================================================
# 示例6: 完整训练循环
# ============================================================================

def demo_training_loop():
    """完整的训练循环"""
    print("\n" + "=" * 80)
    print("【示例6】完整训练循环 - 虚拟NPU上训练")
    print("=" * 80)
    
    # 创建虚拟NPU backend
    npu = VirtualNPUBackend(num_virtual_npus=4)
    
    print("\n使用pmap进行数据并行训练:")
    
    # 定义模型
    def model(params, x):
        w, b = params
        return jnp.matmul(x, w) + b
    
    def loss_fn(params, x, y):
        pred = model(params, x)
        return jnp.mean((pred - y) ** 2)
    
    # 训练步骤 - 完全原生pmap！
    @pmap
    def train_step(params, x, y, lr):
        from jax import grad, value_and_grad
        
        # 计算损失和梯度
        loss, grads = value_and_grad(loss_fn)(params, x, y)
        
        # 跨设备平均梯度
        grads = jax.lax.pmean(grads, axis_name='batch')
        
        # 更新参数
        new_params = jax.tree_map(
            lambda p, g: p - lr * g,
            params, grads
        )
        
        return new_params, loss
    
    # 初始化参数（复制到所有虚拟NPU）
    from jax import random
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    params = [
        jnp.stack([random.normal(k1, (512, 256))] * 4),
        jnp.stack([jnp.zeros(256)] * 4)
    ]
    
    # 训练数据
    x = random.normal(k2, (4, 32, 512))
    y = random.normal(k2, (4, 32, 256))
    
    print(f"  虚拟NPU数: 4")
    print(f"  每NPU批次: 32")
    print(f"  总批次: 128")
    print(f"  模型: 512 → 256")
    
    # 训练
    print(f"\n开始训练...")
    for epoch in range(5):
        params, loss = train_step(params, x, y, lr=0.01)
        print(f"  Epoch {epoch}: Loss = {jnp.mean(loss):.6f}")
    
    print(f"\n✓ 完整训练循环在虚拟NPU上运行成功！")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    
    print("\n💡 核心思路:")
    print("  虚拟NPU = JAX CPU backend（概念上）")
    print("  所有JAX原生API直接可用，无需任何修改")
    print("  可以添加NPU特性（量化等）")
    print("  可以构建任意拓扑（2D mesh、3D mesh等）\n")
    
    try:
        # 基础演示
        demo_basic_usage()
        
        # 2D Mesh
        demo_2d_mesh()
        
        # 高级功能
        demo_shard_map()
        demo_hybrid_parallelism()
        
        # NPU特性
        demo_npu_features()
        
        # 训练
        demo_training_loop()
        
        print("\n" + "=" * 80)
        print("✅ 所有演示完成！")
        print("=" * 80)
        
        print("\n📚 总结:")
        print("  1. 虚拟NPU = JAX CPU backend（底层）")
        print("  2. 所有JAX API直接可用（jit, pmap, Sharding, shard_map）")
        print("  3. 可以构建2D/3D mesh，任意拓扑")
        print("  4. 可以添加NPU特性（量化、专用算子）")
        print("  5. 零修改，零包装，完全原生")
        
        print("\n🎯 关键代码:")
        print("  # 创建虚拟NPU backend")
        print("  npu = VirtualNPUBackend(num_virtual_npus=4)")
        print("  ")
        print("  # 创建2D mesh")
        print("  mesh = npu.create_2d_mesh(2, 2, ('data', 'model'))")
        print("  ")
        print("  # 使用所有JAX原生API")
        print("  sharding = NamedSharding(mesh, P('data', 'model'))")
        print("  data_sharded = jax.device_put(data, sharding)")
        print("  ")
        print("  # pmap, shard_map, jit... 全部直接可用！")
        
    except Exception as e:
        print(f"\n⚠️  需要JAX环境")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
