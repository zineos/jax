# 在虚拟 NPU 上直接使用原生 JAX Sharding API

## 🎯 问题回答

> **有没有办法复用原始的 shard API？**

**✅ 完全可以！而且是直接使用，不需要任何包装！**

---

## 💡 核心原理

```
虚拟 NPU Backend = JAX 真实设备的抽象层

虚拟 NPU-0 → jax.devices()[0]
虚拟 NPU-1 → jax.devices()[1]
虚拟 NPU-2 → jax.devices()[2]
虚拟 NPU-3 → jax.devices()[3]
```

因此：
- ✅ **所有 JAX 原生 API 都可以直接使用**
- ✅ **不需要任何包装或修改**
- ✅ **性能完全一致**

---

## 📋 支持的原生 JAX API

| JAX 原生 API | 是否支持 | 说明 |
|-------------|---------|------|
| `jax.pmap` | ✅ 直接用 | 数据并行 |
| `jax.device_put` | ✅ 直接用 | 手动分片 |
| `NamedSharding` | ✅ 直接用 | 命名分片 |
| `Mesh` | ✅ 直接用 | 设备网格 |
| `PartitionSpec` | ✅ 直接用 | 分片规范 |
| `shard_map` | ✅ 直接用 | 手动分片映射 |
| `with_sharding_constraint` | ✅ 直接用 | 分片约束 |
| `jit` (in_shardings) | ✅ 直接用 | JIT 分片 |
| `pjit` | ✅ 直接用 | 并行 JIT |

---

## 🚀 使用示例

### 示例 1: 直接使用 `jax.device_put` + `NamedSharding`

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# 获取虚拟 NPU 设备（实际上就是 JAX 设备）
devices = jax.devices()[:4]

# 创建设备网格 - 原生 JAX API
mesh = Mesh(np.array(devices), axis_names=('x',))

# 定义分片策略 - 原生 JAX API
sharding = NamedSharding(mesh, P('x', None))

# 创建数据
A = jnp.ones((1024, 512))

# 直接使用原生 API 分片！
A_sharded = jax.device_put(A, sharding)  # ← 原生 JAX API！

# 查看分片
print(f"分片策略: {A_sharded.sharding}")
# 输出: NamedSharding(mesh=Mesh(...), spec=PartitionSpec('x', None))
```

**完全原生，零包装！**

### 示例 2: 直接使用 `jax.pmap`

```python
from jax import pmap

# 直接使用原生 pmap - 零修改！
@pmap  # ← 原生 JAX API！
def parallel_matmul(a, b):
    return jnp.matmul(a, b)

# 准备数据（第一维 = 设备数）
A = jnp.ones((4, 256, 512))
B = jnp.ones((4, 512, 256))

# 执行 - 自动在 4 个虚拟 NPU 上并行
result = parallel_matmul(A, B)

print(f"结果形状: {result.shape}")  # (4, 256, 256)
```

**完全原生，零包装！**

### 示例 3: 直接使用 `shard_map`

```python
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

# 创建设备网格
devices = jax.devices()[:4]
mesh = Mesh(np.array(devices), axis_names=('devices',))

# 直接使用原生 shard_map！
@shard_map(  # ← 原生 JAX API！
    mesh=mesh,
    in_specs=(P('devices', None), P(None, None)),
    out_specs=P('devices', None)
)
def manual_sharded_matmul(a_shard, b):
    # a_shard: 每个设备上的数据分片
    # b: 完整数据（复制到所有设备）
    return jnp.matmul(a_shard, b)

A = jnp.ones((512, 256))
B = jnp.ones((256, 128))

# 执行 - JAX 自动处理分片
result = manual_sharded_matmul(A, B)
```

**完全原生，零包装！**

### 示例 4: 直接使用 `with_sharding_constraint`

```python
from jax.lax import with_sharding_constraint

# 创建设备网格
mesh = Mesh(np.array(devices[:4]), axis_names=('batch',))

with mesh:
    @jit  # ← 原生 JAX API！
    def sharded_forward(x, w):
        # 在计算中插入分片约束 - 原生 API！
        x = with_sharding_constraint(x, P('batch', None))
        h = jnp.matmul(x, w)
        h = with_sharding_constraint(h, P('batch', None))
        return jnp.maximum(0, h)
    
    x = jnp.ones((128, 512))
    w = jnp.ones((512, 256))
    
    result = sharded_forward(x, w)
```

**完全原生，零包装！**

### 示例 5: 直接使用 `jit` 的 `in_shardings`

```python
from jax import jit

# 创建分片策略
mesh = Mesh(np.array(devices[:4]), axis_names=('x',))
sharding = NamedSharding(mesh, P('x', None))

# 在 jit 中直接指定分片 - 原生 API！
@jit(  # ← 原生 JAX API！
    in_shardings=(sharding, None),
    out_shardings=sharding
)
def sharded_matmul(a, b):
    return jnp.matmul(a, b)

A = jnp.ones((1024, 512))
B = jnp.ones((512, 256))

result = sharded_matmul(A, B)
```

**完全原生，零包装！**

### 示例 6: 直接使用二维 `Mesh`

```python
# 创建 2×2 设备网格 - 原生 API！
mesh = Mesh(
    np.array(devices[:4]).reshape(2, 2),
    axis_names=('data', 'model')
)

# 定义分片策略
x_sharding = NamedSharding(mesh, P('data', None))
w_sharding = NamedSharding(mesh, P(None, 'model'))

# 分片数据 - 原生 API！
x = jnp.ones((256, 512))
w = jnp.ones((512, 1024))

x_sharded = jax.device_put(x, x_sharding)
w_sharded = jax.device_put(w, w_sharding)

# 计算
@jit
def compute(x, w):
    return jnp.matmul(x, w)

result = compute(x_sharded, w_sharded)
```

**完全原生，零包装！**

---

## 🎓 完整训练示例（全原生 API）

```python
import jax
import jax.numpy as jnp
from jax import random, pmap, grad

# 获取虚拟 NPU 设备
devices = jax.devices()[:4]

# 定义模型（原生 JAX）
def model(params, x):
    w, b = params
    return jnp.matmul(x, w) + b

def loss_fn(params, x, y):
    pred = model(params, x)
    return jnp.mean((pred - y) ** 2)

# 定义训练步骤 - 原生 pmap！
@pmap  # ← 原生 JAX API！
def train_step(params, x, y, lr):
    # 计算梯度 - 原生 API！
    grads = grad(loss_fn)(params, x, y)
    
    # 跨设备平均 - 原生 API！
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # 更新参数
    new_params = jax.tree_map(
        lambda p, g: p - lr * g,
        params, grads
    )
    
    return new_params

# 初始化参数（复制到所有设备）
key = random.PRNGKey(0)
params = [
    jnp.stack([random.normal(key, (512, 256))] * 4),  # w
    jnp.stack([jnp.zeros(256)] * 4)  # b
]

# 训练数据
x = random.normal(key, (4, 32, 512))  # (4设备, 32样本, 512特征)
y = random.normal(key, (4, 32, 256))

# 训练循环 - 全原生 JAX！
for epoch in range(100):
    params = train_step(params, x, y, lr=0.01)
    
    if epoch % 10 == 0:
        # 计算损失
        losses = jax.vmap(loss_fn, in_axes=(0, 0, 0))(params, x, y)
        print(f"Epoch {epoch}, Loss: {losses.mean():.4f}")
```

**完全原生 JAX 代码，零包装，零修改！**

---

## 📊 原生 API 完整列表

### 分片相关

| API | 包 | 功能 |
|-----|-----|------|
| `Mesh` | `jax.sharding` | 设备网格 |
| `PartitionSpec` | `jax.sharding` | 分片规范 |
| `NamedSharding` | `jax.sharding` | 命名分片 |
| `PositionalSharding` | `jax.sharding` | 位置分片 |
| `device_put` | `jax` | 手动放置数据 |
| `with_sharding_constraint` | `jax.lax` | 插入分片约束 |

### 并行相关

| API | 包 | 功能 |
|-----|-----|------|
| `pmap` | `jax` | 数据并行 |
| `shard_map` | `jax.experimental.shard_map` | 手动分片映射 |
| `pjit` | `jax.experimental.pjit` | 并行 JIT |
| `jit(in_shardings=...)` | `jax` | JIT 分片 |

### 通信相关

| API | 包 | 功能 |
|-----|-----|------|
| `lax.pmean` | `jax.lax` | 跨设备平均 |
| `lax.psum` | `jax.lax` | 跨设备求和 |
| `lax.pmax` | `jax.lax` | 跨设备最大值 |
| `lax.pmin` | `jax.lax` | 跨设备最小值 |
| `lax.all_gather` | `jax.lax` | 收集所有数据 |
| `lax.all_to_all` | `jax.lax` | 全对全通信 |

**所有这些 API 都可以在虚拟 NPU 上直接使用！**

---

## 🔍 对比：包装 vs 原生

### ❌ 不需要包装（之前的方式）

```python
# 之前可能以为需要这样包装
class NPUDistributedCompute:
    def sharded_matmul(self, A, B):
        # 包装 JAX API
        mesh = self.device_manager.get_mesh(...)
        sharding = NamedSharding(mesh, ...)
        A_sharded = jax.device_put(A, sharding)
        # ...
```

### ✅ 直接使用原生 API（正确方式）

```python
# 实际上可以直接使用原生 JAX API！
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices()[:4]  # 这些就是虚拟 NPU
mesh = Mesh(np.array(devices), ('x',))
sharding = NamedSharding(mesh, P('x', None))

A_sharded = jax.device_put(A, sharding)  # 直接用！
```

---

## 💡 关键洞察

### 1. 虚拟 NPU = JAX 设备

```python
# 获取虚拟 NPU 设备
virtual_npu_devices = jax.devices()

# 这些设备可以是：
# - CPU 设备（开发/测试）
# - GPU 设备（如果有的话）
# - TPU 设备（如果有的话）

# 无论底层是什么，都可以当作虚拟 NPU 使用
```

### 2. 不需要任何转换

```python
# JAX 代码
@pmap
def my_function(x):
    return x * 2

# 在虚拟 NPU 上运行 - 完全相同的代码！
@pmap
def my_function(x):
    return x * 2

# 零修改！
```

### 3. NPU 特性是附加的

```python
# 基础：使用原生 JAX API
@pmap
def compute(x, w):
    return jnp.matmul(x, w)

# 附加：添加 NPU 特定优化（可选）
@pmap
def compute_with_npu_features(x, w):
    # NPU 量化（额外功能）
    x_int8 = quantize_int8(x)
    w_int8 = quantize_int8(w)
    
    # 使用原生 JAX 计算
    return jnp.matmul(x_int8, w_int8)
```

---

## 🎯 最佳实践

### 1. 直接使用原生 API

```python
# ✅ 推荐：直接使用
from jax import pmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

@pmap
def my_compute(x):
    return x * 2

# ❌ 不推荐：不必要的包装
class MyWrapper:
    def pmap_compute(self, x):
        @pmap
        def inner(x):
            return x * 2
        return inner(x)
```

### 2. 利用类型提示

```python
import jax
from jax import Array
from jax.sharding import Sharding

def process_on_npu(
    data: Array,
    sharding: Sharding
) -> Array:
    """直接使用 JAX 类型"""
    return jax.device_put(data, sharding)
```

### 3. 组合 NPU 特性和原生 API

```python
from jax import pmap

# NPU 特性：量化函数
def quantize_int8(x):
    # NPU 专用量化
    scale = jnp.max(jnp.abs(x)) / 127.0
    return jnp.round(x / scale).astype(jnp.int8)

# 原生 JAX API：分布式计算
@pmap
def distributed_quantized_compute(x, w):
    # 组合：NPU 特性 + 原生 JAX
    x_q = quantize_int8(x)  # NPU 特性
    w_q = quantize_int8(w)  # NPU 特性
    return jnp.matmul(x_q, w_q)  # 原生 JAX
```

---

## 📚 学习资源

### JAX 官方文档

- [JAX Sharding Guide](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [JAX pmap Tutorial](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)
- [JAX shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html)

### 示例代码

- `examples/npu_use_native_jax_api.py` - 完整的原生 API 示例
- `examples/npu_with_jax_sharding.py` - 分布式计算示例

---

## ✅ 总结

### 问题：有没有办法复用原始的 shard API？

**答案：完全可以！而且是直接使用！**

| 特性 | 是否支持 |
|------|---------|
| ✅ 所有原生 JAX Sharding API | 直接用 |
| ✅ pmap, shard_map, pjit | 直接用 |
| ✅ Mesh, PartitionSpec, NamedSharding | 直接用 |
| ✅ with_sharding_constraint | 直接用 |
| ✅ 集合通信（pmean, psum等） | 直接用 |
| ✅ JIT 编译优化 | 直接用 |
| ✅ 自动微分 | 直接用 |

### 核心原理

```
虚拟 NPU Backend = JAX 真实设备的抽象

因此：
  原生 JAX API → 在虚拟 NPU 上 → 完全兼容 ✅
```

### 关键优势

1. **零学习成本** - 直接使用 JAX 文档
2. **零移植成本** - JAX 代码直接运行
3. **零性能损失** - 没有额外包装开销
4. **完整生态** - JAX 全部功能都可用

---

**创建日期**: 2025-10-15  
**文件**: `examples/npu_use_native_jax_api.py` (完整代码)  

🎉 **虚拟 NPU 完全兼容 JAX 原生 API！**
