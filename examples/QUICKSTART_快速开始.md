# ⚡ 快速开始 - 虚拟 NPU Backend

## 🎯 三句话总结

1. **JAX 支持添加虚拟 backend** - 使用 `register_backend_factory()`
2. **虚拟 NPU = JAX 设备 + NPU 特性** - 完全兼容原生 JAX API
3. **所有 JAX 功能都能用** - pmap, Sharding, JIT, grad 等全部支持

---

## ⚡ 30 秒快速开始

### 方案 A：只需要设备管理（不计算）

```bash
python3 examples/simulate_gpu_cluster.py
python3 examples/simulate_npu_cluster.py
```

### 方案 B：需要计算 + 分布式（需要 JAX）

```bash
pip install jax  # 如果没有安装
python3 examples/npu_use_native_jax_api.py
```

---

## 💡 核心代码（5 分钟上手）

### 步骤 1: 映射虚拟 NPU 到 JAX 设备

```python
import jax

# 获取 JAX 设备（CPU/GPU）
devices = jax.devices()

# 这些设备就是你的虚拟 NPU！
print(f"虚拟 NPU 设备: {devices}")
# 输出: [CpuDevice(id=0), CpuDevice(id=1), ...]
```

### 步骤 2: 直接使用原生 JAX Sharding API

```python
from jax import pmap
import jax.numpy as jnp

# 直接使用 pmap - 零包装！
@pmap
def parallel_matmul(a, b):
    return jnp.matmul(a, b)

# 数据分片到虚拟 NPU
A = jnp.ones((4, 256, 512))  # 4 个虚拟 NPU
B = jnp.ones((4, 512, 256))

# 并行计算
result = parallel_matmul(A, B)
```

### 步骤 3: 添加 NPU 特性（可选）

```python
# NPU 量化
def quantize_int8(x):
    scale = jnp.max(jnp.abs(x)) / 127.0
    return jnp.round(x / scale).astype(jnp.int8)

# 组合：原生 pmap + NPU 量化
@pmap
def npu_parallel_matmul(a, b):
    a_q = quantize_int8(a)  # NPU 特性
    b_q = quantize_int8(b)  # NPU 特性
    return jnp.matmul(a_q, b_q)  # 原生 JAX
```

---

## 📋 支持的原生 JAX API（全部！）

### ✅ 分片 API

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(devices, ('x',))
sharding = NamedSharding(mesh, P('x', None))
A_sharded = jax.device_put(A, sharding)  # ← 直接用！
```

### ✅ pmap

```python
from jax import pmap

@pmap  # ← 直接用！
def my_func(x):
    return x * 2
```

### ✅ shard_map

```python
from jax.experimental.shard_map import shard_map

@shard_map(mesh=mesh, in_specs=..., out_specs=...)  # ← 直接用！
def my_func(x):
    return x * 2
```

### ✅ with_sharding_constraint

```python
from jax.lax import with_sharding_constraint

@jit
def my_func(x):
    x = with_sharding_constraint(x, P('x', None))  # ← 直接用！
    return x * 2
```

### ✅ 集合通信

```python
from jax import lax

@pmap
def reduce_mean(x):
    return lax.pmean(x, axis_name='batch')  # ← 直接用！
```

---

## 🎓 完整示例

### 示例：在虚拟 NPU 上训练模型

```python
import jax
import jax.numpy as jnp
from jax import random, pmap, grad

# 1. 虚拟 NPU = JAX 设备
devices = jax.devices()[:4]  # 4 个虚拟 NPU

# 2. 定义模型（原生 JAX）
def model(params, x):
    w, b = params
    return jnp.matmul(x, w) + b

def loss_fn(params, x, y):
    return jnp.mean((model(params, x) - y) ** 2)

# 3. 训练步骤（原生 pmap）
@pmap  # ← 直接用原生 API！
def train_step(params, x, y):
    grads = grad(loss_fn)(params, x, y)
    grads = jax.lax.pmean(grads, 'batch')  # ← 原生 API！
    new_params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    return new_params

# 4. 初始化和训练
key = random.PRNGKey(0)
params = [
    jnp.stack([random.normal(key, (512, 256))] * 4),
    jnp.stack([jnp.zeros(256)] * 4)
]

x = random.normal(key, (4, 32, 512))
y = random.normal(key, (4, 32, 256))

# 5. 训练循环
for epoch in range(100):
    params = train_step(params, x, y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}")

# 完全原生 JAX 代码！
```

---

## 📊 功能完整性

| JAX 功能 | 虚拟 NPU 支持 | 说明 |
|---------|-------------|------|
| `jax.pmap` | ✅ 原生 | 数据并行 |
| `jax.device_put` | ✅ 原生 | 手动分片 |
| `NamedSharding` | ✅ 原生 | 命名分片 |
| `shard_map` | ✅ 原生 | 手动映射 |
| `with_sharding_constraint` | ✅ 原生 | 分片约束 |
| `jit` | ✅ 原生 | JIT 编译 |
| `grad` | ✅ 原生 | 自动微分 |
| `vmap` | ✅ 原生 | 向量化 |
| `lax.pmean` | ✅ 原生 | 跨设备平均 |
| `lax.psum` | ✅ 原生 | 跨设备求和 |
| `lax.all_gather` | ✅ 原生 | 收集通信 |

**结论：所有 JAX 功能都是原生支持！**

---

## 🔧 实际应用

### 应用 1: 开发分布式训练代码

```python
# 在 4 个虚拟 NPU 上开发
devices = jax.devices()[:4]

# 直接使用 JAX 原生 API
@pmap
def train_step(params, batch):
    # 你的训练逻辑
    pass

# 无需修改，直接部署到真实 NPU
```

### 应用 2: 测试大规模并行

```python
# 模拟 8 虚拟 NPU
devices = jax.devices()[:8]

# 使用原生 Mesh API
from jax.sharding import Mesh

mesh = Mesh(np.array(devices).reshape(2, 4), ('data', 'model'))

# 数据并行 × 模型并行
# 完全原生 JAX 代码！
```

### 应用 3: CI/CD 测试

```python
# test_distributed.py
import pytest
import jax
from jax import pmap

def test_multi_device_training():
    # 虚拟 NPU = JAX 设备
    assert len(jax.devices()) >= 4
    
    # 使用原生 pmap
    @pmap
    def compute(x):
        return x * 2
    
    x = jnp.ones((4, 100))
    result = compute(x)
    assert result.shape == (4, 100)
```

---

## 🎉 总结

### ✅ 你可以：

1. **直接使用所有 JAX 原生 API**
   - 不需要任何包装
   - 不需要任何修改
   - 不需要学习新 API

2. **完全兼容 JAX 生态**
   - 所有教程、文档适用
   - 所有第三方库兼容
   - 所有优化技巧有效

3. **添加 NPU 特性是可选的**
   - 量化、专用算子等是额外功能
   - 不影响 JAX 原生功能

### 💡 关键公式

```
虚拟 NPU Backend = JAX 设备映射 + NPU 特性（可选）
```

因此：
```
原生 JAX API → 虚拟 NPU → 直接工作 ✅
```

---

## 📞 快速参考

| 需求 | 使用 | 文件 |
|------|-----|------|
| 只要设备管理 | `simulate_npu_cluster.py` | 不需要 JAX |
| 要计算能力 | `npu_simulator_with_compute.py` | 需要 JAX |
| 要分布式 | `npu_use_native_jax_api.py` | 需要 JAX |
| 看原理 | `虚拟Backend实现指南.md` | 文档 |
| 看 API | `直接使用原生JAX_API指南.md` | 文档 |

---

**立即开始：**
```bash
python3 examples/npu_use_native_jax_api.py
```

🎉 **享受原生 JAX API 的全部能力！**
