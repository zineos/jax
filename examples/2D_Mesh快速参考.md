# 方案2：构建2D NPU Mesh 快速参考

## 🎯 核心代码（30秒上手）

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# 1. 获取虚拟NPU设备
devices = jax.devices()  # 这些就是虚拟NPU

# 2. 构建2×2 mesh（关键：reshape）
mesh = Mesh(
    devices=np.array(devices[:4]).reshape(2, 2),  # ← 关键！
    axis_names=('data', 'model')  # 第一维=数据并行，第二维=模型并行
)

# 3. 定义2D分片策略
sharding = NamedSharding(mesh, P('data', 'model'))

# 4. 分片数据
X = jnp.ones((256, 1024))
X_sharded = jax.device_put(X, sharding)

# 完成！X已经2D分片到4个虚拟NPU上了
```

---

## 📊 2D Mesh 可视化

### 2×2 Mesh（4个NPU）

```
设备布局:
       model维度 →
       0      1
    ┌─────┬─────┐
d 0 │NPU0 │NPU1 │
a   ├─────┼─────┤
t 1 │NPU2 │NPU3 │
a   └─────┴─────┘
↓
```

**配置：**
```python
mesh = Mesh(
    np.array(devices[:4]).reshape(2, 2),
    ('data', 'model')
)
# mesh.shape = {'data': 2, 'model': 2}
```

### 4×2 Mesh（8个NPU）

```
设备布局:
       model维度 →
       0      1
    ┌─────┬─────┐
  0 │NPU0 │NPU1 │
d   ├─────┼─────┤
a 1 │NPU2 │NPU3 │
t   ├─────┼─────┤
a 2 │NPU4 │NPU5 │
  3 │NPU6 │NPU7 │
↓   └─────┴─────┘
```

**配置：**
```python
mesh = Mesh(
    np.array(devices[:8]).reshape(4, 2),
    ('data', 'model')
)
# mesh.shape = {'data': 4, 'model': 2}
```

---

## 🔑 分片策略（PartitionSpec）

### 常用分片模式

| 数据形状 | PartitionSpec | 说明 | 每设备数据 |
|---------|--------------|------|-----------|
| `(256, 1024)` | `P(None, None)` | 不分片，复制到所有设备 | `(256, 1024)` |
| `(256, 1024)` | `P('data', None)` | 仅数据维度分片 | `(128, 1024)` |
| `(256, 1024)` | `P(None, 'model')` | 仅特征维度分片 | `(256, 512)` |
| `(256, 1024)` | `P('data', 'model')` | 2D分片 | `(128, 512)` |

### 矩阵乘法2D分片

```python
# C = A @ B
# A: (m, k), B: (k, n) → C: (m, n)

mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y'))

# 分片策略
sharding_A = NamedSharding(mesh, P('x', None))  # A按行分片
sharding_B = NamedSharding(mesh, P(None, 'y'))  # B按列分片
sharding_C = NamedSharding(mesh, P('x', 'y'))   # C 2D分片

# 执行
A_sharded = jax.device_put(A, sharding_A)
B_sharded = jax.device_put(B, sharding_B)

@jit
def matmul(a, b):
    return jnp.matmul(a, b)

C = matmul(A_sharded, B_sharded)  # C自动2D分片
```

**设备分布：**
```
A的分片:          B的分片:          C的分片:
  k                 k   n             n
┌────┐           ┌───┬───┐         ┌───┬───┐
│ A0 │ m/2       │B0 │B1 │ k       │C0 │C1 │ m/2
├────┤           └───┴───┘         ├───┼───┤
│ A1 │ m/2                         │C2 │C3 │ m/2
└────┘                             └───┴───┘

NPU0: A0 @ B0 → C0
NPU1: A0 @ B1 → C1
NPU2: A1 @ B0 → C2
NPU3: A1 @ B1 → C3
```

---

## 🚀 实际应用场景

### 场景1: 数据并行训练（batch维度分片）

```python
# 配置: 4×1 mesh (4路数据并行)
mesh = Mesh(np.array(devices[:4]).reshape(4, 1), ('data', 'model'))

# 数据
batch_size = 128
x = jnp.ones((batch_size, 512))  # (batch, features)
w = jnp.ones((512, 256))         # (input, output)

# 分片: 只分batch维度
x_sharding = NamedSharding(mesh, P('data', None))
w_sharding = NamedSharding(mesh, P(None, None))  # 权重复制

x_sharded = jax.device_put(x, x_sharding)
w_sharded = jax.device_put(w, w_sharding)

@jit
def forward(x, w):
    return jnp.matmul(x, w)

# 每个NPU处理32个样本
output = forward(x_sharded, w_sharded)
```

### 场景2: 模型并行（大模型分片）

```python
# 配置: 1×4 mesh (4路模型并行)
mesh = Mesh(np.array(devices[:4]).reshape(1, 4), ('data', 'model'))

# 大权重矩阵
w = jnp.ones((4096, 4096))

# 分片: 按列分片
w_sharding = NamedSharding(mesh, P(None, 'model'))
w_sharded = jax.device_put(w, w_sharding)

# 每个NPU只存储1024列
```

### 场景3: 混合并行（推荐）

```python
# 配置: 2×2 mesh (2路数据 × 2路模型)
mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('data', 'model'))

# 数据和权重
x = jnp.ones((64, 512))   # (batch, input)
w = jnp.ones((512, 2048)) # (input, output)

# 混合分片
x_sharding = NamedSharding(mesh, P('data', None))    # batch维分片
w_sharding = NamedSharding(mesh, P(None, 'model'))   # output维分片

x_sharded = jax.device_put(x, x_sharding)
w_sharded = jax.device_put(w, w_sharding)

@jit
def forward(x, w):
    return jnp.matmul(x, w)  # 自动处理2D分片

output = forward(x_sharded, w_sharded)
# 输出自动2D分片: P('data', 'model')
```

---

## 📋 常见Mesh配置

### 4个NPU

| 配置 | Reshape | 用途 |
|------|---------|------|
| `4×1` | `(4, 1)` | 纯数据并行 |
| `1×4` | `(1, 4)` | 纯模型并行 |
| `2×2` | `(2, 2)` | 平衡（推荐） |

### 8个NPU

| 配置 | Reshape | 用途 |
|------|---------|------|
| `8×1` | `(8, 1)` | 纯数据并行 |
| `1×8` | `(1, 8)` | 纯模型并行 |
| `4×2` | `(4, 2)` | 数据为主 |
| `2×4` | `(2, 4)` | 模型为主 |

### 16个NPU

| 配置 | Reshape | 用途 |
|------|---------|------|
| `16×1` | `(16, 1)` | 纯数据并行 |
| `4×4` | `(4, 4)` | 平衡（推荐） |
| `8×2` | `(8, 2)` | 数据为主 |
| `2×8` | `(2, 8)` | 模型为主 |

---

## 💡 组合NPU特性 + 2D Mesh

```python
import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# NPU模拟器（量化）
class NPUSimulator:
    @staticmethod
    def quantize_int8(x):
        scale = jnp.max(jnp.abs(x)) / 127.0
        quantized = jnp.round(x / scale).astype(jnp.int8)
        return quantized.astype(jnp.float32) * scale
    
    @staticmethod
    def npu_matmul(a, b):
        # NPU特性：INT8量化
        a_q = NPUSimulator.quantize_int8(a)
        b_q = NPUSimulator.quantize_int8(b)
        return jnp.matmul(a_q, b_q)

# 2D Mesh
devices = jax.devices()
mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('x', 'y'))

# 2D分片
A = jnp.ones((512, 256))
B = jnp.ones((256, 128))

A_sharded = jax.device_put(A, NamedSharding(mesh, P('x', None)))
B_sharded = jax.device_put(B, NamedSharding(mesh, P(None, 'y')))

# NPU计算 + 2D分片
@jit
def npu_matmul_2d(a, b):
    return NPUSimulator.npu_matmul(a, b)

result = npu_matmul_2d(A_sharded, B_sharded)

# ✅ NPU量化 + 2D分片 完美结合！
```

---

## 🔍 调试和可视化

### 查看分片信息

```python
# 查看mesh配置
print(f"Mesh形状: {mesh.shape}")
print(f"轴名称: {mesh.axis_names}")

# 查看数据分片
print(f"数据分片: {X_sharded.sharding}")
print(f"分片规范: {X_sharded.sharding.spec}")

# 查看每个设备的数据
for i, shard in enumerate(X_sharded.addressable_shards):
    print(f"NPU-{i}: 数据形状 = {shard.data.shape}")
```

### 可视化分片

```python
import jax

# 可视化工具
def visualize_sharding(array, name="Array"):
    print(f"\n{name}:")
    print(f"  形状: {array.shape}")
    print(f"  分片: {array.sharding}")
    print(f"  设备分布:")
    for i, shard in enumerate(array.addressable_shards):
        print(f"    设备{i}: {shard.data.shape} 在 {shard.device}")

# 使用
visualize_sharding(X_sharded, "X")
visualize_sharding(result, "Result")
```

---

## ⚡ 性能提示

### 1. 选择合适的Mesh形状

```python
# 规则：数据并行 × 模型并行 ≈ 总设备数

# 小模型，大batch → 数据并行为主
mesh = Mesh(devices.reshape(8, 1), ('data', 'model'))

# 大模型，小batch → 模型并行为主
mesh = Mesh(devices.reshape(1, 8), ('data', 'model'))

# 平衡 → 均衡分配
mesh = Mesh(devices.reshape(4, 2), ('data', 'model'))
```

### 2. 避免不必要的通信

```python
# ✅ 好：权重复制（小数据）
w_sharding = NamedSharding(mesh, P(None, None))

# ❌ 差：权重2D分片（增加通信）
w_sharding = NamedSharding(mesh, P('data', 'model'))  # 不推荐
```

### 3. 使用JIT编译

```python
# ✅ 使用JIT
@jit
def compute(x, w):
    return jnp.matmul(x, w)

# ❌ 不使用JIT（慢很多）
def compute(x, w):
    return jnp.matmul(x, w)
```

---

## 🎯 快速检查清单

构建2D NPU Mesh的步骤：

- [ ] 获取JAX设备 (`jax.devices()`)
- [ ] 决定Mesh形状 (如 `2×2`, `4×2`)
- [ ] Reshape设备数组 (`np.array(devices).reshape(...)`)
- [ ] 创建Mesh (`Mesh(devices, axis_names)`)
- [ ] 定义分片策略 (`NamedSharding(mesh, P(...))`)
- [ ] 分片数据 (`jax.device_put(data, sharding)`)
- [ ] 执行计算 (使用`@jit`)
- [ ] (可选) 添加NPU特性（量化等）

---

## 📞 完整示例

运行完整演示：
```bash
python3 examples/npu_2d_mesh_guide.py
```

查看所有方法：
- 方法1: 基础2×2 Mesh
- 方法2: 4×2 Mesh
- 方法3: 数据分片策略
- 方法4: 2D矩阵乘法
- 方法5: 混合并行
- 方法6: 高级配置
- 方法7: NPU特性 + 2D Mesh

---

## ✅ 总结

**核心要点：**

1. **构建2D Mesh** → `np.array(devices).reshape(rows, cols)`
2. **分片策略** → `P('data', 'model')` 
3. **分片数据** → `jax.device_put(data, sharding)`
4. **完全原生** → 使用JAX原生API，不需要包装

**一行代码构建2D Mesh：**
```python
mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('data', 'model'))
```

🎉 **就这么简单！**
