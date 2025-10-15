# JAX 虚拟 Backend 实现指南

## 概述

JAX 支持添加自定义虚拟 backend，这对以下场景非常有用：

1. **测试和调试**：在没有实际硬件的情况下测试分布式代码
2. **模拟环境**：模拟特定硬件配置（如多 GPU、TPU 集群）
3. **教育和演示**：展示 JAX 的 backend 系统工作原理
4. **开发新 backend**：在实现真实 backend 之前快速原型设计

## 核心概念

### Backend 注册系统

JAX 使用工厂模式来管理 backend：

```python
from jax._src import xla_bridge

def my_backend_factory():
    return MyBackend()

xla_bridge.register_backend_factory(
    name="my_backend",
    factory=my_backend_factory,
    priority=100,  # 数值越大优先级越高
    fail_quietly=False,  # 初始化失败时是否抛出异常
    experimental=False   # 是否标记为实验性
)
```

### Backend 接口

一个最小的虚拟 backend 需要实现以下接口：

```python
class MyBackend:
    # 必需的属性
    platform: str  # backend 平台名称（如 "cpu", "gpu", "tpu"）
    platform_version: str  # 平台版本信息
    
    # 必需的方法
    def device_count(self) -> int:
        """返回总设备数"""
        
    def local_device_count(self) -> int:
        """返回本地设备数"""
        
    def process_index(self) -> int:
        """返回当前进程索引"""
        
    def devices(self) -> list:
        """返回所有设备列表"""
        
    def local_devices(self) -> list:
        """返回本地设备列表"""
        
    def _get_all_devices(self) -> list:
        """返回所有设备（包括远程）"""
```

## 使用示例

### 基本使用

```python
import jax
from jax.extend import backend
from examples.virtual_backend_example import register_virtual_backend

# 注册虚拟 backend
register_virtual_backend(
    platform="my_virtual_device",
    device_count=8,
    priority=0
)

# 获取 backend
virtual_backend = backend.get_backend("my_virtual_device")

# 查看设备信息
print(f"Platform: {virtual_backend.platform}")
print(f"Device count: {virtual_backend.device_count()}")
print(f"Devices: {virtual_backend.devices()}")
```

### 模拟多进程环境

```python
# 模拟 4 个进程，每个进程 2 个设备
for process_id in range(4):
    register_virtual_backend(
        platform=f"virtual_process_{process_id}",
        device_count=2,
        process_index=process_id
    )
```

### 通过环境变量指定 Backend

```python
import os

# 设置 JAX 使用虚拟 backend
os.environ['JAX_PLATFORMS'] = 'my_virtual_device'

import jax
# JAX 将自动使用虚拟 backend
```

## 实际应用场景

### 1. 测试分布式代码

```python
# 测试 pmap 代码时，无需实际的多 GPU 硬件
from examples.virtual_backend_example import register_virtual_backend

register_virtual_backend(platform="test_gpu", device_count=8)

# 现在可以测试 8-way 并行代码
```

### 2. 开发 Backend 插件

如果你正在为新硬件开发 PJRT 插件，可以先创建虚拟 backend 原型：

```python
class MyHardwareBackend:
    """模拟你的硬件 backend"""
    platform = "my_hardware"
    
    def __init__(self):
        # 初始化硬件特定的配置
        self.custom_feature = "supported"
    
    # 实现其他必需接口...
```

### 3. CI/CD 测试

在 CI 环境中，可能没有 GPU/TPU，但仍需要测试代码：

```bash
# CI 脚本
export JAX_PLATFORMS=virtual
python run_tests.py  # 使用虚拟 backend 运行测试
```

## 高级主题

### 完整的 PJRT Backend

对于生产级别的 backend，你需要：

1. 实现完整的 PJRT C API
2. 提供编译和执行功能
3. 支持内存管理和设备间通信

参考：
- `jax/_src/xla_bridge.py` - Backend 注册系统
- `jaxlib/xla_client.py` - XLA Client 接口
- PJRT 插件示例：https://github.com/openxla/xla/tree/main/xla/pjrt

### Backend 优先级

Backend 的初始化顺序由优先级决定：

- `cpu`: 0 (基准)
- `tpu`: 300 (高优先级)
- 插件: 400 (默认)

JAX 会选择优先级最高且成功初始化的 backend 作为默认 backend。

### 实验性 Backend

标记为 `experimental=True` 的 backend 会在使用时显示警告：

```
WARNING: Platform 'my_backend' is experimental and not all 
JAX functionality may be correctly supported!
```

## 代码结构

```
examples/
├── virtual_backend_example.py     # 虚拟 backend 实现
├── virtual_backend_README.md      # 本文档
└── test_virtual_backend.py        # 测试用例
```

## 参考资料

- JAX Backend 系统：https://jax.readthedocs.io/en/latest/jax.extend.backend.html
- XLA Bridge 源码：`jax/_src/xla_bridge.py`
- Backend 测试：`tests/xla_bridge_test.py`
- PJRT 文档：https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md

## 常见问题

### Q: 虚拟 backend 可以执行实际计算吗？

A: 本示例中的虚拟 backend 只是一个接口模拟，不能执行实际计算。要执行计算，需要实现完整的编译和运行时功能。

### Q: 如何卸载已注册的 backend？

A: JAX 提供了 `clear_backends()` 函数来清理所有 backend：

```python
from jax.extend import backend
backend.clear_backends()
```

### Q: 可以覆盖现有的 backend（如 CPU）吗？

A: 可以，但不推荐。如果注册同名 backend，会抛出 `RuntimeError`。需要先清理现有 backend。

### Q: 虚拟 backend 的性能如何？

A: 虚拟 backend 本身没有执行引擎，主要用于接口测试，不涉及实际性能。

## 贡献

欢迎提交改进建议和 bug 报告！

## 许可

Apache 2.0 许可证
