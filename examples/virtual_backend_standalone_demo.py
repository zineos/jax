#!/usr/bin/env python3
"""
JAX 虚拟 Backend 概念演示（独立版本）

这个文件演示了虚拟 backend 的核心概念，不依赖完整的 JAX 安装。
它展示了如何设计和实现一个符合 JAX backend 接口的虚拟 backend。

作者: Cursor AI Assistant
日期: 2025-10-15
"""

from typing import List, Optional, Any
from dataclasses import dataclass


# ============================================================================
# 虚拟设备实现
# ============================================================================

@dataclass
class VirtualDevice:
    """虚拟设备类
    
    模拟 JAX 设备接口，代表一个可以执行计算的虚拟设备。
    在真实的 JAX backend 中，这对应于 xla_client.Device。
    
    属性:
        id: 设备 ID
        platform: 所属平台名称
        device_kind: 设备类型
        process_index: 所属进程索引
    """
    id: int
    platform: str
    device_kind: str = "virtual"
    process_index: int = 0
    
    @property
    def host_id(self) -> int:
        """进程 ID 的别名（JAX 的历史遗留接口）"""
        return self.process_index
    
    def __repr__(self) -> str:
        return f"VirtualDevice(id={self.id}, platform={self.platform}, process={self.process_index})"
    
    def __str__(self) -> str:
        return f"{self.platform}:{self.id}"


# ============================================================================
# 虚拟 Backend 实现
# ============================================================================

class VirtualBackend:
    """虚拟 Backend 类
    
    这个类实现了 JAX Backend 的核心接口。在真实的 JAX 中，
    backend 需要继承自 xla_client.Client 并实现编译、执行等功能。
    
    这个虚拟实现只提供接口模拟，用于：
    - 测试框架代码
    - 模拟多设备环境
    - 学习 JAX backend 系统
    
    参数:
        platform: backend 平台名称（如 "cpu", "gpu", "tpu"）
        device_count: 虚拟设备数量
        process_index: 当前进程索引（用于模拟多进程）
    """
    
    def __init__(
        self, 
        platform: str = "virtual",
        device_count: int = 4,
        process_index: int = 0
    ):
        self.platform = platform
        self._device_count = device_count
        self._process_index = process_index
        self.platform_version = f"VirtualBackend/{platform} v1.0"
        
        # 创建虚拟设备列表
        self._devices = [
            VirtualDevice(
                id=i, 
                platform=platform, 
                process_index=process_index
            )
            for i in range(device_count)
        ]
    
    # ------------------------------------------------------------------------
    # 核心接口方法（JAX Backend 必需）
    # ------------------------------------------------------------------------
    
    def device_count(self) -> int:
        """返回设备总数
        
        在多进程环境中，这应该返回所有进程的设备总数。
        对于单进程虚拟 backend，这等同于 local_device_count()。
        """
        return self._device_count
    
    def local_device_count(self) -> int:
        """返回本地（当前进程）的设备数量"""
        return self._device_count
    
    def process_index(self) -> int:
        """返回当前进程的索引
        
        在单机环境中通常返回 0。
        在多进程环境中，每个进程有唯一的索引。
        """
        return self._process_index
    
    def devices(self) -> List[VirtualDevice]:
        """返回所有设备的列表
        
        在多进程环境中，这应该返回所有进程的所有设备。
        对于虚拟 backend，我们简化为只返回本地设备。
        """
        return self._devices
    
    def local_devices(self) -> List[VirtualDevice]:
        """返回当前进程的本地设备列表"""
        return self._devices
    
    def _get_all_devices(self) -> List[VirtualDevice]:
        """返回所有设备（内部方法）
        
        这是一个内部方法，JAX 用它来获取包括远程设备在内的所有设备。
        """
        return self.devices()
    
    # ------------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (
            f"VirtualBackend("
            f"platform='{self.platform}', "
            f"device_count={self._device_count}, "
            f"process_index={self._process_index})"
        )
    
    def summary(self) -> str:
        """返回 backend 的摘要信息"""
        return (
            f"Virtual Backend Summary:\n"
            f"  Platform: {self.platform}\n"
            f"  Version: {self.platform_version}\n"
            f"  Total devices: {self.device_count()}\n"
            f"  Local devices: {self.local_device_count()}\n"
            f"  Process index: {self.process_index()}\n"
            f"  Devices: {[str(d) for d in self._devices]}"
        )


# ============================================================================
# Backend 注册系统（模拟）
# ============================================================================

class BackendRegistry:
    """Backend 注册表（模拟 JAX 的 backend 注册系统）
    
    这是一个简化版本的 backend 注册系统，演示了 JAX 如何管理多个 backend。
    在真实的 JAX 中，这个功能在 jax._src.xla_bridge 模块中实现。
    """
    
    def __init__(self):
        self._factories = {}
        self._backends = {}
        self._default_backend = None
    
    def register_backend_factory(
        self,
        name: str,
        factory,
        priority: int = 0,
        fail_quietly: bool = True
    ):
        """注册一个 backend 工厂函数
        
        参数:
            name: backend 名称
            factory: 返回 backend 实例的工厂函数
            priority: 优先级（数值越大优先级越高）
            fail_quietly: 初始化失败时是否静默
        """
        if name in self._backends:
            raise RuntimeError(f"Backend '{name}' already initialized")
        
        self._factories[name] = {
            'factory': factory,
            'priority': priority,
            'fail_quietly': fail_quietly
        }
        print(f"✓ Registered backend factory: {name} (priority={priority})")
    
    def get_backend(self, name: Optional[str] = None):
        """获取指定名称的 backend，或返回默认 backend
        
        参数:
            name: backend 名称，如果为 None 则返回默认 backend
            
        返回:
            Backend 实例
        """
        if name is None:
            if self._default_backend is None:
                self._initialize_all_backends()
            return self._default_backend
        
        if name not in self._backends:
            if name in self._factories:
                self._initialize_backend(name)
            else:
                raise RuntimeError(f"Unknown backend: {name}")
        
        return self._backends[name]
    
    def _initialize_backend(self, name: str):
        """初始化指定的 backend"""
        factory_info = self._factories[name]
        factory = factory_info['factory']
        
        try:
            backend = factory()
            self._backends[name] = backend
            print(f"✓ Initialized backend: {name}")
        except Exception as e:
            if not factory_info['fail_quietly']:
                raise RuntimeError(f"Failed to initialize backend '{name}': {e}")
            print(f"✗ Failed to initialize backend '{name}': {e}")
    
    def _initialize_all_backends(self):
        """初始化所有已注册的 backend"""
        # 按优先级排序
        sorted_factories = sorted(
            self._factories.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        
        for name, _ in sorted_factories:
            if name not in self._backends:
                self._initialize_backend(name)
        
        # 选择优先级最高的 backend 作为默认
        if sorted_factories and self._backends:
            default_name = sorted_factories[0][0]
            if default_name in self._backends:
                self._default_backend = self._backends[default_name]
    
    def list_backends(self) -> List[str]:
        """列出所有已注册的 backend"""
        return list(self._factories.keys())
    
    def list_initialized_backends(self) -> List[str]:
        """列出所有已初始化的 backend"""
        return list(self._backends.keys())


# ============================================================================
# 使用示例
# ============================================================================

def demo():
    """演示虚拟 backend 的使用"""
    print("=" * 70)
    print("JAX 虚拟 Backend 概念演示")
    print("=" * 70)
    print()
    
    # 创建注册表
    registry = BackendRegistry()
    
    # 示例 1: 注册单个虚拟 backend
    print("【示例 1】注册单个虚拟 backend")
    print("-" * 70)
    
    def create_virtual_cpu():
        return VirtualBackend(platform="virtual_cpu", device_count=4)
    
    registry.register_backend_factory(
        name="virtual_cpu",
        factory=create_virtual_cpu,
        priority=0
    )
    print()
    
    # 示例 2: 注册多个不同优先级的 backend
    print("【示例 2】注册多个虚拟 backend")
    print("-" * 70)
    
    registry.register_backend_factory(
        name="virtual_gpu",
        factory=lambda: VirtualBackend("virtual_gpu", device_count=8),
        priority=100
    )
    
    registry.register_backend_factory(
        name="virtual_tpu",
        factory=lambda: VirtualBackend("virtual_tpu", device_count=16),
        priority=200
    )
    print()
    
    # 示例 3: 获取和使用 backend
    print("【示例 3】获取和使用 backend")
    print("-" * 70)
    
    # 获取特定 backend
    cpu_backend = registry.get_backend("virtual_cpu")
    print(cpu_backend.summary())
    print()
    
    gpu_backend = registry.get_backend("virtual_gpu")
    print(gpu_backend.summary())
    print()
    
    # 示例 4: 获取默认 backend（优先级最高的）
    print("【示例 4】获取默认 backend")
    print("-" * 70)
    default_backend = registry.get_backend()  # 应该是 virtual_tpu（优先级最高）
    print(f"默认 backend: {default_backend.platform}")
    print(default_backend.summary())
    print()
    
    # 示例 5: 列出所有 backend
    print("【示例 5】所有已注册的 backend")
    print("-" * 70)
    print(f"已注册: {registry.list_backends()}")
    print(f"已初始化: {registry.list_initialized_backends()}")
    print()
    
    # 示例 6: 模拟多进程环境
    print("【示例 6】模拟多进程环境")
    print("-" * 70)
    
    # 创建 4 个进程，每个进程 2 个设备
    for process_id in range(4):
        backend = VirtualBackend(
            platform="distributed_gpu",
            device_count=2,
            process_index=process_id
        )
        print(f"进程 {process_id}:")
        print(f"  设备: {[str(d) for d in backend.local_devices()]}")
    print()
    
    # 示例 7: 访问设备信息
    print("【示例 7】访问设备详细信息")
    print("-" * 70)
    
    backend = registry.get_backend("virtual_gpu")
    for device in backend.devices():
        print(f"  {device}")
        print(f"    - ID: {device.id}")
        print(f"    - Platform: {device.platform}")
        print(f"    - Kind: {device.device_kind}")
        print(f"    - Process: {device.process_index}")
    print()
    
    print("=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
