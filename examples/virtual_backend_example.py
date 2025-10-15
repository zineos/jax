#!/usr/bin/env python3
"""
示例：如何在 JAX 中添加虚拟 Backend

这个示例展示了如何创建一个自定义的虚拟 backend，可用于：
1. 测试和调试
2. 模拟特定硬件环境
3. 在没有实际硬件的情况下进行开发

使用方法:
    import jax
    from examples.virtual_backend_example import register_virtual_backend
    
    # 注册虚拟 backend
    register_virtual_backend()
    
    # 使用虚拟 backend
    backend = jax.extend.backend.get_backend("virtual")
    print(f"Backend platform: {backend.platform}")
    print(f"Device count: {backend.device_count()}")
"""

from typing import Any, Optional
from jax._src import xla_bridge
from jax._src.lib import xla_client


class VirtualDevice:
    """虚拟设备类，模拟 JAX 设备接口"""
    
    def __init__(self, device_id: int, platform: str, process_index: int = 0):
        self.id = device_id
        self.platform = platform
        self.device_kind = "virtual"
        self.process_index = process_index
        self.host_id = process_index
        
    def __repr__(self):
        return f"VirtualDevice(id={self.id}, platform={self.platform})"
    
    def __str__(self):
        return f"{self.platform}:{self.id}"


class VirtualBackend:
    """虚拟 Backend 类，实现 JAX Backend 接口
    
    这个类模拟了一个完整的 JAX backend，包含了所有必要的方法。
    实际的 backend 需要继承自 xla_client.Client，但对于简单的虚拟 backend，
    我们可以只实现需要的接口。
    """
    
    def __init__(self, 
                 platform: str = "virtual",
                 device_count: int = 4,
                 process_index: int = 0):
        """初始化虚拟 backend
        
        Args:
            platform: backend 平台名称
            device_count: 虚拟设备数量
            process_index: 进程索引（用于多进程模拟）
        """
        self.platform = platform
        self._device_count = device_count
        self._process_index = process_index
        self.platform_version = "VirtualBackend v1.0"
        
        # 创建虚拟设备
        self._devices = [
            VirtualDevice(i, platform, process_index) 
            for i in range(device_count)
        ]
    
    def device_count(self) -> int:
        """返回设备总数"""
        return self._device_count
    
    def local_device_count(self) -> int:
        """返回本地设备数量"""
        return self._device_count
    
    def process_index(self) -> int:
        """返回进程索引"""
        return self._process_index
    
    def devices(self) -> list:
        """返回所有设备列表"""
        return self._devices
    
    def local_devices(self) -> list:
        """返回本地设备列表"""
        return self._devices
    
    def _get_all_devices(self) -> list:
        """返回所有设备（包括非本地设备）"""
        return self.devices()
    
    def __repr__(self):
        return (f"VirtualBackend(platform='{self.platform}', "
                f"device_count={self._device_count}, "
                f"process_index={self._process_index})")


def create_virtual_backend(
    platform: str = "virtual",
    device_count: int = 4,
    process_index: int = 0
) -> VirtualBackend:
    """创建虚拟 backend 的工厂函数
    
    这是一个工厂函数，用于创建虚拟 backend 实例。
    JAX 的 backend 注册系统需要一个无参数的工厂函数。
    
    Args:
        platform: backend 平台名称
        device_count: 虚拟设备数量
        process_index: 进程索引
        
    Returns:
        VirtualBackend 实例
    """
    return VirtualBackend(
        platform=platform,
        device_count=device_count,
        process_index=process_index
    )


def register_virtual_backend(
    platform: str = "virtual",
    device_count: int = 4,
    priority: int = 0,
    fail_quietly: bool = True,
    experimental: bool = False
) -> None:
    """注册虚拟 backend 到 JAX
    
    将虚拟 backend 注册到 JAX 的 backend 系统中。
    注册后，可以通过 jax.extend.backend.get_backend(platform) 获取该 backend。
    
    Args:
        platform: backend 平台名称
        device_count: 虚拟设备数量
        priority: backend 优先级（数值越大优先级越高）
        fail_quietly: 如果初始化失败，是否静默失败
        experimental: 是否标记为实验性 backend
        
    Example:
        >>> from jax.extend import backend
        >>> register_virtual_backend("my_virtual_backend", device_count=8, priority=100)
        >>> backend_instance = backend.get_backend("my_virtual_backend")
        >>> print(backend_instance.device_count())
        8
    """
    # 创建一个返回虚拟 backend 的工厂函数
    def factory():
        return create_virtual_backend(
            platform=platform,
            device_count=device_count,
            process_index=0
        )
    
    # 注册 backend 工厂函数
    xla_bridge.register_backend_factory(
        name=platform,
        factory=factory,
        priority=priority,
        fail_quietly=fail_quietly,
        experimental=experimental
    )
    
    print(f"✓ Virtual backend '{platform}' registered successfully!")
    print(f"  - Device count: {device_count}")
    print(f"  - Priority: {priority}")
    print(f"  - Experimental: {experimental}")


# ============================================================================
# 使用示例
# ============================================================================

def demo():
    """演示如何使用虚拟 backend"""
    import jax
    from jax.extend import backend
    
    print("=" * 70)
    print("JAX 虚拟 Backend 示例")
    print("=" * 70)
    print()
    
    # 1. 注册一个简单的虚拟 backend
    print("1. 注册虚拟 backend 'virtual'...")
    register_virtual_backend(
        platform="virtual",
        device_count=4,
        priority=0,  # 低优先级，不会成为默认 backend
        experimental=True
    )
    print()
    
    # 2. 注册一个模拟 GPU 的虚拟 backend
    print("2. 注册虚拟 backend 'virtual_gpu'...")
    register_virtual_backend(
        platform="virtual_gpu",
        device_count=8,
        priority=50,
        experimental=True
    )
    print()
    
    # 3. 获取并使用虚拟 backend
    print("3. 获取虚拟 backend...")
    try:
        virtual_backend = backend.get_backend("virtual")
        print(f"   Backend: {virtual_backend}")
        print(f"   Platform: {virtual_backend.platform}")
        print(f"   Device count: {virtual_backend.device_count()}")
        print(f"   Devices: {virtual_backend.devices()}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # 4. 查看所有可用的 backends
    print("4. 所有注册的 backends:")
    try:
        all_backends = backend.backends()
        for name, backend_instance in all_backends.items():
            print(f"   - {name}: {backend_instance.platform}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # 5. 获取默认 backend
    print("5. 默认 backend:")
    try:
        default_backend = backend.get_backend()
        print(f"   Platform: {default_backend.platform}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    demo()
