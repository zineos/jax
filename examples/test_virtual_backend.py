#!/usr/bin/env python3
"""
测试虚拟 Backend 实现

这个测试文件演示了如何测试虚拟 backend 的功能。
"""

import sys
import os

# 确保可以导入示例模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.virtual_backend_example import (
    VirtualBackend, 
    VirtualDevice,
    register_virtual_backend,
    create_virtual_backend
)


def test_virtual_device():
    """测试虚拟设备类"""
    print("测试 VirtualDevice...")
    
    device = VirtualDevice(device_id=0, platform="test_platform")
    
    assert device.id == 0
    assert device.platform == "test_platform"
    assert device.device_kind == "virtual"
    assert device.process_index == 0
    
    print(f"  ✓ Device: {device}")
    print(f"  ✓ ID: {device.id}")
    print(f"  ✓ Platform: {device.platform}")
    print()


def test_virtual_backend():
    """测试虚拟 backend 类"""
    print("测试 VirtualBackend...")
    
    backend = VirtualBackend(
        platform="test_backend",
        device_count=4,
        process_index=0
    )
    
    assert backend.platform == "test_backend"
    assert backend.device_count() == 4
    assert backend.local_device_count() == 4
    assert backend.process_index() == 0
    assert len(backend.devices()) == 4
    assert len(backend.local_devices()) == 4
    
    print(f"  ✓ Backend: {backend}")
    print(f"  ✓ Platform: {backend.platform}")
    print(f"  ✓ Device count: {backend.device_count()}")
    print(f"  ✓ Process index: {backend.process_index()}")
    print()


def test_create_virtual_backend():
    """测试 backend 工厂函数"""
    print("测试 create_virtual_backend...")
    
    backend = create_virtual_backend(
        platform="factory_test",
        device_count=8,
        process_index=1
    )
    
    assert isinstance(backend, VirtualBackend)
    assert backend.device_count() == 8
    assert backend.process_index() == 1
    
    print(f"  ✓ Created backend: {backend}")
    print()


def test_backend_registration():
    """测试 backend 注册"""
    print("测试 backend 注册...")
    
    try:
        # 注册一个测试 backend
        register_virtual_backend(
            platform="registration_test",
            device_count=2,
            priority=0,
            experimental=True
        )
        print("  ✓ Backend registered successfully")
        
        # 验证 backend 已注册
        from jax._src import xla_bridge
        assert "registration_test" in xla_bridge._backend_factories
        print("  ✓ Backend found in factory registry")
        
        # 获取注册信息
        registration = xla_bridge._backend_factories["registration_test"]
        print(f"  ✓ Priority: {registration.priority}")
        print(f"  ✓ Experimental: {registration.experimental}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise
    
    print()


def test_multiple_backends():
    """测试注册多个 backend"""
    print("测试多个 backend...")
    
    try:
        # 注册多个不同优先级的 backend
        register_virtual_backend("test_low_priority", device_count=2, priority=10)
        register_virtual_backend("test_high_priority", device_count=4, priority=100)
        register_virtual_backend("test_medium_priority", device_count=3, priority=50)
        
        print("  ✓ 成功注册 3 个 backend")
        
        # 验证所有 backend 都已注册
        from jax._src import xla_bridge
        
        backends_to_check = [
            "test_low_priority",
            "test_high_priority", 
            "test_medium_priority"
        ]
        
        for name in backends_to_check:
            if name in xla_bridge._backend_factories:
                reg = xla_bridge._backend_factories[name]
                print(f"  ✓ {name}: priority={reg.priority}")
            else:
                print(f"  ✗ {name} not found!")
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        raise
    
    print()


def test_backend_interface():
    """测试 backend 接口的完整性"""
    print("测试 Backend 接口完整性...")
    
    backend = VirtualBackend("interface_test", device_count=4)
    
    # 检查所有必需的方法和属性
    required_methods = [
        'device_count',
        'local_device_count',
        'process_index',
        'devices',
        'local_devices',
        '_get_all_devices'
    ]
    
    required_attributes = [
        'platform',
        'platform_version'
    ]
    
    print("  检查必需的方法:")
    for method_name in required_methods:
        if hasattr(backend, method_name):
            method = getattr(backend, method_name)
            if callable(method):
                print(f"    ✓ {method_name}()")
            else:
                print(f"    ✗ {method_name} 不是方法")
        else:
            print(f"    ✗ 缺少方法: {method_name}")
    
    print("  检查必需的属性:")
    for attr_name in required_attributes:
        if hasattr(backend, attr_name):
            value = getattr(backend, attr_name)
            print(f"    ✓ {attr_name} = {value}")
        else:
            print(f"    ✗ 缺少属性: {attr_name}")
    
    print()


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("运行虚拟 Backend 测试套件")
    print("=" * 70)
    print()
    
    tests = [
        ("VirtualDevice", test_virtual_device),
        ("VirtualBackend", test_virtual_backend),
        ("工厂函数", test_create_virtual_backend),
        ("Backend 注册", test_backend_registration),
        ("多个 Backend", test_multiple_backends),
        ("接口完整性", test_backend_interface),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ 测试 '{test_name}' 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
