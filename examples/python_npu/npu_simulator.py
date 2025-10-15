"""
纯Python NPU模拟器实现

使用JAX的pure_callback和custom_vjp机制实现自定义NPU操作，
无需C++扩展，完全基于Python。
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
from jax import pure_callback, ShapeDtypeStruct

class NPUDevice:
    """NPU设备模拟器"""
    
    def __init__(self, device_id: int, memory_gb: float = 8.0, compute_units: int = 128):
        self.device_id = device_id
        self.memory_gb = memory_gb
        self.compute_units = compute_units
        self.allocated_memory = 0
        self.is_busy = False
        self.peak_gflops = 100.0 + device_id * 10  # 不同设备略有差异
        self._lock = threading.Lock()
        
        print(f"✅ NPU-{device_id} initialized: {memory_gb}GB memory, {compute_units} compute units, {self.peak_gflops:.1f} GFLOPS")
    
    def allocate_memory(self, size_bytes: int) -> bool:
        """分配设备内存"""
        size_gb = size_bytes / (1024**3)
        with self._lock:
            if self.allocated_memory + size_gb > self.memory_gb:
                return False
            self.allocated_memory += size_gb
            return True
    
    def free_memory(self, size_bytes: int):
        """释放设备内存"""
        size_gb = size_bytes / (1024**3)
        with self._lock:
            self.allocated_memory = max(0, self.allocated_memory - size_gb)
    
    def set_busy(self, busy: bool):
        """设置设备忙碌状态"""
        with self._lock:
            self.is_busy = busy
    
    def get_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        with self._lock:
            return {
                "device_id": self.device_id,
                "memory_used_gb": self.allocated_memory,
                "memory_total_gb": self.memory_gb,
                "memory_utilization": self.allocated_memory / self.memory_gb,
                "is_busy": self.is_busy,
                "compute_units": self.compute_units,
                "peak_gflops": self.peak_gflops
            }

class NPURuntime:
    """NPU运行时管理器"""
    
    def __init__(self):
        self.devices: List[NPUDevice] = []
        self.current_device = 0
        self.initialized = False
        self.executor = None
        
    def initialize(self, num_devices: int = 4, memory_per_device: float = 8.0):
        """初始化NPU运行时"""
        if self.initialized:
            print("⚠️  NPU Runtime already initialized")
            return
            
        print(f"🚀 Initializing NPU Runtime with {num_devices} devices...")
        
        self.devices = [
            NPUDevice(i, memory_per_device) 
            for i in range(num_devices)
        ]
        
        # 创建线程池用于并行计算
        self.executor = ThreadPoolExecutor(max_workers=num_devices)
        self.initialized = True
        
        print(f"✅ NPU Runtime initialized successfully!")
        self._print_device_info()
    
    def _print_device_info(self):
        """打印设备信息"""
        print("\n📊 NPU Device Information:")
        print("-" * 60)
        for device in self.devices:
            status = device.get_status()
            print(f"NPU-{status['device_id']}: {status['memory_total_gb']:.1f}GB, "
                  f"{status['compute_units']} CUs, {status['peak_gflops']:.1f} GFLOPS")
        print("-" * 60)
    
    def get_device_count(self) -> int:
        """获取设备数量"""
        return len(self.devices)
    
    def get_device(self, device_id: int) -> NPUDevice:
        """获取指定设备"""
        if not self.initialized:
            raise RuntimeError("NPU Runtime not initialized")
        if device_id < 0 or device_id >= len(self.devices):
            raise ValueError(f"Invalid device_id {device_id}")
        return self.devices[device_id]
    
    def set_current_device(self, device_id: int):
        """设置当前设备"""
        if device_id < 0 or device_id >= len(self.devices):
            raise ValueError(f"Invalid device_id {device_id}")
        self.current_device = device_id
    
    def get_current_device(self) -> int:
        """获取当前设备ID"""
        return self.current_device
    
    def shutdown(self):
        """关闭运行时"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.initialized = False
        print("🔄 NPU Runtime shutdown")

# 全局NPU运行时实例
_npu_runtime = NPURuntime()

# NPU核心计算函数
def _npu_matmul_impl(a: np.ndarray, b: np.ndarray, device_id: int = 0) -> np.ndarray:
    """NPU矩阵乘法的实际实现"""
    device = _npu_runtime.get_device(device_id)
    
    # 计算FLOPS和预估执行时间
    flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]
    compute_time = flops / (device.peak_gflops * 1e9)  # 秒
    
    print(f"🔥 NPU-{device_id} executing MatMul: ({a.shape[0]}×{a.shape[1]}) × ({b.shape[0]}×{b.shape[1]}) "
          f"= {flops/1e9:.2f} GFLOPS, estimated {compute_time*1000:.2f}ms")
    
    # 模拟NPU计算延迟
    device.set_busy(True)
    time.sleep(compute_time * 0.1)  # 缩放因子，避免等待太久
    
    # 执行实际计算 - 这里可以插入NPU特定的优化算法
    result = _npu_optimized_matmul(a, b, device_id)
    
    device.set_busy(False)
    return result

def _npu_optimized_matmul(a: np.ndarray, b: np.ndarray, device_id: int) -> np.ndarray:
    """NPU优化的矩阵乘法算法"""
    # 模拟NPU的块矩阵乘法优化
    BLOCK_SIZE = 64  # NPU优化的块大小
    
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"Matrix dimensions mismatch: {k} != {k2}"
    
    # 创建结果矩阵
    c = np.zeros((m, n), dtype=np.float32)
    
    # 分块计算（模拟NPU的并行计算单元）
    for i in range(0, m, BLOCK_SIZE):
        for j in range(0, n, BLOCK_SIZE):
            for k_idx in range(0, k, BLOCK_SIZE):
                # 定义当前块的边界
                i_end = min(i + BLOCK_SIZE, m)
                j_end = min(j + BLOCK_SIZE, n)
                k_end = min(k_idx + BLOCK_SIZE, k)
                
                # 执行块矩阵乘法
                a_block = a[i:i_end, k_idx:k_end]
                b_block = b[k_idx:k_end, j:j_end]
                c_block = c[i:i_end, j:j_end]
                
                # NPU优化：使用高效的BLAS实现
                c[i:i_end, j:j_end] = c_block + np.dot(a_block, b_block)
    
    return c

def _npu_batch_matmul_impl(a: np.ndarray, b: np.ndarray, device_id: int = 0) -> np.ndarray:
    """NPU批量矩阵乘法实现"""
    batch_size = a.shape[0]
    device = _npu_runtime.get_device(device_id)
    
    print(f"🔥 NPU-{device_id} executing BatchMatMul: batch_size={batch_size}")
    
    device.set_busy(True)
    
    # 并行处理批次中的每个矩阵乘法
    results = []
    for i in range(batch_size):
        result_i = _npu_optimized_matmul(a[i], b[i], device_id)
        results.append(result_i)
    
    device.set_busy(False)
    return np.stack(results, axis=0)

# JAX回调函数包装
def _npu_matmul_callback(a: np.ndarray, b: np.ndarray, device_id: int) -> np.ndarray:
    """NPU矩阵乘法回调函数"""
    return _npu_matmul_impl(a, b, device_id)

def _npu_batch_matmul_callback(a: np.ndarray, b: np.ndarray, device_id: int) -> np.ndarray:
    """NPU批量矩阵乘法回调函数"""
    return _npu_batch_matmul_impl(a, b, device_id)

# JAX操作定义
@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def npu_matmul(a: jax.Array, b: jax.Array, device_id: int = 0) -> jax.Array:
    """
    NPU矩阵乘法操作
    
    Args:
        a: 左矩阵 (M, K)
        b: 右矩阵 (K, N)
        device_id: NPU设备ID
        
    Returns:
        result: 结果矩阵 (M, N)
    """
    if not _npu_runtime.initialized:
        raise RuntimeError("NPU Runtime not initialized. Call initialize_npu() first.")
    
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices")
    
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrix dimension mismatch: {a.shape[1]} != {b.shape[0]}")
    
    # 定义输出形状和类型
    output_shape = (a.shape[0], b.shape[1])
    output_dtype = jnp.result_type(a.dtype, b.dtype)
    
    # 使用pure_callback调用NPU实现
    result = pure_callback(
        _npu_matmul_callback,
        ShapeDtypeStruct(output_shape, output_dtype),
        a, b, device_id,
        vectorized=False
    )
    
    return result

def npu_batch_matmul(a: jax.Array, b: jax.Array, device_id: int = 0) -> jax.Array:
    """
    NPU批量矩阵乘法
    
    Args:
        a: 左矩阵批次 (B, M, K)
        b: 右矩阵批次 (B, K, N)
        device_id: NPU设备ID
        
    Returns:
        result: 结果矩阵批次 (B, M, N)
    """
    if not _npu_runtime.initialized:
        raise RuntimeError("NPU Runtime not initialized")
    
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Both inputs must be 3D tensors")
    
    output_shape = (a.shape[0], a.shape[1], b.shape[2])
    output_dtype = jnp.result_type(a.dtype, b.dtype)
    
    result = pure_callback(
        _npu_batch_matmul_callback,
        ShapeDtypeStruct(output_shape, output_dtype),
        a, b, device_id,
        vectorized=False
    )
    
    return result

# 梯度定义
def npu_matmul_fwd(a: jax.Array, b: jax.Array, device_id: int = 0):
    """前向传播"""
    result = npu_matmul(a, b, device_id)
    return result, (a, b)

def npu_matmul_bwd(device_id: int, res, g):
    """反向传播"""
    a, b = res
    # ∂L/∂A = (∂L/∂C) @ B^T
    # ∂L/∂B = A^T @ (∂L/∂C)
    grad_a = npu_matmul(g, b.T, device_id)
    grad_b = npu_matmul(a.T, g, device_id)
    return grad_a, grad_b

# 注册梯度
npu_matmul.defvjp(npu_matmul_fwd, npu_matmul_bwd)

# 公共API
def initialize_npu(num_devices: int = 4, memory_per_device: float = 8.0):
    """初始化NPU运行时"""
    _npu_runtime.initialize(num_devices, memory_per_device)

def get_npu_device_count() -> int:
    """获取NPU设备数量"""
    return _npu_runtime.get_device_count()

def set_npu_device(device_id: int):
    """设置当前NPU设备"""
    _npu_runtime.set_current_device(device_id)

def get_current_npu_device() -> int:
    """获取当前NPU设备"""
    return _npu_runtime.get_current_device()

def get_npu_device_info(device_id: Optional[int] = None) -> Dict[str, Any]:
    """获取NPU设备信息"""
    if device_id is None:
        device_id = _npu_runtime.get_current_device()
    device = _npu_runtime.get_device(device_id)
    return device.get_status()

def shutdown_npu():
    """关闭NPU运行时"""
    _npu_runtime.shutdown()