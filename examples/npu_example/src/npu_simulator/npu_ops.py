"""
NPU核心操作实现

这个模块提供了NPU的基本操作，包括矩阵乘法和设备管理功能。
"""

import functools
import numpy as np
from typing import Optional, Tuple, Any
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

# 导入C++扩展模块
try:
    from npu_simulator import _npu_simulator
    _npu_available = True
except ImportError as e:
    _npu_available = False
    _import_error = e

def _check_npu_availability():
    """检查NPU模拟器是否可用"""
    if not _npu_available:
        raise RuntimeError(f"NPU Simulator not available: {_import_error}")

# 注册FFI目标
def _register_npu_ffi_targets():
    """注册NPU FFI目标到JAX"""
    _check_npu_availability()
    
    registrations = _npu_simulator.registrations()
    for name, target in registrations.items():
        jax.ffi.register_ffi_target(name, target, platform="cpu")  # 在CPU上模拟NPU

# 初始化时自动注册
if _npu_available:
    _register_npu_ffi_targets()

class NPUArray:
    """
    NPU数组包装类，提供NPU特定的数组操作
    """
    def __init__(self, data: jax.Array, device_id: int = 0):
        self.data = data
        self.device_id = device_id
        
    @property
    def shape(self):
        return self.data.shape
        
    @property 
    def dtype(self):
        return self.data.dtype
        
    def __array__(self):
        return np.array(self.data)
        
    def __repr__(self):
        return f"NPUArray(shape={self.shape}, dtype={self.dtype}, device_id={self.device_id})"

def initialize_npu(num_devices: int = 4):
    """
    初始化NPU运行时
    
    Args:
        num_devices: NPU设备数量
    """
    _check_npu_availability()
    _npu_simulator.initialize_npu_runtime(num_devices)
    print(f"NPU Runtime initialized with {num_devices} devices")

def get_npu_device_count() -> int:
    """获取NPU设备数量"""
    _check_npu_availability()
    return _npu_simulator.get_npu_device_count()

def set_npu_device(device_id: int):
    """设置当前NPU设备"""
    _check_npu_availability()
    _npu_simulator.set_current_npu_device(device_id)

def get_current_npu_device() -> int:
    """获取当前NPU设备ID"""
    _check_npu_availability()
    return _npu_simulator.get_current_npu_device()

def npu_device_put(array: jax.Array, device_id: Optional[int] = None) -> NPUArray:
    """
    将数组放到NPU设备上
    
    Args:
        array: 输入数组
        device_id: 目标设备ID，None表示使用当前设备
        
    Returns:
        NPUArray: NPU数组对象
    """
    if device_id is None:
        device_id = get_current_npu_device()
    
    # 在实际实现中，这里会执行设备间的数据传输
    # 目前只是创建一个NPUArray包装器
    return NPUArray(array, device_id)

def npu_device_get(npu_array: NPUArray) -> jax.Array:
    """
    从NPU设备获取数组数据
    
    Args:
        npu_array: NPU数组
        
    Returns:
        jax.Array: JAX数组
    """
    return npu_array.data

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
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices")
        
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Matrix dimension mismatch: {a.shape[1]} != {b.shape[0]}")
    
    output_shape = (a.shape[0], b.shape[1])
    output_dtype = jnp.result_type(a.dtype, b.dtype)
    
    if output_dtype != jnp.float32:
        raise ValueError("NPU currently only supports float32")
    
    result = jax.ffi.ffi_call(
        "npu_matmul",
        ShapeDtypeStruct(output_shape, output_dtype),
        vmap_method="broadcast_all",
    )(a, b, device_id=device_id)
    
    return result

def npu_optimized_matmul(a: jax.Array, b: jax.Array, device_id: int = 0) -> jax.Array:
    """
    NPU优化版本矩阵乘法（使用NPU特定优化）
    
    Args:
        a: 左矩阵 (M, K)
        b: 右矩阵 (K, N)
        device_id: NPU设备ID
        
    Returns:
        result: 结果矩阵 (M, N)
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices")
        
    output_shape = (a.shape[0], b.shape[1]) 
    output_dtype = jnp.result_type(a.dtype, b.dtype)
    
    if output_dtype != jnp.float32:
        raise ValueError("NPU currently only supports float32")
    
    result = jax.ffi.ffi_call(
        "npu_optimized_matmul",
        ShapeDtypeStruct(output_shape, output_dtype),
        vmap_method="broadcast_all",
    )(a, b, device_id=device_id)
    
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
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Both inputs must be 3D tensors for batch matmul")
        
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Batch size mismatch: {a.shape[0]} != {b.shape[0]}")
        
    if a.shape[2] != b.shape[1]:
        raise ValueError(f"Matrix dimension mismatch: {a.shape[2]} != {b.shape[1]}")
    
    output_shape = (a.shape[0], a.shape[1], b.shape[2])
    output_dtype = jnp.result_type(a.dtype, b.dtype)
    
    if output_dtype != jnp.float32:
        raise ValueError("NPU currently only supports float32")
    
    result = jax.ffi.ffi_call(
        "npu_batch_matmul", 
        ShapeDtypeStruct(output_shape, output_dtype),
        vmap_method="broadcast_all",
    )(a, b, device_id=device_id)
    
    return result

# 为npu_matmul定义梯度计算
def npu_matmul_fwd(a: jax.Array, b: jax.Array, device_id: int = 0) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """NPU矩阵乘法前向传播（保存梯度计算所需的中间值）"""
    result = npu_matmul(a, b, device_id)
    return result, (a, b)

def npu_matmul_bwd(device_id: int, res: Tuple[jax.Array, jax.Array], g: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """NPU矩阵乘法反向传播（计算梯度）"""
    a, b = res
    # 梯度计算: ∂L/∂A = (∂L/∂C) @ B^T, ∂L/∂B = A^T @ (∂L/∂C)
    grad_a = npu_matmul(g, jnp.transpose(b), device_id)
    grad_b = npu_matmul(jnp.transpose(a), g, device_id)
    return grad_a, grad_b

# 注册梯度计算
npu_matmul.defvjp(npu_matmul_fwd, npu_matmul_bwd)