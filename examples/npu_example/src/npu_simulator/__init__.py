"""
NPU Simulator for JAX

This module provides a complete NPU (Neural Processing Unit) simulation framework
that integrates with JAX through the Foreign Function Interface (FFI).

Features:
- Matrix multiplication operations optimized for NPU architecture
- Multi-device NPU simulation
- JAX integration with automatic differentiation support
- Parallel computation strategies
"""

from .npu_ops import (
    npu_matmul,
    npu_optimized_matmul, 
    npu_batch_matmul,
    initialize_npu,
    get_npu_device_count,
    set_npu_device,
    get_current_npu_device,
    NPUArray,
    npu_device_put,
    npu_device_get,
)

from .parallel_strategies import (
    NPUMesh,
    create_npu_mesh,
    shard_array_to_npu,
    distributed_npu_matmul,
)

from .performance_profiling import (
    NPUProfiler,
    profile_npu_operation,
)

__version__ = "0.1.0"
__all__ = [
    # Core operations
    "npu_matmul",
    "npu_optimized_matmul", 
    "npu_batch_matmul",
    
    # Device management
    "initialize_npu",
    "get_npu_device_count",
    "set_npu_device", 
    "get_current_npu_device",
    
    # Array operations
    "NPUArray",
    "npu_device_put",
    "npu_device_get",
    
    # Parallel strategies
    "NPUMesh",
    "create_npu_mesh",
    "shard_array_to_npu",
    "distributed_npu_matmul",
    
    # Profiling
    "NPUProfiler",
    "profile_npu_operation",
]