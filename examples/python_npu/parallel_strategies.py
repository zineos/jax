"""
NPU并行策略 - 纯Python实现

展示如何在Python NPU模拟器上实现各种并行计算策略
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from .npu_simulator import npu_matmul, get_npu_device_count, get_npu_device_info

class NPUParallelCompute:
    """NPU并行计算管理器"""
    
    def __init__(self):
        self.device_count = get_npu_device_count()
        
    def data_parallel_matmul(self, 
                           batch_a: jax.Array, 
                           batch_b: jax.Array,
                           devices: Optional[List[int]] = None) -> jax.Array:
        """
        数据并行矩阵乘法
        
        Args:
            batch_a: 批量矩阵A (batch_size, M, K)
            batch_b: 批量矩阵B (batch_size, K, N) 或 (K, N)
            devices: 使用的设备列表
            
        Returns:
            批量计算结果 (batch_size, M, N)
        """
        if devices is None:
            devices = list(range(self.device_count))
        
        batch_size = batch_a.shape[0]
        num_devices = len(devices)
        
        if batch_size % num_devices != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by number of devices {num_devices}")
        
        per_device_batch = batch_size // num_devices
        
        print(f"🔄 Data Parallel MatMul: {batch_size} batches across {num_devices} NPU devices")
        print(f"   Each device processes {per_device_batch} batches")
        
        def compute_on_device(device_id: int, start_idx: int, end_idx: int):
            """在指定设备上计算"""
            device_batch_a = batch_a[start_idx:end_idx]
            
            if batch_b.ndim == 3:  # 批量矩阵
                device_batch_b = batch_b[start_idx:end_idx]
                results = jax.vmap(lambda a, b: npu_matmul(a, b, device_id))(device_batch_a, device_batch_b)
            else:  # 共享矩阵B
                results = jax.vmap(lambda a: npu_matmul(a, batch_b, device_id))(device_batch_a)
            
            return results
        
        # 并行计算
        results = []
        for i, device_id in enumerate(devices):
            start_idx = i * per_device_batch
            end_idx = start_idx + per_device_batch
            
            result = compute_on_device(device_id, start_idx, end_idx)
            results.append(result)
        
        # 合并结果
        return jnp.concatenate(results, axis=0)
    
    def model_parallel_matmul(self,
                            a: jax.Array,
                            b: jax.Array,
                            split_dim: str = "output",
                            devices: Optional[List[int]] = None) -> jax.Array:
        """
        模型并行矩阵乘法
        
        Args:
            a: 输入矩阵 (M, K)
            b: 权重矩阵 (K, N)
            split_dim: 分割维度 ("input" 或 "output")
            devices: 使用的设备列表
            
        Returns:
            计算结果
        """
        if devices is None:
            devices = list(range(self.device_count))
        
        num_devices = len(devices)
        
        print(f"🔄 Model Parallel MatMul: split_dim={split_dim} across {num_devices} devices")
        
        if split_dim == "output":
            # 按输出维度分割权重矩阵
            if b.shape[1] % num_devices != 0:
                raise ValueError(f"Output dimension {b.shape[1]} must be divisible by {num_devices}")
            
            split_size = b.shape[1] // num_devices
            results = []
            
            for i, device_id in enumerate(devices):
                start_col = i * split_size
                end_col = start_col + split_size
                
                b_split = b[:, start_col:end_col]
                result_split = npu_matmul(a, b_split, device_id)
                results.append(result_split)
            
            # 按列拼接结果
            return jnp.concatenate(results, axis=1)
        
        elif split_dim == "input":
            # 按输入维度分割
            if a.shape[1] % num_devices != 0:
                raise ValueError(f"Input dimension {a.shape[1]} must be divisible by {num_devices}")
            
            split_size = a.shape[1] // num_devices
            results = []
            
            for i, device_id in enumerate(devices):
                start_col = i * split_size
                end_col = start_col + split_size
                
                a_split = a[:, start_col:end_col]
                b_split = b[start_col:end_col, :]
                
                result_split = npu_matmul(a_split, b_split, device_id)
                results.append(result_split)
            
            # 求和（All-Reduce）
            return sum(results)
        
        else:
            raise ValueError("split_dim must be 'input' or 'output'")
    
    def pipeline_parallel_matmul(self,
                                x: jax.Array,
                                weights: List[jax.Array],
                                devices: Optional[List[int]] = None) -> jax.Array:
        """
        流水线并行矩阵乘法（多层神经网络）
        
        Args:
            x: 输入数据 (batch_size, input_dim)
            weights: 各层权重列表 [(input_dim, hidden1), (hidden1, hidden2), ...]
            devices: 使用的设备列表
            
        Returns:
            最终输出
        """
        if devices is None:
            devices = list(range(min(len(weights), self.device_count)))
        
        num_layers = len(weights)
        if len(devices) != num_layers:
            raise ValueError(f"Number of devices {len(devices)} must match number of layers {num_layers}")
        
        print(f"🔄 Pipeline Parallel: {num_layers} layers across {len(devices)} devices")
        
        current_output = x
        for layer_idx, (weight, device_id) in enumerate(zip(weights, devices)):
            print(f"   Layer {layer_idx}: NPU-{device_id} computing {current_output.shape} × {weight.shape}")
            
            current_output = npu_matmul(current_output, weight, device_id)
            
            # 添加激活函数（除了最后一层）
            if layer_idx < num_layers - 1:
                current_output = jnp.maximum(current_output, 0)  # ReLU
        
        return current_output

def create_npu_distributed_array(array: jax.Array, 
                                sharding_strategy: str = "data_parallel",
                                num_devices: Optional[int] = None) -> jax.Array:
    """
    创建NPU分布式数组
    
    Args:
        array: 要分布的数组
        sharding_strategy: 分片策略 ("data_parallel", "model_parallel")
        num_devices: 设备数量
        
    Returns:
        分布式数组（在实际实现中会真正分布到不同设备）
    """
    if num_devices is None:
        num_devices = get_npu_device_count()
    
    print(f"📦 Creating distributed array: {array.shape}, strategy={sharding_strategy}")
    print(f"   Distributing across {num_devices} NPU devices")
    
    # 在实际实现中，这里会真正将数据分片到不同的NPU设备上
    # 目前只是模拟分片过程
    
    if sharding_strategy == "data_parallel":
        if array.shape[0] % num_devices != 0:
            raise ValueError(f"Batch dimension {array.shape[0]} must be divisible by {num_devices}")
        
        per_device = array.shape[0] // num_devices
        for i in range(num_devices):
            start_idx = i * per_device
            end_idx = start_idx + per_device
            print(f"   NPU-{i}: indices {start_idx}:{end_idx}")
    
    elif sharding_strategy == "model_parallel":
        if len(array.shape) < 2:
            raise ValueError("Model parallel requires at least 2D array")
        
        if array.shape[-1] % num_devices != 0:
            raise ValueError(f"Last dimension {array.shape[-1]} must be divisible by {num_devices}")
        
        per_device = array.shape[-1] // num_devices
        for i in range(num_devices):
            start_idx = i * per_device  
            end_idx = start_idx + per_device
            print(f"   NPU-{i}: feature dimensions {start_idx}:{end_idx}")
    
    return array

def benchmark_npu_parallel_strategies(matrix_size: int = 1024, 
                                    batch_size: int = 32,
                                    num_devices: Optional[int] = None) -> Dict[str, Any]:
    """
    基准测试NPU并行策略
    
    Args:
        matrix_size: 矩阵大小
        batch_size: 批次大小
        num_devices: 设备数量
        
    Returns:
        基准测试结果
    """
    if num_devices is None:
        num_devices = get_npu_device_count()
    
    print(f"\n🎯 NPU Parallel Strategies Benchmark")
    print(f"Matrix size: {matrix_size}×{matrix_size}, Batch size: {batch_size}, Devices: {num_devices}")
    print("=" * 60)
    
    # 生成测试数据
    key = jax.random.key(42)
    key1, key2, key3 = jax.random.split(key, 3)
    
    batch_a = jax.random.normal(key1, (batch_size, matrix_size, matrix_size))
    batch_b = jax.random.normal(key2, (batch_size, matrix_size, matrix_size)) 
    single_b = jax.random.normal(key3, (matrix_size, matrix_size))
    
    parallel_compute = NPUParallelCompute()
    results = {}
    
    # 1. 数据并行测试
    print("\n1️⃣  Data Parallel Test")
    import time
    start_time = time.time()
    data_parallel_result = parallel_compute.data_parallel_matmul(batch_a, single_b)
    data_parallel_time = time.time() - start_time
    
    results["data_parallel"] = {
        "time_seconds": data_parallel_time,
        "output_shape": data_parallel_result.shape,
        "throughput_gflops": (2 * batch_size * matrix_size**3) / (data_parallel_time * 1e9)
    }
    
    # 2. 模型并行测试 (输出分割)
    print("\n2️⃣  Model Parallel Test (Output Split)")
    start_time = time.time()
    model_parallel_result = parallel_compute.model_parallel_matmul(
        batch_a[0], single_b, split_dim="output"
    )
    model_parallel_time = time.time() - start_time
    
    results["model_parallel_output"] = {
        "time_seconds": model_parallel_time,
        "output_shape": model_parallel_result.shape,
        "throughput_gflops": (2 * matrix_size**3) / (model_parallel_time * 1e9)
    }
    
    # 3. 流水线并行测试
    print("\n3️⃣  Pipeline Parallel Test")
    weights = [
        jax.random.normal(jax.random.key(i), (matrix_size, matrix_size)) 
        for i in range(min(3, num_devices))
    ]
    
    start_time = time.time()
    pipeline_result = parallel_compute.pipeline_parallel_matmul(
        batch_a[0], weights
    )
    pipeline_time = time.time() - start_time
    
    results["pipeline_parallel"] = {
        "time_seconds": pipeline_time,
        "output_shape": pipeline_result.shape,
        "layers": len(weights)
    }
    
    # 打印基准测试结果
    print("\n📊 Benchmark Results")
    print("=" * 60)
    for strategy, result in results.items():
        print(f"{strategy:25}: {result['time_seconds']:.3f}s, {result['output_shape']}")
        if 'throughput_gflops' in result:
            print(f"{'':25}  Throughput: {result['throughput_gflops']:.2f} GFLOPS")
    
    return results