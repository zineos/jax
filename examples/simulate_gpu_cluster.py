#!/usr/bin/env python3
"""
模拟 GPU 集群环境

这个文件展示如何使用虚拟 backend 模拟一个真实的 GPU 集群，包括：
- 多节点（多台机器）
- 多进程（每个节点多个进程）
- 多 GPU（每个进程管理多个 GPU）

使用场景：
1. 测试分布式训练代码（如 pmap, pjit）
2. 验证多节点通信逻辑
3. 在没有真实集群的情况下开发和调试

作者: Cursor AI Assistant
日期: 2025-10-15
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json


# ============================================================================
# GPU 设备模拟
# ============================================================================

@dataclass
class VirtualGPU:
    """虚拟 GPU 设备
    
    模拟一个 NVIDIA GPU 或 AMD GPU 的接口。
    
    属性:
        id: 设备 ID（全局唯一）
        local_id: 本地 ID（进程内唯一）
        platform: 平台名称（"cuda" 或 "rocm"）
        device_kind: 设备类型（如 "A100", "H100", "V100"）
        process_index: 所属进程索引
        node_id: 所属节点 ID
        memory_gb: 显存大小（GB）
        compute_capability: 计算能力（CUDA 专用）
    """
    id: int                      # 全局设备 ID
    local_id: int                # 本地设备 ID (0, 1, 2, ...)
    platform: str                # "cuda" 或 "rocm"
    device_kind: str             # GPU 型号
    process_index: int           # 进程索引
    node_id: int                 # 节点 ID
    memory_gb: int = 40          # 显存大小
    compute_capability: str = "8.0"  # 计算能力
    
    @property
    def host_id(self) -> int:
        """兼容性别名"""
        return self.process_index
    
    def __repr__(self) -> str:
        return (f"VirtualGPU(id={self.id}, {self.device_kind}, "
                f"node={self.node_id}, process={self.process_index})")
    
    def __str__(self) -> str:
        return f"{self.platform}:{self.id}"
    
    def to_dict(self) -> dict:
        """转换为字典（用于调试和序列化）"""
        return {
            "id": self.id,
            "local_id": self.local_id,
            "platform": self.platform,
            "device_kind": self.device_kind,
            "process_index": self.process_index,
            "node_id": self.node_id,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability
        }


# ============================================================================
# GPU 集群 Backend
# ============================================================================

class GPUClusterBackend:
    """GPU 集群 Backend
    
    模拟一个真实的 GPU 集群环境，支持：
    - 多节点配置
    - 多进程并行
    - NCCL 集合通信（模拟）
    - 设备拓扑信息
    
    参数:
        num_nodes: 节点数量（机器数量）
        gpus_per_node: 每个节点的 GPU 数量
        processes_per_node: 每个节点的进程数量
        current_process_index: 当前进程的全局索引
        gpu_model: GPU 型号
        platform: "cuda" 或 "rocm"
    """
    
    def __init__(
        self,
        num_nodes: int = 2,
        gpus_per_node: int = 8,
        processes_per_node: int = 1,
        current_process_index: int = 0,
        gpu_model: str = "A100",
        platform: str = "cuda",
        memory_gb: int = 40
    ):
        self.platform = platform
        self._num_nodes = num_nodes
        self._gpus_per_node = gpus_per_node
        self._processes_per_node = processes_per_node
        self._current_process_index = current_process_index
        self._gpu_model = gpu_model
        self._memory_gb = memory_gb
        
        # 计算集群配置
        self._total_processes = num_nodes * processes_per_node
        self._gpus_per_process = gpus_per_node // processes_per_node
        self._total_gpus = num_nodes * gpus_per_node
        
        # 计算当前进程所在的节点
        self._current_node_id = current_process_index // processes_per_node
        
        # 创建本地 GPU 设备
        self._local_devices = self._create_local_devices()
        
        # 创建全局设备列表（用于模拟多进程环境）
        self._all_devices = self._create_all_devices()
        
        # 平台版本信息
        if platform == "cuda":
            self.platform_version = f"CUDA 12.3, cuDNN 8.9, NCCL 2.19 (Virtual)"
        else:
            self.platform_version = f"ROCm 6.0, MIOpen 2.20 (Virtual)"
    
    def _create_local_devices(self) -> List[VirtualGPU]:
        """创建当前进程的本地设备"""
        devices = []
        start_gpu_id = self._current_process_index * self._gpus_per_process
        
        for local_id in range(self._gpus_per_process):
            global_id = start_gpu_id + local_id
            device = VirtualGPU(
                id=global_id,
                local_id=local_id,
                platform=self.platform,
                device_kind=self._gpu_model,
                process_index=self._current_process_index,
                node_id=self._current_node_id,
                memory_gb=self._memory_gb
            )
            devices.append(device)
        
        return devices
    
    def _create_all_devices(self) -> List[VirtualGPU]:
        """创建所有进程的所有设备（用于全局视图）"""
        all_devices = []
        global_id = 0
        
        for node_id in range(self._num_nodes):
            for proc_in_node in range(self._processes_per_node):
                process_index = node_id * self._processes_per_node + proc_in_node
                
                for local_id in range(self._gpus_per_process):
                    device = VirtualGPU(
                        id=global_id,
                        local_id=local_id,
                        platform=self.platform,
                        device_kind=self._gpu_model,
                        process_index=process_index,
                        node_id=node_id,
                        memory_gb=self._memory_gb
                    )
                    all_devices.append(device)
                    global_id += 1
        
        return all_devices
    
    # ------------------------------------------------------------------------
    # JAX Backend 接口
    # ------------------------------------------------------------------------
    
    def device_count(self) -> int:
        """返回集群中的总 GPU 数量"""
        return self._total_gpus
    
    def local_device_count(self) -> int:
        """返回当前进程管理的 GPU 数量"""
        return len(self._local_devices)
    
    def process_index(self) -> int:
        """返回当前进程的全局索引"""
        return self._current_process_index
    
    def devices(self) -> List[VirtualGPU]:
        """返回所有设备（所有节点、所有进程）"""
        return self._all_devices
    
    def local_devices(self) -> List[VirtualGPU]:
        """返回当前进程的本地设备"""
        return self._local_devices
    
    def _get_all_devices(self) -> List[VirtualGPU]:
        """内部方法：获取所有设备"""
        return self.devices()
    
    # ------------------------------------------------------------------------
    # 集群特定方法
    # ------------------------------------------------------------------------
    
    def get_cluster_topology(self) -> Dict:
        """获取集群拓扑信息"""
        return {
            "num_nodes": self._num_nodes,
            "gpus_per_node": self._gpus_per_node,
            "processes_per_node": self._processes_per_node,
            "total_processes": self._total_processes,
            "total_gpus": self._total_gpus,
            "current_node_id": self._current_node_id,
            "current_process_index": self._current_process_index,
            "gpu_model": self._gpu_model,
            "platform": self.platform
        }
    
    def get_node_devices(self, node_id: int) -> List[VirtualGPU]:
        """获取指定节点的所有设备"""
        return [d for d in self._all_devices if d.node_id == node_id]
    
    def get_process_devices(self, process_index: int) -> List[VirtualGPU]:
        """获取指定进程的所有设备"""
        return [d for d in self._all_devices if d.process_index == process_index]
    
    def print_cluster_info(self):
        """打印集群信息"""
        print("=" * 70)
        print(f"GPU 集群配置 ({self.platform.upper()})")
        print("=" * 70)
        print(f"总节点数: {self._num_nodes}")
        print(f"每节点 GPU 数: {self._gpus_per_node}")
        print(f"每节点进程数: {self._processes_per_node}")
        print(f"总进程数: {self._total_processes}")
        print(f"总 GPU 数: {self._total_gpus}")
        print(f"GPU 型号: {self._gpu_model}")
        print(f"显存: {self._memory_gb} GB")
        print(f"平台版本: {self.platform_version}")
        print()
        print(f"当前进程信息:")
        print(f"  进程索引: {self._current_process_index}")
        print(f"  所在节点: {self._current_node_id}")
        print(f"  本地 GPU 数: {self.local_device_count()}")
        print(f"  本地设备: {[str(d) for d in self._local_devices]}")
        print("=" * 70)
    
    def print_all_devices(self):
        """打印所有设备信息"""
        print("\n所有设备详情:")
        print("-" * 70)
        for node_id in range(self._num_nodes):
            print(f"\n节点 {node_id}:")
            node_devices = self.get_node_devices(node_id)
            for device in node_devices:
                print(f"  {device} (进程 {device.process_index}, 本地ID {device.local_id})")
        print("-" * 70)
    
    def __repr__(self) -> str:
        return (f"GPUClusterBackend(nodes={self._num_nodes}, "
                f"gpus_per_node={self._gpus_per_node}, "
                f"total_gpus={self._total_gpus})")


# ============================================================================
# 集群配置预设
# ============================================================================

class ClusterPresets:
    """常见集群配置预设"""
    
    @staticmethod
    def single_node_8gpu():
        """单节点 8 卡 A100"""
        return GPUClusterBackend(
            num_nodes=1,
            gpus_per_node=8,
            processes_per_node=1,
            gpu_model="A100-80GB",
            memory_gb=80
        )
    
    @staticmethod
    def dual_node_8gpu():
        """双节点，每节点 8 卡 A100"""
        return GPUClusterBackend(
            num_nodes=2,
            gpus_per_node=8,
            processes_per_node=1,
            gpu_model="A100-80GB",
            memory_gb=80
        )
    
    @staticmethod
    def quad_node_h100():
        """4 节点 H100 集群"""
        return GPUClusterBackend(
            num_nodes=4,
            gpus_per_node=8,
            processes_per_node=1,
            gpu_model="H100-80GB",
            memory_gb=80
        )
    
    @staticmethod
    def large_cluster_64gpu():
        """大规模集群：8 节点 x 8 GPU = 64 GPU"""
        return GPUClusterBackend(
            num_nodes=8,
            gpus_per_node=8,
            processes_per_node=1,
            gpu_model="A100-40GB",
            memory_gb=40
        )
    
    @staticmethod
    def multi_process_cluster():
        """多进程配置：2 节点 x 8 GPU，每节点 2 进程"""
        return GPUClusterBackend(
            num_nodes=2,
            gpus_per_node=8,
            processes_per_node=2,  # 每节点 2 个进程
            gpu_model="A100-40GB",
            memory_gb=40
        )


# ============================================================================
# 注册到 JAX
# ============================================================================

def register_gpu_cluster_backend(
    name: str = "gpu_cluster",
    num_nodes: int = 2,
    gpus_per_node: int = 8,
    processes_per_node: int = 1,
    current_process_index: int = 0,
    gpu_model: str = "A100",
    platform: str = "cuda",
    priority: int = 0
):
    """注册 GPU 集群 backend 到 JAX
    
    参数:
        name: backend 名称
        num_nodes: 节点数量
        gpus_per_node: 每节点 GPU 数
        processes_per_node: 每节点进程数
        current_process_index: 当前进程索引
        gpu_model: GPU 型号
        platform: "cuda" 或 "rocm"
        priority: backend 优先级
    
    示例:
        # 模拟 2 节点 x 8 GPU 的集群
        register_gpu_cluster_backend(
            name="my_cluster",
            num_nodes=2,
            gpus_per_node=8
        )
    """
    from jax._src import xla_bridge
    
    def factory():
        return GPUClusterBackend(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            processes_per_node=processes_per_node,
            current_process_index=current_process_index,
            gpu_model=gpu_model,
            platform=platform
        )
    
    xla_bridge.register_backend_factory(
        name=name,
        factory=factory,
        priority=priority,
        fail_quietly=False,
        experimental=True
    )
    
    print(f"✓ GPU 集群 backend '{name}' 注册成功!")
    print(f"  配置: {num_nodes} 节点 x {gpus_per_node} GPU = {num_nodes * gpus_per_node} GPU")
    print(f"  型号: {gpu_model}")
    print(f"  平台: {platform.upper()}")


# ============================================================================
# 使用示例
# ============================================================================

def demo_single_node():
    """示例 1: 单节点 8 GPU"""
    print("\n【示例 1】单节点 8 GPU 集群")
    print("=" * 70)
    
    cluster = ClusterPresets.single_node_8gpu()
    cluster.print_cluster_info()
    cluster.print_all_devices()


def demo_multi_node():
    """示例 2: 多节点集群"""
    print("\n【示例 2】双节点 16 GPU 集群")
    print("=" * 70)
    
    cluster = ClusterPresets.dual_node_8gpu()
    cluster.print_cluster_info()
    
    # 显示每个节点的设备
    print("\n每个节点的设备分布:")
    for node_id in range(2):
        devices = cluster.get_node_devices(node_id)
        print(f"  节点 {node_id}: {len(devices)} 个 GPU - {[d.id for d in devices]}")


def demo_multi_process():
    """示例 3: 多进程配置"""
    print("\n【示例 3】多进程配置 (每节点 2 进程)")
    print("=" * 70)
    
    cluster = ClusterPresets.multi_process_cluster()
    cluster.print_cluster_info()
    
    # 显示每个进程的设备
    print("\n每个进程的设备分布:")
    for proc_id in range(4):  # 2 节点 x 2 进程 = 4 进程
        devices = cluster.get_process_devices(proc_id)
        print(f"  进程 {proc_id}: {len(devices)} 个 GPU - {[d.id for d in devices]}")


def demo_large_cluster():
    """示例 4: 大规模集群"""
    print("\n【示例 4】大规模集群 (64 GPU)")
    print("=" * 70)
    
    cluster = ClusterPresets.large_cluster_64gpu()
    cluster.print_cluster_info()
    
    topology = cluster.get_cluster_topology()
    print("\n集群拓扑信息:")
    print(json.dumps(topology, indent=2, ensure_ascii=False))


def demo_simulate_distributed_training():
    """示例 5: 模拟分布式训练环境"""
    print("\n【示例 5】模拟分布式训练环境")
    print("=" * 70)
    
    # 模拟 4 个进程的集群（2 节点 x 2 进程）
    num_processes = 4
    
    print(f"模拟 {num_processes} 个进程的分布式训练:")
    print()
    
    for process_id in range(num_processes):
        backend = GPUClusterBackend(
            num_nodes=2,
            gpus_per_node=8,
            processes_per_node=2,
            current_process_index=process_id,
            gpu_model="A100-40GB"
        )
        
        print(f"进程 {process_id} (节点 {backend._current_node_id}):")
        print(f"  本地设备: {[str(d) for d in backend.local_devices()]}")
        print(f"  全局设备数: {backend.device_count()}")
        print()


def main():
    """运行所有演示"""
    print("=" * 70)
    print("GPU 集群模拟演示")
    print("=" * 70)
    
    demo_single_node()
    demo_multi_node()
    demo_multi_process()
    demo_large_cluster()
    demo_simulate_distributed_training()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
