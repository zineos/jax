#!/usr/bin/env python3
"""
方案2：构建2D NPU拓扑网格

定义虚拟NPU设备的物理拓扑结构，而不是JAX的计算Mesh。

核心区别：
- NPU拓扑Mesh：虚拟NPU设备的物理排列（2x4网格、环形、树形等）
- JAX计算Mesh：数据分片的逻辑组织

作者: Cursor AI Assistant
日期: 2025-10-15
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


print("=" * 80)
print("方案2：虚拟NPU的2D拓扑Mesh")
print("=" * 80)


# ============================================================================
# 虚拟NPU设备定义（带拓扑信息）
# ============================================================================

@dataclass
class VirtualNPU:
    """虚拟NPU设备（带2D拓扑位置）"""
    
    # 基本信息
    id: int                      # 全局唯一ID
    jax_device: any             # 对应的JAX设备
    
    # 拓扑位置（2D mesh中的位置）
    mesh_row: int               # 在mesh中的行位置
    mesh_col: int               # 在mesh中的列位置
    
    # 邻居信息（2D mesh的邻居）
    neighbors: Dict[str, Optional['VirtualNPU']] = None
    
    # NPU特性
    chip_model: str = "NPU-900"
    memory_gb: int = 32
    compute_units: int = 32
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = {
                'north': None,
                'south': None,
                'east': None,
                'west': None,
            }
    
    def __repr__(self):
        return f"NPU[{self.id}]@({self.mesh_row},{self.mesh_col})"
    
    def get_position(self) -> Tuple[int, int]:
        """获取在mesh中的位置"""
        return (self.mesh_row, self.mesh_col)
    
    def get_neighbors(self) -> List['VirtualNPU']:
        """获取所有邻居NPU"""
        return [n for n in self.neighbors.values() if n is not None]
    
    def distance_to(self, other: 'VirtualNPU') -> int:
        """计算到另一个NPU的曼哈顿距离"""
        return abs(self.mesh_row - other.mesh_row) + \
               abs(self.mesh_col - other.mesh_col)


# ============================================================================
# 2D NPU Mesh拓扑管理器
# ============================================================================

class NPUTopologyMesh:
    """2D NPU拓扑Mesh管理器"""
    
    def __init__(
        self,
        rows: int,
        cols: int,
        topology_type: str = "mesh",  # mesh, torus, ring
        chip_model: str = "NPU-900"
    ):
        """
        创建2D NPU拓扑
        
        Args:
            rows: mesh的行数
            cols: mesh的列数
            topology_type: 拓扑类型（mesh, torus, ring）
            chip_model: NPU芯片型号
        """
        self.rows = rows
        self.cols = cols
        self.topology_type = topology_type
        self.chip_model = chip_model
        
        # 获取JAX设备
        jax_devices = jax.devices()
        total_npus = rows * cols
        
        if len(jax_devices) < total_npus:
            raise ValueError(
                f"需要{total_npus}个JAX设备，但只有{len(jax_devices)}个"
            )
        
        # 创建NPU设备（2D网格）
        self.npus: List[List[VirtualNPU]] = []
        self._build_npu_grid(jax_devices[:total_npus])
        
        # 建立邻居关系
        self._setup_neighbors()
        
        # 扁平化列表（方便访问）
        self.npu_list = [npu for row in self.npus for npu in row]
    
    def _build_npu_grid(self, jax_devices):
        """构建NPU网格"""
        npu_id = 0
        for row in range(self.rows):
            npu_row = []
            for col in range(self.cols):
                npu = VirtualNPU(
                    id=npu_id,
                    jax_device=jax_devices[npu_id],
                    mesh_row=row,
                    mesh_col=col,
                    chip_model=self.chip_model,
                )
                npu_row.append(npu)
                npu_id += 1
            self.npus.append(npu_row)
    
    def _setup_neighbors(self):
        """建立邻居关系"""
        for row in range(self.rows):
            for col in range(self.cols):
                npu = self.npus[row][col]
                
                # North (上)
                if row > 0:
                    npu.neighbors['north'] = self.npus[row - 1][col]
                elif self.topology_type == 'torus':
                    npu.neighbors['north'] = self.npus[self.rows - 1][col]
                
                # South (下)
                if row < self.rows - 1:
                    npu.neighbors['south'] = self.npus[row + 1][col]
                elif self.topology_type == 'torus':
                    npu.neighbors['south'] = self.npus[0][col]
                
                # West (左)
                if col > 0:
                    npu.neighbors['west'] = self.npus[row][col - 1]
                elif self.topology_type == 'torus':
                    npu.neighbors['west'] = self.npus[row][self.cols - 1]
                
                # East (右)
                if col < self.cols - 1:
                    npu.neighbors['east'] = self.npus[row][col + 1]
                elif self.topology_type == 'torus':
                    npu.neighbors['east'] = self.npus[row][0]
    
    def get_npu(self, row: int, col: int) -> VirtualNPU:
        """获取指定位置的NPU"""
        return self.npus[row][col]
    
    def get_npu_by_id(self, npu_id: int) -> VirtualNPU:
        """根据ID获取NPU"""
        return self.npu_list[npu_id]
    
    def get_jax_devices(self) -> List:
        """获取所有JAX设备（用于JAX计算）"""
        return [npu.jax_device for npu in self.npu_list]
    
    def print_topology(self):
        """打印拓扑结构"""
        print(f"\n2D NPU Mesh拓扑 ({self.rows}×{self.cols})")
        print(f"拓扑类型: {self.topology_type}")
        print(f"总NPU数: {len(self.npu_list)}")
        print("\n网格布局:")
        
        # 打印列号
        print("     " + "  ".join([f"col{c}" for c in range(self.cols)]))
        print("   +" + "------+" * self.cols)
        
        # 打印每一行
        for row in range(self.rows):
            row_str = f"r{row} |"
            for col in range(self.cols):
                npu = self.npus[row][col]
                row_str += f" NPU{npu.id:2d}|"
            print(row_str)
            print("   +" + "------+" * self.cols)
        
        print(f"\n互联信息:")
        print(f"  每个NPU的邻居数: {self._get_avg_neighbors():.1f}")
        if self.topology_type == "torus":
            print(f"  环形互联: 是（边缘NPU环绕连接）")
        else:
            print(f"  环形互联: 否（边缘NPU无环绕）")
    
    def _get_avg_neighbors(self):
        """计算平均邻居数"""
        total_neighbors = sum(
            len(npu.get_neighbors()) for npu in self.npu_list
        )
        return total_neighbors / len(self.npu_list)
    
    def print_npu_details(self, row: int, col: int):
        """打印指定NPU的详细信息"""
        npu = self.get_npu(row, col)
        print(f"\n{npu} 详细信息:")
        print(f"  位置: 第{row}行, 第{col}列")
        print(f"  JAX设备: {npu.jax_device}")
        print(f"  芯片型号: {npu.chip_model}")
        print(f"  内存: {npu.memory_gb}GB")
        print(f"  计算单元: {npu.compute_units}")
        print(f"  邻居NPU:")
        for direction, neighbor in npu.neighbors.items():
            if neighbor:
                print(f"    {direction:5s}: {neighbor}")
            else:
                print(f"    {direction:5s}: 无")


# ============================================================================
# 示例1: 基础2×2 Mesh
# ============================================================================

def demo_2x2_mesh():
    """2×2 NPU mesh"""
    print("\n" + "=" * 80)
    print("【示例1】2×2 NPU Mesh拓扑")
    print("=" * 80)
    
    # 创建2×2 mesh
    mesh = NPUTopologyMesh(rows=2, cols=2, topology_type="mesh")
    
    # 打印拓扑
    mesh.print_topology()
    
    # 查看具体NPU
    print("\n" + "-" * 80)
    mesh.print_npu_details(0, 0)  # 左上角
    mesh.print_npu_details(1, 1)  # 右下角
    
    return mesh


# ============================================================================
# 示例2: 4×2 Mesh（常见配置）
# ============================================================================

def demo_4x2_mesh():
    """4×2 NPU mesh"""
    print("\n" + "=" * 80)
    print("【示例2】4×2 NPU Mesh拓扑")
    print("=" * 80)
    
    # 创建4×2 mesh
    mesh = NPUTopologyMesh(rows=4, cols=2, topology_type="mesh")
    
    # 打印拓扑
    mesh.print_topology()
    
    # 分析拓扑
    print("\n拓扑分析:")
    print(f"  角NPU（2个邻居）: NPU0, NPU1, NPU6, NPU7")
    print(f"  边NPU（3个邻居）: NPU2, NPU3, NPU4, NPU5")
    print(f"  中心NPU（4个邻居）: 无（此拓扑无中心）")
    
    return mesh


# ============================================================================
# 示例3: 2×4 Torus（环形互联）
# ============================================================================

def demo_2x4_torus():
    """2×4 NPU torus（环形）"""
    print("\n" + "=" * 80)
    print("【示例3】2×4 NPU Torus拓扑（环形互联）")
    print("=" * 80)
    
    # 创建2×4 torus
    mesh = NPUTopologyMesh(rows=2, cols=4, topology_type="torus")
    
    # 打印拓扑
    mesh.print_topology()
    
    # 演示环形连接
    print("\n环形互联示例:")
    npu_top_left = mesh.get_npu(0, 0)
    npu_top_right = mesh.get_npu(0, 3)
    
    print(f"  {npu_top_left} 的东边邻居: {npu_top_left.neighbors['east']}")
    print(f"  {npu_top_right} 的东边邻居: {npu_top_right.neighbors['east']} (环绕)")
    
    npu_bottom_left = mesh.get_npu(1, 0)
    npu_top_left = mesh.get_npu(0, 0)
    
    print(f"  {npu_top_left} 的北边邻居: {npu_top_left.neighbors['north']} (环绕)")
    print(f"  {npu_bottom_left} 的南边邻居: {npu_bottom_left.neighbors['south']} (环绕)")
    
    return mesh


# ============================================================================
# 示例4: NPU拓扑 + JAX计算
# ============================================================================

def demo_topology_with_jax():
    """结合NPU拓扑和JAX计算"""
    print("\n" + "=" * 80)
    print("【示例4】NPU拓扑 + JAX计算结合")
    print("=" * 80)
    
    # 创建2×2 NPU mesh拓扑
    npu_mesh = NPUTopologyMesh(rows=2, cols=2, topology_type="mesh")
    
    print("\n1️⃣  NPU物理拓扑:")
    npu_mesh.print_topology()
    
    # 使用这些NPU对应的JAX设备进行计算
    from jax import pmap
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    
    print("\n2️⃣  使用这些NPU进行JAX计算:")
    
    # 获取JAX设备
    jax_devices = npu_mesh.get_jax_devices()
    print(f"\nJAX设备列表:")
    for i, (npu, dev) in enumerate(zip(npu_mesh.npu_list, jax_devices)):
        print(f"  {npu} → {dev}")
    
    # 方法A: 使用pmap（自动按NPU ID顺序）
    print(f"\n方法A: pmap并行计算（按NPU ID顺序分配）")
    
    @pmap
    def compute_on_npus(x):
        return x * 2
    
    x = jnp.ones((4, 100))  # 4个NPU
    result = compute_on_npus(x)
    
    print(f"  输入: {x.shape}")
    print(f"  输出: {result.shape}")
    print(f"  数据分配:")
    print(f"    NPU0 处理: x[0]")
    print(f"    NPU1 处理: x[1]")
    print(f"    NPU2 处理: x[2]")
    print(f"    NPU3 处理: x[3]")
    
    # 方法B: 使用Mesh（按2D拓扑分配）
    print(f"\n方法B: Mesh 2D分片（按物理拓扑分配）")
    
    # 创建JAX Mesh，形状对应NPU物理拓扑
    jax_mesh = Mesh(
        np.array(jax_devices).reshape(2, 2),  # 和NPU拓扑相同的形状！
        ('row', 'col')  # 对应NPU的row和col
    )
    
    print(f"  JAX Mesh形状: {jax_mesh.shape}")
    print(f"  对应NPU拓扑: {npu_mesh.rows}×{npu_mesh.cols}")
    
    # 2D分片
    X = jnp.ones((200, 400))
    sharding = NamedSharding(jax_mesh, P('row', 'col'))
    X_sharded = jax.device_put(X, sharding)
    
    print(f"\n  数据分片（对应NPU位置）:")
    print(f"    NPU[0]@(0,0) 存储: X[0:100, 0:200]")
    print(f"    NPU[1]@(0,1) 存储: X[0:100, 200:400]")
    print(f"    NPU[2]@(1,0) 存储: X[100:200, 0:200]")
    print(f"    NPU[3]@(1,1) 存储: X[100:200, 200:400]")
    
    print(f"\n✓ NPU物理拓扑与JAX计算完美对应！")


# ============================================================================
# 示例5: 基于拓扑的通信模拟
# ============================================================================

def demo_topology_communication():
    """基于2D拓扑的通信模拟"""
    print("\n" + "=" * 80)
    print("【示例5】基于2D拓扑的通信模拟")
    print("=" * 80)
    
    # 创建4×2 mesh
    mesh = NPUTopologyMesh(rows=4, cols=2, topology_type="mesh")
    
    print("\n拓扑结构:")
    mesh.print_topology()
    
    # 通信场景1: 相邻NPU通信
    print("\n场景1: 相邻NPU通信（1-hop）")
    npu0 = mesh.get_npu(0, 0)
    npu1 = mesh.get_npu(0, 1)
    
    print(f"  {npu0} → {npu1}")
    print(f"  距离: {npu0.distance_to(npu1)} hop")
    print(f"  是邻居: {npu1 in npu0.get_neighbors()}")
    
    # 通信场景2: 跨对角线通信
    print("\n场景2: 对角线通信（多hop）")
    npu_topleft = mesh.get_npu(0, 0)
    npu_bottomright = mesh.get_npu(3, 1)
    
    print(f"  {npu_topleft} → {npu_bottomright}")
    print(f"  距离: {npu_topleft.distance_to(npu_bottomright)} hop")
    
    # 通信场景3: 集体通信（All-Reduce）
    print("\n场景3: 集体通信模拟（All-Reduce）")
    print(f"  参与NPU: 全部{len(mesh.npu_list)}个")
    print(f"  拓扑: {mesh.rows}×{mesh.cols} mesh")
    print(f"  最优算法: 2D递归减半（Recursive Halving）")
    print(f"  通信步数: {mesh.rows + mesh.cols - 2} 步")


# ============================================================================
# 示例6: 多种拓扑对比
# ============================================================================

def demo_topology_comparison():
    """对比不同拓扑类型"""
    print("\n" + "=" * 80)
    print("【示例6】不同2D拓扑对比")
    print("=" * 80)
    
    # 创建不同拓扑
    mesh_2x4 = NPUTopologyMesh(rows=2, cols=4, topology_type="mesh")
    torus_2x4 = NPUTopologyMesh(rows=2, cols=4, topology_type="torus")
    
    print("\n配置1: 2×4 Mesh（标准网格）")
    print(f"  总NPU: 8")
    print(f"  平均邻居: {mesh_2x4._get_avg_neighbors():.1f}")
    print(f"  最大距离: 4 hop")
    
    print("\n配置2: 2×4 Torus（环形网格）")
    print(f"  总NPU: 8")
    print(f"  平均邻居: {torus_2x4._get_avg_neighbors():.1f}")
    print(f"  最大距离: 2 hop（环绕）")
    
    print("\n对比:")
    print(f"  Torus的优势: 更低的最大通信延迟")
    print(f"  Mesh的优势: 更简单的物理实现")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """运行所有演示"""
    
    print("\n💡 核心概念:")
    print("  NPU拓扑Mesh: 虚拟NPU设备的物理排列")
    print("  JAX计算Mesh: 数据在设备上的逻辑分片")
    print("  两者可以对应: NPU拓扑 → JAX Mesh形状\n")
    
    try:
        # 基础演示
        demo_2x2_mesh()
        demo_4x2_mesh()
        demo_2x4_torus()
        
        # 高级演示
        demo_topology_with_jax()
        demo_topology_communication()
        demo_topology_comparison()
        
        print("\n" + "=" * 80)
        print("✅ 所有演示完成！")
        print("=" * 80)
        
        print("\n📚 总结:")
        print("  1. NPUTopologyMesh: 定义虚拟NPU的2D物理布局")
        print("  2. 支持mesh和torus拓扑类型")
        print("  3. 每个NPU知道自己的邻居")
        print("  4. 可以计算NPU间的距离")
        print("  5. NPU拓扑可映射到JAX Mesh进行计算")
        
        print("\n🎯 关键代码:")
        print("  # 创建2D NPU拓扑")
        print("  npu_mesh = NPUTopologyMesh(rows=2, cols=4, topology_type='mesh')")
        print("  ")
        print("  # 获取NPU")
        print("  npu = npu_mesh.get_npu(row=1, col=2)")
        print("  ")
        print("  # 查看邻居")
        print("  neighbors = npu.get_neighbors()")
        print("  ")
        print("  # 用于JAX计算")
        print("  jax_devices = npu_mesh.get_jax_devices()")
        print("  jax_mesh = Mesh(np.array(jax_devices).reshape(2, 4), ...)")
        
    except Exception as e:
        print(f"\n⚠️  需要JAX环境")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
