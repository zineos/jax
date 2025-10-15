#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>

namespace npu {

// NPU设备信息
struct NPUDeviceInfo {
    int device_id;
    size_t memory_size;  // 以字节为单位
    int compute_units;   // 计算单元数量
    float peak_flops;    // 峰值FLOPS
};

// NPU内存管理
class NPUMemory {
public:
    NPUMemory(size_t size, int device_id);
    ~NPUMemory();
    
    void* data() const { return data_; }
    size_t size() const { return size_; }
    int device_id() const { return device_id_; }
    
private:
    void* data_;
    size_t size_;
    int device_id_;
};

// NPU计算内核
class NPUKernel {
public:
    // 矩阵乘法: C = A * B
    static void MatMul(
        const float* a, int a_rows, int a_cols,
        const float* b, int b_rows, int b_cols,
        float* c, int device_id = 0
    );
    
    // 批量矩阵乘法
    static void BatchMatMul(
        const float* a, const float* b, float* c,
        int batch_size, int m, int k, int n,
        int device_id = 0
    );
    
    // NPU特定的优化矩阵乘法（模拟NPU架构特性）
    static void OptimizedMatMul(
        const float* a, int a_rows, int a_cols,
        const float* b, int b_rows, int b_cols,
        float* c, int device_id = 0
    );

private:
    // 模拟NPU计算延迟
    static void SimulateComputeDelay(size_t flops, int device_id);
};

// NPU设备管理器
class NPUDevice {
public:
    NPUDevice(int device_id, const NPUDeviceInfo& info);
    
    int device_id() const { return device_id_; }
    const NPUDeviceInfo& info() const { return info_; }
    
    // 内存分配
    std::unique_ptr<NPUMemory> AllocateMemory(size_t size);
    
    // 数据传输
    void MemcpyHostToDevice(void* dst, const void* src, size_t size);
    void MemcpyDeviceToHost(void* dst, const void* src, size_t size);
    
    // 同步操作
    void Synchronize();
    
private:
    int device_id_;
    NPUDeviceInfo info_;
    size_t allocated_memory_;
    std::vector<std::unique_ptr<NPUMemory>> memory_pool_;
};

// NPU运行时管理器
class NPURuntime {
public:
    static NPURuntime& GetInstance();
    
    // 初始化NPU设备
    void Initialize(int num_devices = 4);
    
    // 获取设备数量
    int GetDeviceCount() const { return static_cast<int>(devices_.size()); }
    
    // 获取设备
    NPUDevice* GetDevice(int device_id);
    
    // 获取当前设备
    int GetCurrentDevice() const { return current_device_; }
    void SetCurrentDevice(int device_id);
    
private:
    NPURuntime() = default;
    std::vector<std::unique_ptr<NPUDevice>> devices_;
    int current_device_ = 0;
};

} // namespace npu