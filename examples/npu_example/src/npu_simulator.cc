#include "npu_simulator.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace npu {

// NPU内存实现
NPUMemory::NPUMemory(size_t size, int device_id) 
    : size_(size), device_id_(device_id) {
    data_ = std::aligned_alloc(64, size);  // 64字节对齐，模拟NPU内存要求
    if (!data_) {
        throw std::runtime_error("Failed to allocate NPU memory");
    }
    std::memset(data_, 0, size);
    std::cout << "NPU[" << device_id << "] Allocated " << size << " bytes" << std::endl;
}

NPUMemory::~NPUMemory() {
    if (data_) {
        std::free(data_);
        std::cout << "NPU[" << device_id_ << "] Freed " << size_ << " bytes" << std::endl;
    }
}

// NPU内核实现
void NPUKernel::MatMul(
    const float* a, int a_rows, int a_cols,
    const float* b, int b_rows, int b_cols,
    float* c, int device_id) {
    
    if (a_cols != b_rows) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    size_t flops = static_cast<size_t>(a_rows) * a_cols * b_cols * 2;
    std::cout << "NPU[" << device_id << "] MatMul: (" << a_rows << "x" << a_cols 
              << ") × (" << b_rows << "x" << b_cols << ") = " << flops << " FLOPS" << std::endl;
    
    // 模拟NPU计算延迟
    SimulateComputeDelay(flops, device_id);
    
    // 标准矩阵乘法实现（模拟NPU计算）
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; ++k) {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            c[i * b_cols + j] = sum;
        }
    }
}

void NPUKernel::BatchMatMul(
    const float* a, const float* b, float* c,
    int batch_size, int m, int k, int n,
    int device_id) {
    
    std::cout << "NPU[" << device_id << "] BatchMatMul: batch=" << batch_size 
              << " (" << m << "x" << k << ") × (" << k << "x" << n << ")" << std::endl;
    
    size_t batch_flops = static_cast<size_t>(m) * k * n * 2;
    SimulateComputeDelay(batch_flops * batch_size, device_id);
    
    for (int b = 0; b < batch_size; ++b) {
        const float* a_batch = a + b * m * k;
        const float* b_batch = b + b * k * n;
        float* c_batch = c + b * m * n;
        
        MatMul(a_batch, m, k, b_batch, k, n, c_batch, device_id);
    }
}

void NPUKernel::OptimizedMatMul(
    const float* a, int a_rows, int a_cols,
    const float* b, int b_rows, int b_cols,
    float* c, int device_id) {
    
    std::cout << "NPU[" << device_id << "] OptimizedMatMul: Using NPU-specific optimizations" << std::endl;
    
    // 模拟NPU特定优化：块矩阵乘法
    const int BLOCK_SIZE = 64;  // NPU优化的块大小
    
    size_t flops = static_cast<size_t>(a_rows) * a_cols * b_cols * 2;
    // NPU优化版本比标准版本快30%
    SimulateComputeDelay(static_cast<size_t>(flops * 0.7), device_id);
    
    // 分块矩阵乘法（模拟NPU并行计算单元）
    for (int ii = 0; ii < a_rows; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < b_cols; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < a_cols; kk += BLOCK_SIZE) {
                
                int i_end = std::min(ii + BLOCK_SIZE, a_rows);
                int j_end = std::min(jj + BLOCK_SIZE, b_cols);
                int k_end = std::min(kk + BLOCK_SIZE, a_cols);
                
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        float sum = (kk == 0) ? 0.0f : c[i * b_cols + j];
                        for (int k = kk; k < k_end; ++k) {
                            sum += a[i * a_cols + k] * b[k * b_cols + j];
                        }
                        c[i * b_cols + j] = sum;
                    }
                }
            }
        }
    }
}

void NPUKernel::SimulateComputeDelay(size_t flops, int device_id) {
    // 模拟NPU计算延迟：假设每个设备有不同的计算能力
    const float base_gflops = 100.0f;  // 基础100 GFLOPS
    const float device_factor = 1.0f + 0.1f * device_id;  // 不同设备性能略有差异
    
    float gflops = base_gflops * device_factor;
    float compute_time_us = (flops / 1e9f) / gflops * 1e6f;  // 微秒
    
    // 实际延迟模拟（缩放1000倍以便观察）
    auto delay_us = static_cast<int>(compute_time_us / 1000.0f);
    if (delay_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }
}

// NPU设备实现
NPUDevice::NPUDevice(int device_id, const NPUDeviceInfo& info)
    : device_id_(device_id), info_(info), allocated_memory_(0) {
    std::cout << "Initialized NPU Device " << device_id 
              << " with " << info.memory_size / (1024*1024) << " MB memory, "
              << info.compute_units << " compute units, "
              << info.peak_flops / 1e9f << " GFLOPS" << std::endl;
}

std::unique_ptr<NPUMemory> NPUDevice::AllocateMemory(size_t size) {
    if (allocated_memory_ + size > info_.memory_size) {
        throw std::runtime_error("NPU device out of memory");
    }
    
    auto memory = std::make_unique<NPUMemory>(size, device_id_);
    allocated_memory_ += size;
    return memory;
}

void NPUDevice::MemcpyHostToDevice(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
    // 模拟数据传输延迟（假设10 GB/s带宽）
    auto delay_us = static_cast<int>(size / 10000.0f);  // 微秒
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
}

void NPUDevice::MemcpyDeviceToHost(void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
    auto delay_us = static_cast<int>(size / 10000.0f);
    std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
}

void NPUDevice::Synchronize() {
    // 模拟设备同步
    std::this_thread::sleep_for(std::chrono::microseconds(10));
}

// NPU运行时实现
NPURuntime& NPURuntime::GetInstance() {
    static NPURuntime instance;
    return instance;
}

void NPURuntime::Initialize(int num_devices) {
    devices_.clear();
    devices_.reserve(num_devices);
    
    std::cout << "Initializing NPU Runtime with " << num_devices << " devices..." << std::endl;
    
    for (int i = 0; i < num_devices; ++i) {
        NPUDeviceInfo info;
        info.device_id = i;
        info.memory_size = 8ULL * 1024 * 1024 * 1024;  // 8GB per device
        info.compute_units = 128;  // 128个计算单元
        info.peak_flops = 100e9f * (1.0f + 0.1f * i);  // 每设备略有不同的性能
        
        devices_.push_back(std::make_unique<NPUDevice>(i, info));
    }
    
    current_device_ = 0;
    std::cout << "NPU Runtime initialized successfully!" << std::endl;
}

NPUDevice* NPURuntime::GetDevice(int device_id) {
    if (device_id < 0 || device_id >= static_cast<int>(devices_.size())) {
        return nullptr;
    }
    return devices_[device_id].get();
}

void NPURuntime::SetCurrentDevice(int device_id) {
    if (device_id >= 0 && device_id < static_cast<int>(devices_.size())) {
        current_device_ = device_id;
    }
}

} // namespace npu