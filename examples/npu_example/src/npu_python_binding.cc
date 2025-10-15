#include <cstdint>
#include <memory>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "npu_simulator.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// NPU MatMul FFI 实现
ffi::Error NpuMatMulImpl(
    ffi::Buffer<ffi::F32> input_a,
    ffi::Buffer<ffi::F32> input_b,
    ffi::ResultBuffer<ffi::F32> output_c,
    int64_t device_id
) {
    // 获取矩阵维度
    auto a_dims = input_a.dimensions();
    auto b_dims = input_b.dimensions();
    auto c_dims = output_c->dimensions();
    
    if (a_dims.size() != 2 || b_dims.size() != 2 || c_dims.size() != 2) {
        return ffi::Error::InvalidArgument("All inputs must be 2D matrices");
    }
    
    int a_rows = a_dims[0], a_cols = a_dims[1];
    int b_rows = b_dims[0], b_cols = b_dims[1];
    int c_rows = c_dims[0], c_cols = c_dims[1];
    
    if (a_cols != b_rows || a_rows != c_rows || b_cols != c_cols) {
        return ffi::Error::InvalidArgument("Matrix dimensions mismatch");
    }
    
    try {
        // 执行NPU矩阵乘法
        npu::NPUKernel::MatMul(
            input_a.typed_data(),
            a_rows, a_cols,
            input_b.typed_data(),
            b_rows, b_cols,
            output_c->typed_data(),
            static_cast<int>(device_id)
        );
        return ffi::Error::Success();
    } catch (const std::exception& e) {
        return ffi::Error::Internal(std::string("NPU MatMul failed: ") + e.what());
    }
}

// NPU Optimized MatMul FFI 实现  
ffi::Error NpuOptimizedMatMulImpl(
    ffi::Buffer<ffi::F32> input_a,
    ffi::Buffer<ffi::F32> input_b,
    ffi::ResultBuffer<ffi::F32> output_c,
    int64_t device_id
) {
    auto a_dims = input_a.dimensions();
    auto b_dims = input_b.dimensions();
    auto c_dims = output_c->dimensions();
    
    if (a_dims.size() != 2 || b_dims.size() != 2 || c_dims.size() != 2) {
        return ffi::Error::InvalidArgument("All inputs must be 2D matrices");
    }
    
    int a_rows = a_dims[0], a_cols = a_dims[1];
    int b_rows = b_dims[0], b_cols = b_dims[1];
    
    try {
        npu::NPUKernel::OptimizedMatMul(
            input_a.typed_data(),
            a_rows, a_cols,
            input_b.typed_data(),
            b_rows, b_cols,
            output_c->typed_data(),
            static_cast<int>(device_id)
        );
        return ffi::Error::Success();
    } catch (const std::exception& e) {
        return ffi::Error::Internal(std::string("NPU Optimized MatMul failed: ") + e.what());
    }
}

// NPU Batch MatMul FFI 实现
ffi::Error NpuBatchMatMulImpl(
    ffi::Buffer<ffi::F32> input_a,
    ffi::Buffer<ffi::F32> input_b, 
    ffi::ResultBuffer<ffi::F32> output_c,
    int64_t device_id
) {
    auto a_dims = input_a.dimensions();
    auto b_dims = input_b.dimensions();
    auto c_dims = output_c->dimensions();
    
    if (a_dims.size() != 3 || b_dims.size() != 3 || c_dims.size() != 3) {
        return ffi::Error::InvalidArgument("All inputs must be 3D tensors for batch matmul");
    }
    
    int batch_size = a_dims[0];
    int m = a_dims[1], k = a_dims[2];
    int n = b_dims[2];
    
    if (a_dims[0] != b_dims[0] || a_dims[2] != b_dims[1]) {
        return ffi::Error::InvalidArgument("Batch matmul dimension mismatch");
    }
    
    try {
        npu::NPUKernel::BatchMatMul(
            input_a.typed_data(),
            input_b.typed_data(),
            output_c->typed_data(),
            batch_size, m, k, n,
            static_cast<int>(device_id)
        );
        return ffi::Error::Success();
    } catch (const std::exception& e) {
        return ffi::Error::Internal(std::string("NPU Batch MatMul failed: ") + e.what());
    }
}

// 注册FFI处理函数
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NpuMatMul, NpuMatMulImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()    // input_a
        .Arg<ffi::Buffer<ffi::F32>>()    // input_b
        .Ret<ffi::Buffer<ffi::F32>>()    // output_c
        .Attr<int64_t>("device_id")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NpuOptimizedMatMul, NpuOptimizedMatMulImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("device_id")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NpuBatchMatMul, NpuBatchMatMulImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("device_id")
);

// Python模块绑定
NB_MODULE(_npu_simulator, m) {
    m.doc() = "NPU Simulator for JAX";
    
    // 导出FFI函数注册表
    m.def("registrations", []() {
        nb::dict registrations;
        registrations["npu_matmul"] = nb::capsule(reinterpret_cast<void*>(NpuMatMul));
        registrations["npu_optimized_matmul"] = nb::capsule(reinterpret_cast<void*>(NpuOptimizedMatMul));
        registrations["npu_batch_matmul"] = nb::capsule(reinterpret_cast<void*>(NpuBatchMatMul));
        return registrations;
    });
    
    // NPU运行时控制
    m.def("initialize_npu_runtime", [](int num_devices) {
        npu::NPURuntime::GetInstance().Initialize(num_devices);
    }, "Initialize NPU runtime with specified number of devices");
    
    m.def("get_npu_device_count", []() {
        return npu::NPURuntime::GetInstance().GetDeviceCount();
    });
    
    m.def("set_current_npu_device", [](int device_id) {
        npu::NPURuntime::GetInstance().SetCurrentDevice(device_id);
    });
    
    m.def("get_current_npu_device", []() {
        return npu::NPURuntime::GetInstance().GetCurrentDevice();
    });
}