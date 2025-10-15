cmake_minimum_required(VERSION 3.15...3.30)
project(npu_simulator LANGUAGES CXX)

option(NPU_EXAMPLE_ENABLE_CUDA "Enable CUDA acceleration for NPU simulation" OFF)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)

# NPU Simulator Library
add_library(npu_simulator SHARED "src/npu_simulator.cc")
set_target_properties(npu_simulator PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17)
target_include_directories(npu_simulator PUBLIC ${XLA_DIR})

# Python binding
nanobind_add_module(_npu_simulator NB_STATIC "src/npu_python_binding.cc")
target_include_directories(_npu_simulator PUBLIC ${XLA_DIR})
target_link_libraries(_npu_simulator PRIVATE npu_simulator)
install(TARGETS _npu_simulator LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

if(NPU_EXAMPLE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(npu_simulator PRIVATE NPU_CUDA_ACCELERATION)
  target_link_libraries(npu_simulator PRIVATE CUDA::cudart)
endif()