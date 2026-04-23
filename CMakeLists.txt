cmake_minimum_required(VERSION 3.24)

project(NeuralResampling LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(nrc_overfit_test
    src/nrc_overfit_test.cu
    openpgl/openpgl/directional/neural/NeuralRadianceCacheKernels.cu
)

target_compile_features(nrc_overfit_test PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(nrc_overfit_test PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_EXTENSIONS OFF
    CXX_EXTENSIONS OFF
)

target_include_directories(nrc_overfit_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/openpgl
    ${CMAKE_CURRENT_SOURCE_DIR}/openpgl/openpgl
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/openpgl/third-party/tiny-cuda-nn/include
    ${CMAKE_CURRENT_SOURCE_DIR}/openpgl/third-party/tiny-cuda-nn/dependencies
    ${CMAKE_CURRENT_SOURCE_DIR}/openpgl/third-party/tiny-cuda-nn/dependencies/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/openpgl/third-party/tiny-cuda-nn/dependencies/cutlass/tools/util/include
)

target_compile_definitions(nrc_overfit_test PRIVATE
    TCNN_MIN_GPU_ARCH=75
    TCNN_HALF_PRECISION=1
)

target_link_libraries(nrc_overfit_test PRIVATE CUDA::cudart)
