<div align="center">
🚧 WORK IN PROGRESS 🚧
</div>

# Singe

A machine learning framework that sets tensors ablaze.

## Ecosystem

- [singe-cublas](./singe-cublas): Safe Rust wrappers for NVIDIA cuBLAS dense linear algebra on the GPU: the classic BLAS API, cuBLASLt for configurable matrix multiplication, and cuBLASXt for BLAS3 across several GPUs.
- [singe-cuda](./singe-cuda): Safe Rust wrappers for the CUDA driver and runtime APIs, including runtime compilation of CUDA C++ and NVTX annotations.
- [singe-cuda-macros](./singe-cuda-macros): Derive macros for types that can be copied to the GPU, and `cuda_module!` for typed CUDA kernel launches.
- [singe-cuda-find](./singe-cuda-find): Discovery tools for the CUDA Toolkit and NVIDIA libraries.
- [singe-cufile](./singe-cufile): Safe Rust wrappers for the NVIDIA cuFile library, which moves data directly between files and GPU memory with GPUDirect Storage.
- [singe-curand](./singe-curand): Safe Rust wrappers for the NVIDIA cuRAND host API, for random number generation on the GPU or CPU.
- [singe-cudnn](./singe-cudnn): Safe Rust wrappers for NVIDIA cuDNN, the deep neural network library, including a Rust port of NVIDIA's cuDNN Frontend.
- [singe-cudss](./singe-cudss): Safe Rust wrappers for NVIDIA cuDSS, the CUDA direct sparse solver library.
- [singe-cufft](./singe-cufft): Safe Rust wrappers for NVIDIA cuFFT, for Fourier transforms on the GPU.
- [singe-cusolver](./singe-cusolver): Safe Rust wrappers for NVIDIA cuSOLVER, for dense matrix factorizations and solvers on the GPU.
- [singe-cupti](./singe-cupti): Safe Rust wrappers for NVIDIA CUPTI, for tracing and profiling CUDA applications.
- [singe-cusparse](./singe-cusparse): Safe Rust wrappers for NVIDIA cuSPARSE, for sparse linear algebra on the GPU.
- [singe-cutensor](./singe-cutensor): Safe Rust wrappers for NVIDIA cuTENSOR, for tensor contractions and other tensor operations on the GPU.
- [singe-kernel](./singe-kernel): Reusable CPU and GPU kernels.
- [singe-model](./singe-model): Model definitions built on `singe`.
- [singe-nvcomp](./singe-nvcomp): Safe Rust wrappers for the NVIDIA nvCOMP compression library.
- [singe-nvimagecodec](./singe-nvimagecodec): Safe Rust wrappers for the NVIDIA nvImageCodec library.
- [singe-nvjpeg](./singe-nvjpeg): Safe Rust wrappers for the NVIDIA nvJPEG and nvJPEG 2000 libraries.
- [singe-nvml](./singe-nvml): Safe Rust wrappers for the NVIDIA Management Library (NVML), for monitoring and managing GPUs.
- [singe-nvtiff](./singe-nvtiff): Safe Rust wrappers for the NVIDIA nvTIFF library.
- [singe-nccl](./singe-nccl): Safe Rust wrappers for NCCL, the NVIDIA Collective Communications Library, for exchanging data between GPUs.
- [singe-npp](./singe-npp): Safe Rust wrappers for NVIDIA Performance Primitives (NPP), for image and signal processing on the GPU.
- [singe-onnx](./singe-onnx): ONNX model utilities.
- [singe-ptx](./singe-ptx): A PTX parser and a CPU emulator that runs PTX kernels without an NVIDIA GPU.
