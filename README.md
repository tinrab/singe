<p align="center">
  <img height="200" src="./assets/logo-text.svg" alt="Singe">
</p>

<p align="center">
    A machine learning framework that sets tensors ablaze.
</p>

<div align="center">
🚧 WORK IN PROGRESS 🚧
</div>

## CUDA

- [singe-cuda](./singe-cuda): the CUDA driver and runtime APIs, including runtime compilation of CUDA C++ and NVTX annotations.
- [singe-cuda-macros](./singe-cuda-macros): derive macros for types that can be copied to the GPU, and `cuda_module!` for type-safe kernel launches.
- [singe-cuda-find](./singe-cuda-find): CUDA Toolkit and NVIDIA libraries path finder for build scripts.

## NVIDIA libraries

Raw bindings and safe Rust wrapper crates for NVIDIA libraries.

- [singe-cublas](./singe-cublas): cuBLAS dense linear algebra on the GPU: the classic BLAS API, cuBLASLt for configurable matrix multiplication, and cuBLASXt for BLAS3 across several GPUs.
- [singe-cudnn](./singe-cudnn): cuDNN, the deep neural network library, including a Rust port of NVIDIA's cuDNN Frontend.
- [singe-cudss](./singe-cudss): cuDSS, the CUDA direct sparse solver library.
- [singe-cufft](./singe-cufft): cuFFT, for Fourier transforms on the GPU.
- [singe-cufile](./singe-cufile): the NVIDIA cuFile, which moves data directly between files and GPU memory with GPUDirect Storage.
- [singe-cupti](./singe-cupti): CUPTI, for tracing and profiling CUDA applications.
- [singe-curand](./singe-curand): cuRAND, for random number generation.
- [singe-cusolver](./singe-cusolver): cuSOLVER, for matrix factorizations, linear solves, eigenvalue problems, and singular value decompositions on the GPU.
- [singe-cusparse](./singe-cusparse): cuSPARSE, for sparse linear algebra on the GPU.
- [singe-cutensor](./singe-cutensor): cuTENSOR, for tensor contractions, reductions, permutations, and elementwise operations on the GPU.
- [singe-nccl](./singe-nccl): NCCL, the NVIDIA Collective Communications Library, for exchanging data between GPUs.
- [singe-npp](./singe-npp): NVIDIA Performance Primitives (NPP), for image and signal processing on the GPU.
- [singe-nvcomp](./singe-nvcomp): nvCOMP, for compressing and decompressing data on the GPU.
- [singe-nvimagecodec](./singe-nvimagecodec): nvImageCodec, for decoding and encoding images on the GPU.
- [singe-nvjpeg](./singe-nvjpeg): the nvJPEG and nvJPEG 2000 libraries.
- [singe-nvml](./singe-nvml): the NVIDIA Management Library (NVML), for monitoring and managing GPUs.
- [singe-nvtiff](./singe-nvtiff): nvTIFF, for decoding and encoding TIFF images on the GPU.

## Tools

- [singe-cli](./singe-cli): the `singe` command line tool.
- [singe-onnx](./singe-onnx): ONNX model format utilities.
- [singe-ptx](./singe-ptx): PTX parser and emulator.
- [singe-cuda-emulator](./singe-cuda-emulator): Emulated CUDA and NCCL on the CPU.
