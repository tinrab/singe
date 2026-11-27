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

- [singe-cuda](./singe-cuda): the CUDA driver and runtime APIs, NVRTC, NVVM, nvJitLink, NVTX, and the CUDA profiler.
- [singe-cuda-macros](./singe-cuda-macros): derive macros for types that can be copied to the GPU, and `cuda_module!` for type-safe kernel launches.
- [singe-cuda-find](./singe-cuda-find): CUDA Toolkit and NVIDIA libraries path finder for build scripts.

## NVIDIA libraries

Raw bindings and safe Rust wrapper crates for NVIDIA libraries.

- [singe-cublas](./singe-cublas): cuBLAS, cuBLASLt, and cuBLASXt: BLAS, configurable matrix multiplication, and BLAS level 3 across several GPUs.
- [singe-cudnn](./singe-cudnn): cuDNN deep neural network primitives, with a Rust port of the cuDNN Frontend graph API.
- [singe-cudss](./singe-cudss): cuDSS, for solving sparse linear systems with direct methods on one or several GPUs.
- [singe-cufft](./singe-cufft): cuFFT, cuFFTXt, and the cuFFT device API, for fast Fourier transforms on one or several GPUs.
- [singe-cufile](./singe-cufile): cuFile, for reading and writing files directly to GPU memory with GPUDirect Storage.
- [singe-cupti](./singe-cupti): CUPTI, the CUDA Profiling Tools Interface, for tracing and profiling CUDA applications.
- [singe-curand](./singe-curand): cuRAND, for generating random numbers on the GPU.
- [singe-cusolver](./singe-cusolver): cuSOLVER, for matrix factorizations, linear solves, eigenvalue problems, and singular value decompositions on the GPU.
- [singe-cusparse](./singe-cusparse): cuSPARSE, for sparse matrix products, format conversions, and triangular solves on the GPU.
- [singe-cutensor](./singe-cutensor): cuTENSOR, cuTENSORMg, and cuTENSORMp, for tensor contractions, reductions, permutations, and elementwise operations on one or several GPUs.
- [singe-nccl](./singe-nccl): NCCL, the NVIDIA Collective Communications Library, for exchanging data between GPUs.
- [singe-npp](./singe-npp): NPP, the NVIDIA Performance Primitives, for image and signal processing on the GPU.
- [singe-nvcomp](./singe-nvcomp): nvCOMP, for compressing and decompressing data on the GPU with LZ4, Snappy, Zstd, Deflate, Gzip, GDeflate, ANS, Cascaded, and Bitcomp.
- [singe-nvimagecodec](./singe-nvimagecodec): nvImageCodec, for decoding and encoding JPEG, JPEG 2000, TIFF, PNG, BMP, and other image formats on the GPU or CPU.
- [singe-nvjpeg](./singe-nvjpeg): nvJPEG and nvJPEG2000, for decoding and encoding JPEG and JPEG 2000 images on the GPU.
- [singe-nvml](./singe-nvml): NVML, the NVIDIA Management Library, for monitoring and managing GPUs.
- [singe-nvtiff](./singe-nvtiff): nvTIFF, for decoding and encoding TIFF images on the GPU.

## Tools

- [singe-cli](./singe-cli): the `singe` command line tool.
- [singe-onnx](./singe-onnx): ONNX model format utilities.
- [singe-ptx](./singe-ptx): PTX parser and emulator.
- [singe-cuda-emulator](./singe-cuda-emulator): Emulated CUDA and NCCL on the CPU.
