# Singe

A machine learning framework that sets tensors ablaze.

<div align="center">
🚧 WORK IN PROGRESS 🚧
</div>

## CUDA

- [singe-cuda](./singe-cuda): the CUDA driver and runtime APIs, including runtime compilation of CUDA C++ and NVTX annotations.
- [singe-cuda-macros](./singe-cuda-macros): derive macros for types that can be copied to the GPU, and `cuda_module!` for typed kernel launches.
- [singe-cuda-find](./singe-cuda-find): finds the CUDA Toolkit and NVIDIA libraries for build scripts.

## NVIDIA libraries

Each wrapper crate has a `-sys` crate of raw bindings beside it, such as [singe-cuda-sys](./singe-cuda-sys).

- [singe-cublas](./singe-cublas): cuBLAS dense linear algebra, cuBLASLt matrix multiplication, and cuBLASXt BLAS3 across several GPUs.
- [singe-cudnn](./singe-cudnn): cuDNN deep neural network primitives, including a Rust port of NVIDIA's cuDNN Frontend.
- [singe-cudss](./singe-cudss): cuDSS direct sparse solvers.
- [singe-cufft](./singe-cufft): cuFFT Fourier transforms.
- [singe-cufile](./singe-cufile): cuFile, which moves data between files and GPU memory with GPUDirect Storage.
- [singe-cupti](./singe-cupti): CUPTI tracing and profiling.
- [singe-curand](./singe-curand): cuRAND random number generation on the GPU or CPU.
- [singe-cusolver](./singe-cusolver): cuSOLVER dense matrix factorizations and solvers.
- [singe-cusparse](./singe-cusparse): cuSPARSE sparse linear algebra.
- [singe-cutensor](./singe-cutensor): cuTENSOR tensor contractions, reductions, and elementwise operations.
- [singe-nccl](./singe-nccl): NCCL collective communication between GPUs.
- [singe-npp](./singe-npp): NPP image and signal processing.
- [singe-nvcomp](./singe-nvcomp): nvCOMP compression and decompression on the GPU.
- [singe-nvimagecodec](./singe-nvimagecodec): nvImageCodec image decoding and encoding.
- [singe-nvjpeg](./singe-nvjpeg): nvJPEG and nvJPEG 2000 image decoding and encoding.
- [singe-nvml](./singe-nvml): NVML GPU monitoring and management.
- [singe-nvtiff](./singe-nvtiff): nvTIFF TIFF decoding and encoding.

## Other tools

- [singe-onnx](./singe-onnx): decodes ONNX models from their protobuf encoding into typed Rust values.
- [singe-ptx](./singe-ptx): a PTX parser, and a CPU emulator that runs PTX kernels without an NVIDIA GPU.
