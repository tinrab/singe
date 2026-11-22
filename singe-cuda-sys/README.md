# singe-cuda-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda-sys.svg)](https://crates.io/crates/singe-cuda-sys)
[![Documentation](https://docs.rs/singe-cuda-sys/badge.svg)](https://docs.rs/singe-cuda-sys)
![License](https://img.shields.io/crates/l/singe-cuda-sys.svg)

Low-level FFI bindings for the CUDA driver and runtime APIs, NVRTC, nvJitLink, NVVM, NVTX 3, and the shared CUDA library types, generated with bindgen from their headers.
Each API has its own module: `driver`, `runtime`, `nvrtc`, `nvjitlink`, `nvvm`, `nvtx`, `profiler`, and `library_types`.
Every library loads at run time through [`singe-loader`](../singe-loader/README.md), on the first call of one of its functions or up front with `load_driver` and the other `load_*` functions, so nothing links at build time.
Most users want the safe crate, [`singe-cuda`](../singe-cuda/README.md).
