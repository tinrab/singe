# singe-cublas-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cublas-sys.svg)](https://crates.io/crates/singe-cublas-sys)
[![Documentation](https://docs.rs/singe-cublas-sys/badge.svg)](https://docs.rs/singe-cublas-sys)
![License](https://img.shields.io/crates/l/singe-cublas-sys.svg)

Low-level FFI bindings for cuBLAS 13.8 (`cublas_v2.h`), with optional bindings for cuBLASLt (`cublasLt.h`) and cuBLASXt (`cublasXt.h`).
Each library loads at run time through [`singe-loader`](../singe-loader/README.md), on the first call of one of its functions or up front with `load` and `load_lt`, so building links no NVIDIA library.
If the library or one of its symbols is missing, a function that returns a status returns a failure status, and `take_absence` reports what was missing.
Most users want the safe [`singe-cublas`](../singe-cublas/README.md) crate.

## Features

- `lt`: cuBLASLt, for matrix multiplication with fused epilogues and algorithm selection.
- `xt`: cuBLASXt, for BLAS level 3 routines split across several GPUs.
