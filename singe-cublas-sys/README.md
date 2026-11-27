# singe-cublas-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cublas-sys.svg)](https://crates.io/crates/singe-cublas-sys)
[![Documentation](https://docs.rs/singe-cublas-sys/badge.svg)](https://docs.rs/singe-cublas-sys)
![License](https://img.shields.io/crates/l/singe-cublas-sys.svg)

Low-level FFI bindings for NVIDIA cuBLAS, cuBLASLt, and cuBLASXt.

Most users want the safe crate, [`singe-cublas`](../singe-cublas/README.md).

## Bindings

| Binds                                                                            | Header        | Module     | Feature       |
| -------------------------------------------------------------------------------- | ------------- | ---------- | ------------- |
| cuBLAS, the BLAS routines and their extensions                                   | `cublas_v2.h` | crate root | `cublas_13_8` |
| cuBLASLt, for matrix multiplication with fused epilogues and algorithm selection | `cublasLt.h`  | crate root | `lt`          |
| cuBLASXt, for BLAS level 3 routines split across several GPUs                    | `cublasXt.h`  | crate root | `xt`          |

The `cublas_13_8` feature is on by default, and the others are off.
The bindings are generated with bindgen from the cuBLAS 13.8 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads each library on the first call of one of its functions, or up front with `load` for cuBLAS and cuBLASXt and `load_lt` for cuBLASLt.
When a library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
