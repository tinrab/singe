singe-cublas
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-cublas.svg)](https://crates.io/crates/singe-cublas)
[![Documentation](https://docs.rs/singe-cublas/badge.svg)](https://docs.rs/singe-cublas)
![License](https://img.shields.io/crates/l/singe-cublas.svg)

Safe cuBLAS and cuBLASLt wrappers for dense GPU linear algebra.

This crate provides BLAS level routines, cuBLAS context management, cuBLASLt matmul descriptors, algorithm selection, layouts, preferences, epilogues, and execution helpers over the raw `singe-cublas-sys` bindings.

## Examples

Runnable examples are available under [`examples/cublas`](examples/cublas/) and [`examples/lt`](examples/lt/).
