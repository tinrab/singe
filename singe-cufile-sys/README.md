# singe-cufile-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufile-sys.svg)](https://crates.io/crates/singe-cufile-sys)
[![Documentation](https://docs.rs/singe-cufile-sys/badge.svg)](https://docs.rs/singe-cufile-sys)
![License](https://img.shields.io/crates/l/singe-cufile-sys.svg)

Low-level FFI bindings for cuFile, the NVIDIA `GPUDirect` Storage library, generated from `cufile.h`.
`singe-loader` loads `libcufile.so.0` at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-cufile`](../singe-cufile/README.md) crate.
