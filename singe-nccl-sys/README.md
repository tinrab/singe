# singe-nccl-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nccl-sys.svg)](https://crates.io/crates/singe-nccl-sys)
[![Documentation](https://docs.rs/singe-nccl-sys/badge.svg)](https://docs.rs/singe-nccl-sys)
![License](https://img.shields.io/crates/l/singe-nccl-sys.svg)

Low-level FFI bindings for NCCL 2.32, the NVIDIA Collective Communications Library, generated from `nccl.h`.
`singe-loader` loads the library at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-nccl`](../singe-nccl/README.md) crate.
