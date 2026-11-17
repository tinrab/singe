# singe-nvml-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvml-sys.svg)](https://crates.io/crates/singe-nvml-sys)
[![Documentation](https://docs.rs/singe-nvml-sys/badge.svg)](https://docs.rs/singe-nvml-sys)
![License](https://img.shields.io/crates/l/singe-nvml-sys.svg)

Low-level FFI bindings for NVML 13, the NVIDIA Management Library, generated from `nvml.h`.
`singe-loader` loads the NVML library that ships with the NVIDIA driver at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-nvml`](../singe-nvml/README.md) crate.
