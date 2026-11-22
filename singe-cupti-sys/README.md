# singe-cupti-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cupti-sys.svg)](https://crates.io/crates/singe-cupti-sys)
[![Documentation](https://docs.rs/singe-cupti-sys/badge.svg)](https://docs.rs/singe-cupti-sys)
![License](https://img.shields.io/crates/l/singe-cupti-sys.svg)

Low-level FFI bindings for CUPTI, the CUDA Profiling Tools Interface, generated from `cupti.h` and the profiler headers.
`singe-loader` loads `libcupti` from the CUDA Toolkit at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-cupti`](../singe-cupti/README.md) crate.
