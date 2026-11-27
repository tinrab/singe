# singe-nvml-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvml-sys.svg)](https://crates.io/crates/singe-nvml-sys)
[![Documentation](https://docs.rs/singe-nvml-sys/badge.svg)](https://docs.rs/singe-nvml-sys)
![License](https://img.shields.io/crates/l/singe-nvml-sys.svg)

Low-level FFI bindings for NVML, the NVIDIA Management Library.

Most users want the safe crate, [`singe-nvml`](../singe-nvml/README.md).

## Bindings

| Binds                               | Header   | Module     | Feature   |
| ----------------------------------- | -------- | ---------- | --------- |
| NVML, GPU monitoring and management | `nvml.h` | crate root | `nvml_13` |

The `nvml_13` feature is on by default.
The bindings are generated with bindgen from the NVML 13 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
NVML ships with the NVIDIA driver, not the CUDA Toolkit.
