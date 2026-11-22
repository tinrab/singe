# singe-nvcomp-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvcomp-sys.svg)](https://crates.io/crates/singe-nvcomp-sys)
[![Documentation](https://docs.rs/singe-nvcomp-sys/badge.svg)](https://docs.rs/singe-nvcomp-sys)
![License](https://img.shields.io/crates/l/singe-nvcomp-sys.svg)

Low-level FFI bindings for the nvCOMP C API, the NVIDIA GPU compression library, generated from `nvcomp.h`.
`singe-loader` loads the library at run time, on the first call or through `load`, so building links no NVIDIA library unless the `manager` feature is on.
Most users want the safe [`singe-nvcomp`](../singe-nvcomp/README.md) crate.

## Features

- `manager`: compiles a C++ bridge to nvCOMP's manager API, which needs the nvCOMP and CUDA headers at build time and links nvCOMP.
