# singe-nvjpeg-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvjpeg-sys.svg)](https://crates.io/crates/singe-nvjpeg-sys)
[![Documentation](https://docs.rs/singe-nvjpeg-sys/badge.svg)](https://docs.rs/singe-nvjpeg-sys)
![License](https://img.shields.io/crates/l/singe-nvjpeg-sys.svg)

Low-level FFI bindings for nvJPEG 13.2, from the CUDA Toolkit, and for nvJPEG 2000 0.11, generated from `nvjpeg.h` and `nvjpeg2k.h`.
`singe-loader` loads each library at run time, on the first call or through `load` and `load_k2`, so building links no NVIDIA library.
Most users want the safe [`singe-nvjpeg`](../singe-nvjpeg/README.md) crate.

## Features

- `k2`: adds the nvJPEG 2000 bindings in the `k2` module.
