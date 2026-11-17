# singe-nvtiff-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvtiff-sys.svg)](https://crates.io/crates/singe-nvtiff-sys)
[![Documentation](https://docs.rs/singe-nvtiff-sys/badge.svg)](https://docs.rs/singe-nvtiff-sys)
![License](https://img.shields.io/crates/l/singe-nvtiff-sys.svg)

Low-level FFI bindings for nvTIFF 0.8, the NVIDIA GPU TIFF codec library, generated from `nvtiff.h`.
`singe-loader` loads the library at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-nvtiff`](../singe-nvtiff/README.md) crate.
