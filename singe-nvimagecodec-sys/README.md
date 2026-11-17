# singe-nvimagecodec-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvimagecodec-sys.svg)](https://crates.io/crates/singe-nvimagecodec-sys)
[![Documentation](https://docs.rs/singe-nvimagecodec-sys/badge.svg)](https://docs.rs/singe-nvimagecodec-sys)
![License](https://img.shields.io/crates/l/singe-nvimagecodec-sys.svg)

Low-level FFI bindings for nvImageCodec 0.9, the NVIDIA image codec library, generated from `nvimgcodec.h`.
`singe-loader` loads the library at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-nvimagecodec`](../singe-nvimagecodec/README.md) crate.
