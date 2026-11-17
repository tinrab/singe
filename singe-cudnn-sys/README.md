# singe-cudnn-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cudnn-sys.svg)](https://crates.io/crates/singe-cudnn-sys)
[![Documentation](https://docs.rs/singe-cudnn-sys/badge.svg)](https://docs.rs/singe-cudnn-sys)
![License](https://img.shields.io/crates/l/singe-cudnn-sys.svg)

Low-level FFI bindings for cuDNN 9.27 (`cudnn.h`).
The library loads at run time through [`singe-loader`](../singe-loader/README.md), on the first call of one of its functions or up front with `load`, so building links no NVIDIA library.
If the library or one of its symbols is missing, a function that returns a status returns a failure status, and `take_absence` reports what was missing.
Most users want the safe [`singe-cudnn`](../singe-cudnn/README.md) crate.
