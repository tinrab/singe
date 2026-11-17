# singe-npp-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-npp-sys.svg)](https://crates.io/crates/singe-npp-sys)
[![Documentation](https://docs.rs/singe-npp-sys/badge.svg)](https://docs.rs/singe-npp-sys)
![License](https://img.shields.io/crates/l/singe-npp-sys.svg)

Low-level FFI bindings for NPP 13.2, the NVIDIA Performance Primitives, generated from `npp.h`.
NPP is split into several libraries, such as `nppc` for core functions, `nppi*` for image processing, and `npps` for signal processing.
`singe-loader` loads each library at run time, on the first call or through `load`, so building links no NVIDIA library.
Most users want the safe [`singe-npp`](../singe-npp/README.md) crate.
