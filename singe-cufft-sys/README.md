# singe-cufft-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufft-sys.svg)](https://crates.io/crates/singe-cufft-sys)
[![Documentation](https://docs.rs/singe-cufft-sys/badge.svg)](https://docs.rs/singe-cufft-sys)
![License](https://img.shields.io/crates/l/singe-cufft-sys.svg)

Low-level FFI bindings for cuFFT 12.4 (`cufft.h`).
The `xt` module binds the extended API from `cufftXt.h`, and the `device` module binds the device-code generation API from `cufft_device.h`.
The library loads at run time through [`singe-loader`](../singe-loader/README.md), on the first call of one of its functions or up front with `load`, so building links no NVIDIA library.
If the library or one of its symbols is missing, a function that returns a status returns a failure status, and `take_absence` reports what was missing.
Most users want the safe [`singe-cufft`](../singe-cufft/README.md) crate.
