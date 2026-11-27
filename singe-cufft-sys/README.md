# singe-cufft-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufft-sys.svg)](https://crates.io/crates/singe-cufft-sys)
[![Documentation](https://docs.rs/singe-cufft-sys/badge.svg)](https://docs.rs/singe-cufft-sys)
![License](https://img.shields.io/crates/l/singe-cufft-sys.svg)

Low-level FFI bindings for NVIDIA cuFFT, cuFFTXt, and the cuFFT device API.

Most users want the safe crate, [`singe-cufft`](../singe-cufft/README.md).

## Bindings

| Binds                                                                            | Header           | Module     | Feature      |
| -------------------------------------------------------------------------------- | ---------------- | ---------- | ------------ |
| cuFFT, fast Fourier transforms                                                   | `cufft.h`        | crate root | `cufft_12_4` |
| cuFFTXt, for CUDA data types, callbacks, and transforms across several GPUs      | `cufftXt.h`      | `xt`       | `cufft_12_4` |
| The experimental cuFFT device API, which generates FFT code for your own kernels | `cufft_device.h` | `device`   | `cufft_12_4` |

The `cufft_12_4` feature is on by default.
The bindings are generated with bindgen from the cuFFT 12.4 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
