# singe-cudnn-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cudnn-sys.svg)](https://crates.io/crates/singe-cudnn-sys)
[![Documentation](https://docs.rs/singe-cudnn-sys/badge.svg)](https://docs.rs/singe-cudnn-sys)
![License](https://img.shields.io/crates/l/singe-cudnn-sys.svg)

Low-level FFI bindings for NVIDIA cuDNN, the deep neural network library.

Most users want the safe crate, [`singe-cudnn`](../singe-cudnn/README.md).

## Bindings

| Binds                                      | Header    | Module     | Feature      |
| ------------------------------------------ | --------- | ---------- | ------------ |
| cuDNN, its graph API and legacy operations | `cudnn.h` | crate root | `cudnn_9_27` |

The `cudnn_9_27` feature is on by default.
The bindings are generated with bindgen from the cuDNN 9.27 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
cuDNN loads its own sub-libraries.
