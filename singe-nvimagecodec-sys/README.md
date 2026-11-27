# singe-nvimagecodec-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvimagecodec-sys.svg)](https://crates.io/crates/singe-nvimagecodec-sys)
[![Documentation](https://docs.rs/singe-nvimagecodec-sys/badge.svg)](https://docs.rs/singe-nvimagecodec-sys)
![License](https://img.shields.io/crates/l/singe-nvimagecodec-sys.svg)

Low-level FFI bindings for NVIDIA nvImageCodec, the image codec library.

Most users want the safe crate, [`singe-nvimagecodec`](../singe-nvimagecodec/README.md).

## Bindings

| Binds                                                        | Header         | Module     | Feature            |
| ------------------------------------------------------------ | -------------- | ---------- | ------------------ |
| nvImageCodec decoding, encoding, and its codec extension API | `nvimgcodec.h` | crate root | `nvimagecodec_0_9` |

The `nvimagecodec_0_9` feature is on by default.
The bindings are generated with bindgen from the nvImageCodec 0.9 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
