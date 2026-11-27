# singe-nvjpeg-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvjpeg-sys.svg)](https://crates.io/crates/singe-nvjpeg-sys)
[![Documentation](https://docs.rs/singe-nvjpeg-sys/badge.svg)](https://docs.rs/singe-nvjpeg-sys)
![License](https://img.shields.io/crates/l/singe-nvjpeg-sys.svg)

Low-level FFI bindings for NVIDIA nvJPEG and nvJPEG2000.

Most users want the safe crate, [`singe-nvjpeg`](../singe-nvjpeg/README.md).

## Bindings

| Binds                                       | Header       | Module     | Feature       |
| ------------------------------------------- | ------------ | ---------- | ------------- |
| nvJPEG, JPEG decoding and encoding          | `nvjpeg.h`   | crate root | `nvjpeg_13_2` |
| nvJPEG2000, JPEG 2000 decoding and encoding | `nvjpeg2k.h` | `k2`       | `k2`          |

The `nvjpeg_13_2` feature is on by default, and `k2` is off.
The bindings are generated with bindgen from the nvJPEG 13.2 and nvJPEG2000 0.11 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads each library on the first call of one of its functions, or up front with `load` for nvJPEG and `load_k2` for nvJPEG2000.
When a library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
