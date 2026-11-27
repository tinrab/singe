# singe-nvtiff-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvtiff-sys.svg)](https://crates.io/crates/singe-nvtiff-sys)
[![Documentation](https://docs.rs/singe-nvtiff-sys/badge.svg)](https://docs.rs/singe-nvtiff-sys)
![License](https://img.shields.io/crates/l/singe-nvtiff-sys.svg)

Low-level FFI bindings for NVIDIA nvTIFF, the TIFF codec library.

Most users want the safe crate, [`singe-nvtiff`](../singe-nvtiff/README.md).

## Bindings

| Binds                              | Header     | Module     | Feature      |
| ---------------------------------- | ---------- | ---------- | ------------ |
| nvTIFF, TIFF decoding and encoding | `nvtiff.h` | crate root | `nvtiff_0_8` |

The `nvtiff_0_8` feature is on by default.
The bindings are generated with bindgen from the nvTIFF 0.8 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
