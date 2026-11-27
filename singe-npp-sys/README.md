# singe-npp-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-npp-sys.svg)](https://crates.io/crates/singe-npp-sys)
[![Documentation](https://docs.rs/singe-npp-sys/badge.svg)](https://docs.rs/singe-npp-sys)
![License](https://img.shields.io/crates/l/singe-npp-sys.svg)

Low-level FFI bindings for NPP, the NVIDIA Performance Primitives for image and signal processing.

Most users want the safe crate, [`singe-npp`](../singe-npp/README.md).

## Bindings

| Binds                                      | Header  | Module     | Feature    |
| ------------------------------------------ | ------- | ---------- | ---------- |
| NPP image and signal processing primitives | `npp.h` | crate root | `npp_13_2` |

The `npp_13_2` feature is on by default.
The bindings are generated with bindgen from the NPP 13.2 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
NPP's functions are spread over its core library, `nppc`, and ten libraries by domain, which load together.
