# singe-cudss-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cudss-sys.svg)](https://crates.io/crates/singe-cudss-sys)
[![Documentation](https://docs.rs/singe-cudss-sys/badge.svg)](https://docs.rs/singe-cudss-sys)
![License](https://img.shields.io/crates/l/singe-cudss-sys.svg)

Low-level FFI bindings for NVIDIA cuDSS, the direct sparse solver library.

Most users want the safe crate, [`singe-cudss`](../singe-cudss/README.md).

## Bindings

| Binds                                           | Header    | Module     | Feature     |
| ----------------------------------------------- | --------- | ---------- | ----------- |
| cuDSS, direct solvers for sparse linear systems | `cudss.h` | crate root | `cudss_1_0` |

The `cudss_1_0` feature is on by default.
The bindings are generated with bindgen from the cuDSS 1.0 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
