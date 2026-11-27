# singe-cusolver-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cusolver-sys.svg)](https://crates.io/crates/singe-cusolver-sys)
[![Documentation](https://docs.rs/singe-cusolver-sys/badge.svg)](https://docs.rs/singe-cusolver-sys)
![License](https://img.shields.io/crates/l/singe-cusolver-sys.svg)

Low-level FFI bindings for NVIDIA cuSOLVERDn, the dense cuSOLVER API.

Most users want the safe crate, [`singe-cusolver`](../singe-cusolver/README.md).

## Bindings

| Binds                                                                                      | Header         | Module     | Feature         |
| ------------------------------------------------------------------------------------------ | -------------- | ---------- | --------------- |
| cuSOLVERDn, dense factorizations, solves, and eigenvalue and singular value decompositions | `cusolverDn.h` | crate root | `cusolver_12_3` |

The `cusolver_12_3` feature is on by default.
The bindings are generated with bindgen from the cuSOLVER 12.3 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
