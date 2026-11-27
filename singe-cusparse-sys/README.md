# singe-cusparse-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cusparse-sys.svg)](https://crates.io/crates/singe-cusparse-sys)
[![Documentation](https://docs.rs/singe-cusparse-sys/badge.svg)](https://docs.rs/singe-cusparse-sys)
![License](https://img.shields.io/crates/l/singe-cusparse-sys.svg)

Low-level FFI bindings for NVIDIA cuSPARSE, the sparse linear algebra library.

Most users want the safe crate, [`singe-cusparse`](../singe-cusparse/README.md).

## Bindings

| Binds                                         | Header       | Module     | Feature         |
| --------------------------------------------- | ------------ | ---------- | --------------- |
| cuSPARSE, sparse matrix and vector operations | `cusparse.h` | crate root | `cusparse_12_8` |

The `cusparse_12_8` feature is on by default.
The bindings are generated with bindgen from the cuSPARSE 12.8 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
