# singe-cufile-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufile-sys.svg)](https://crates.io/crates/singe-cufile-sys)
[![Documentation](https://docs.rs/singe-cufile-sys/badge.svg)](https://docs.rs/singe-cufile-sys)
![License](https://img.shields.io/crates/l/singe-cufile-sys.svg)

Low-level FFI bindings for NVIDIA cuFile, the GPUDirect Storage library.

Most users want the safe crate, [`singe-cufile`](../singe-cufile/README.md).

## Bindings

| Binds                                   | Header     | Module     | Feature       |
| --------------------------------------- | ---------- | ---------- | ------------- |
| cuFile, file I/O to and from GPU memory | `cufile.h` | crate root | `cufile_1_19` |

The `cufile_1_19` feature is on by default.
The bindings are generated with bindgen from the cuFile 1.19 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
cuFile has no Windows release.
