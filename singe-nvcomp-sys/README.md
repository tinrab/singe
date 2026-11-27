# singe-nvcomp-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvcomp-sys.svg)](https://crates.io/crates/singe-nvcomp-sys)
[![Documentation](https://docs.rs/singe-nvcomp-sys/badge.svg)](https://docs.rs/singe-nvcomp-sys)
![License](https://img.shields.io/crates/l/singe-nvcomp-sys.svg)

Low-level FFI bindings for NVIDIA nvCOMP, the GPU compression library, and its C++ manager API.

Most users want the safe crate, [`singe-nvcomp`](../singe-nvcomp/README.md).

## Bindings

| Binds                                                           | Header                        | Module     | Feature      |
| --------------------------------------------------------------- | ----------------------------- | ---------- | ------------ |
| The nvCOMP C API: batched compression for each codec, and CRC32 | `nvcomp.h`, `nvcomp/*.h`      | crate root | `nvcomp_6_0` |
| A C++ bridge to nvCOMP's managers and streaming API             | `cpp/bridge.h`, in this crate | `manager`  | `manager`    |

The `nvcomp_6_0` feature is on by default, and `manager` is off.
The bindings are generated with bindgen from the nvCOMP 6.0 headers and this crate's bridge header.

## Loading

Nothing links at build time unless the `manager` feature is on; its C++ bridge needs the nvCOMP and CUDA headers to build, and links nvCOMP.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
