# singe-curand-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-curand-sys.svg)](https://crates.io/crates/singe-curand-sys)
[![Documentation](https://docs.rs/singe-curand-sys/badge.svg)](https://docs.rs/singe-curand-sys)
![License](https://img.shields.io/crates/l/singe-curand-sys.svg)

Low-level FFI bindings for NVIDIA cuRAND, the random number generation library.

Most users want the safe crate, [`singe-curand`](../singe-curand/README.md).

## Bindings

| Binds               | Header     | Module     | Feature       |
| ------------------- | ---------- | ---------- | ------------- |
| The cuRAND host API | `curand.h` | crate root | `curand_10_4` |

The `curand_10_4` feature is on by default.
The bindings are generated with bindgen from the cuRAND 10.4 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
