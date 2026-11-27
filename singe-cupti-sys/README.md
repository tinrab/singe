# singe-cupti-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cupti-sys.svg)](https://crates.io/crates/singe-cupti-sys)
[![Documentation](https://docs.rs/singe-cupti-sys/badge.svg)](https://docs.rs/singe-cupti-sys)
![License](https://img.shields.io/crates/l/singe-cupti-sys.svg)

Low-level FFI bindings for NVIDIA CUPTI, the CUDA Profiling Tools Interface.

Most users want the safe crate, [`singe-cupti`](../singe-cupti/README.md).

## Bindings

| Binds                                      | Header                                                                       | Module     | Feature      |
| ------------------------------------------ | ---------------------------------------------------------------------------- | ---------- | ------------ |
| CUPTI callbacks and activity tracing       | `cupti.h`, `cupti_target.h`                                                  | crate root | `cupti_13_4` |
| CUPTI hardware metrics and range profiling | `cupti_profiler_host.h`, `cupti_profiler_target.h`, `cupti_range_profiler.h` | crate root | `cupti_13_4` |

The `cupti_13_4` feature is on by default.
The bindings are generated with bindgen from the CUPTI 13.4 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
