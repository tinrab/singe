# singe-nccl-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nccl-sys.svg)](https://crates.io/crates/singe-nccl-sys)
[![Documentation](https://docs.rs/singe-nccl-sys/badge.svg)](https://docs.rs/singe-nccl-sys)
![License](https://img.shields.io/crates/l/singe-nccl-sys.svg)

Low-level FFI bindings for NCCL, the NVIDIA Collective Communications Library.

Most users want the safe crate, [`singe-nccl`](../singe-nccl/README.md).

## Bindings

| Binds                                                            | Header               | Module     | Feature     |
| ---------------------------------------------------------------- | -------------------- | ---------- | ----------- |
| NCCL communicators, collectives, and point-to-point transfers    | `nccl.h`             | crate root | `nccl_2_32` |
| The host side of NCCL's device API, such as device communicators | `nccl_device/host.h` | crate root | `nccl_2_32` |

The `nccl_2_32` feature is on by default.
The bindings are generated with bindgen from the NCCL 2.32 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads the library on the first call of one of its functions, or up front with `load`.
When the library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
`provide` serves NCCL from an in-process implementation instead, such as [`singe-cuda-emulator`](../singe-cuda-emulator/README.md).
NCCL has no Windows release.
