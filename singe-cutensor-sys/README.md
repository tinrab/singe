# singe-cutensor-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cutensor-sys.svg)](https://crates.io/crates/singe-cutensor-sys)
[![Documentation](https://docs.rs/singe-cutensor-sys/badge.svg)](https://docs.rs/singe-cutensor-sys)
![License](https://img.shields.io/crates/l/singe-cutensor-sys.svg)

Low-level FFI bindings for NVIDIA cuTENSOR, cuTENSORMg, and cuTENSORMp.

Most users want the safe crate, [`singe-cutensor`](../singe-cutensor/README.md).

## Bindings

| Binds                                                                 | Header         | Module     | Feature        |
| --------------------------------------------------------------------- | -------------- | ---------- | -------------- |
| cuTENSOR, tensor contractions, reductions, and elementwise operations | `cutensor.h`   | crate root | `cutensor_2_8` |
| cuTENSORMg, for tensor operations across several GPUs in one process  | `cutensorMg.h` | crate root | `mg`           |
| cuTENSORMp, for tensor operations across processes over NCCL          | `cutensorMp.h` | crate root | `mp`           |

The `cutensor_2_8` feature is on by default, and the others are off; `mp` adds a dependency on `singe-nccl-sys`.
The bindings are generated with bindgen from the cuTENSOR 2.8 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads each library on the first call of one of its functions, or up front with `load` for cuTENSOR, `load_mg` for cuTENSORMg, and `load_mp` for cuTENSORMp.
When a library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
