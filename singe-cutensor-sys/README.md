# singe-cutensor-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cutensor-sys.svg)](https://crates.io/crates/singe-cutensor-sys)
[![Documentation](https://docs.rs/singe-cutensor-sys/badge.svg)](https://docs.rs/singe-cutensor-sys)
![License](https://img.shields.io/crates/l/singe-cutensor-sys.svg)

Low-level FFI bindings for cuTENSOR 2.8 (`cutensor.h`), with optional bindings for cuTENSORMg (`cutensorMg.h`) and cuTENSORMp (`cutensorMp.h`).
Each library loads at run time through [`singe-loader`](../singe-loader/README.md), on the first call of one of its functions or up front with `load`, `load_mg`, and `load_mp`, so building links no NVIDIA library.
If the library or one of its symbols is missing, a function that returns a status returns a failure status, and `take_absence` reports what was missing.
Most users want the safe [`singe-cutensor`](../singe-cutensor/README.md) crate.

## Features

- `mg`: cuTENSORMg, for tensor operations across several GPUs in one process.
- `mp`: cuTENSORMp, for tensor operations across processes over NCCL; adds a dependency on `singe-nccl-sys`.
