singe-cufile
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-cufile.svg)](https://crates.io/crates/singe-cufile)
[![Documentation](https://docs.rs/singe-cufile/badge.svg)](https://docs.rs/singe-cufile)
![License](https://img.shields.io/crates/l/singe-cufile.svg)

Safe Rust wrappers for NVIDIA cuFile GPUDirect Storage library.

This crate wraps cuFile driver initialization, file handles, buffer registration, direct GPU-memory reads and writes, batch I/O, stream I/O, statistics, and utility queries over the raw `singe-cufile-sys` bindings.

## Examples

Runnable examples are available under [`examples/`](examples/).
