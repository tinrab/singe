singe-nccl
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-nccl.svg)](https://crates.io/crates/singe-nccl)
[![Documentation](https://docs.rs/singe-nccl/badge.svg)](https://docs.rs/singe-nccl)
![License](https://img.shields.io/crates/l/singe-nccl.svg)

Safe NCCL communicator, collective, group, local multi-rank, and buffer wrappers.

This crate provides Rust ownership around NCCL communicators, unique IDs,
grouped calls, collective operations, device buffers, and NCCL memory helpers.

For single-process multi-GPU programs, prefer `local::LocalCommunicatorGroup`.
It owns one communicator and stream per local rank and launches each collective
inside an NCCL group internally. The lower-level `communicator::Communicator`
API remains available for process-per-rank programs, custom scheduling, and
direct NCCL behavior tests.

## Examples

Runnable examples are available under [`examples/`](examples/).
