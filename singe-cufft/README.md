singe-cufft
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-cufft.svg)](https://crates.io/crates/singe-cufft)
[![Documentation](https://docs.rs/singe-cufft/badge.svg)](https://docs.rs/singe-cufft)
![License](https://img.shields.io/crates/l/singe-cufft.svg)

Safe cuFFT plan and execution wrappers for GPU Fourier transforms.

This crate wraps cuFFT plan creation, stream binding, workspace management, and transform execution over the raw `singe-cufft-sys` bindings.

## Examples

Runnable examples are available under [`examples/`](examples/).
