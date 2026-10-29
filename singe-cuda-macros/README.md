# singe-cuda-macros

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda-macros.svg)](https://crates.io/crates/singe-cuda-macros)
[![Documentation](https://docs.rs/singe-cuda-macros/badge.svg)](https://docs.rs/singe-cuda-macros)
![License](https://img.shields.io/crates/l/singe-cuda-macros.svg)

Derive macros for types that can be copied to the GPU, and `cuda_module!` for typed CUDA kernel launches.

This crate provides:

- Derives that check a type's layout is valid on the GPU.
- `cuda_module!`, which compiles CUDA source and generates a typed method for each kernel.
- Generated launch methods that also work with stream capture and CUDA graphs.
- Parameter access declarations such as `read`, `write`, and `read_write`.

The macros are re-exported by [`singe-cuda`](../singe-cuda/README.md) with its default `macros` feature.

## Example

Define a C-compatible value that can be copied between host and device:

```rust
use singe_cuda::{AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr};

#[repr(C)]
#[derive(Clone, Copy, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr, AnyBitPatternDeviceRepr)]
pub struct Position {
	pub x: f32,
	pub y: f32,
}
```

`DeviceRepr` checks that the layout is valid on the GPU.
The other derives mark the type as having no padding bytes, as valid when zeroed, and as valid for any bit pattern.
Each field must implement the same traits, and the no-padding derive fails if the struct has padding.

## More examples

- [Vector addition](../singe-cuda/examples/vector_add.rs): declare and launch a typed CUDA kernel.
- [Graph replay](../singe-cuda/examples/graph_replay.rs): record a generated launch method and replay it.
- [API reference](https://docs.rs/singe-cuda-macros).
