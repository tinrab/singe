# singe-cusolver

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cusolver.svg)](https://crates.io/crates/singe-cusolver)
[![Documentation](https://docs.rs/singe-cusolver/badge.svg)](https://docs.rs/singe-cusolver)
![License](https://img.shields.io/crates/l/singe-cusolver.svg)

Safe Rust wrappers for NVIDIA cuSOLVER, for dense matrix factorizations and solvers on the GPU.

This crate provides:

- Dense matrix factorizations and linear solves.
- Eigenvalue problems, SVD, and iterative refinement.
- Typed operation descriptions and workspace-size queries.
- Methods that wait for completion, alongside the `unsafe` native functions.

## Example

LU-factorize a 3x3 matrix on the GPU.
Assume `a` holds the matrix in column-major order, `pivots` has room for three `i64` pivot indices, `info` for one `i32` status, and `params` holds the solver settings:

```rust,ignore
use std::mem::MaybeUninit;
use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_cusolver::{
	dense::{Getrf, GetrfArgs},
	layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut},
};

let factor = Getrf::<f64>::new(3, 3);
let sizes = factor.workspace_size(&context, &params, MatrixLayout::new(3))?;
let mut device_scratch = DeviceMemory::<u8>::create_uninit(&cuda, sizes.device_bytes)?;
let mut host_scratch = vec![MaybeUninit::<u8>::uninit(); sizes.host_bytes];
factor.run(
	&context,
	&params,
	GetrfArgs {
		a: MatrixMut::new(&mut a, 3),
		pivots: Some(DeviceOutputView::from_output(&mut pivots)),
		workspace: ByteWorkspaceMut::new(&mut device_scratch, &mut host_scratch),
		dev_info: DeviceOutputView::from_output(&mut info),
	},
)?;
assert_eq!(info.copy_to_host_vec()?, [0]);
```

`Getrf` overwrites the matrix with LU factors and produces pivot indices.

See the [API reference](https://docs.rs/singe-cusolver).
