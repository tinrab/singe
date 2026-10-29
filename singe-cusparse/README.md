# singe-cusparse

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cusparse.svg)](https://crates.io/crates/singe-cusparse)
[![Documentation](https://docs.rs/singe-cusparse/badge.svg)](https://docs.rs/singe-cusparse)
![License](https://img.shields.io/crates/l/singe-cusparse.svg)

Safe Rust wrappers for NVIDIA cuSPARSE, for sparse linear algebra on the GPU.

This crate provides:

- Sparse and dense vector and matrix descriptors.
- Sparse matrix-vector and matrix-matrix products.
- Sampled products, sparse matrix multiplication, and triangular solves.
- Typed scalar arguments and explicit analysis and workspace APIs.

## Example

Compute `y = A * x` for a sparse matrix `A`.
Assume `matrix` describes a CSR matrix and `x` and `y` are dense vectors:

```rust,ignore
use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_cusparse::{
	scalar::Scalar,
	spmv::{spmv, spmv_buffer_size},
	types::{Operation, SpMvAlgorithm},
};

let alpha = 1.0_f32;
let beta = 0.0_f32;
// SAFETY: descriptors and CSR indices match the initialized buffers.
let size = unsafe {
	spmv_buffer_size::<f32>(
		&context, Operation::NonTranspose, Scalar::host(&alpha), &matrix,
		&x, Scalar::host(&beta), &mut y, SpMvAlgorithm::Default,
	)?
};
let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, size.max(1))?;
context.stream()?.sync_scope(|| {
	// SAFETY: descriptors, buffers, scalars, and scratch survive the draining scope.
	unsafe {
		spmv::<f32>(
			&context, Operation::NonTranspose, Scalar::host(&alpha), &matrix,
			&x, Scalar::host(&beta), &mut y, SpMvAlgorithm::Default,
			Some(DeviceOutputView::from_output(&mut workspace)),
		)
	}
})?;
```

The [complete example](examples/spmv_csr.rs) builds the matrix and checks the result.

See the [API reference](https://docs.rs/singe-cusparse).
