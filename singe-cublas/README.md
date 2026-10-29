# singe-cublas

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cublas.svg)](https://crates.io/crates/singe-cublas)
[![Documentation](https://docs.rs/singe-cublas/badge.svg)](https://docs.rs/singe-cublas)
![License](https://img.shields.io/crates/l/singe-cublas.svg)

Rust wrappers for NVIDIA cuBLAS dense linear algebra on the GPU: the classic BLAS API, cuBLASLt for configurable matrix multiplication, and cuBLASXt for BLAS3 across several GPUs.
The low-level asynchronous functions are public too.
When you call them, you must keep every buffer alive until the stream has used it.

This crate provides:

- Classic BLAS routines for device-resident vectors and matrices.
- cuBLASLt matmul descriptors, algorithm selection, and fused epilogues.
- cuBLASXt BLAS3 routines that take host memory and split the work across the GPUs you select.
- Strided copies, and GEMM operations that own their buffers and can be captured in CUDA graphs.

## Example

Multiply two column-major matrices with classic cuBLAS:

```rust
use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};
use singe_cublas::{blas::level3::dgemm, context::Context, types::Operation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let cuda = CudaContext::create()?;
	let context = Context::create(&cuda)?;
	let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
	let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
	let mut c = DeviceMemory::<f64>::zeroed(&cuda, 4)?;
	dgemm(
		&context, Operation::NonTranspose, Operation::NonTranspose,
		2, 2, 2, &1.0, &a, 2, &b, 2, &0.0, &mut c, 2,
	)?;
	assert_eq!(c.copy_to_host_vec()?, [19.0, 43.0, 22.0, 50.0]);
	Ok(())
}
```

## More examples

- [Classic GEMM](examples/cublas/cublas_level3_gemm.rs): matrix multiplication with classic BLAS.
- [Lt GEMM](examples/lt/lt_sgemm.rs): descriptors, heuristics, and workspace.
- [Lt epilogue](examples/lt/lt_epilogue.rs): fuse post-processing with matmul.
- [Xt GEMM](examples/xt/xt_sgemm.rs): host inputs and explicit GPU selection.

See the [API reference](https://docs.rs/singe-cublas).
