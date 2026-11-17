# singe-cublas

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cublas.svg)](https://crates.io/crates/singe-cublas)
[![Documentation](https://docs.rs/singe-cublas/badge.svg)](https://docs.rs/singe-cublas)
![License](https://img.shields.io/crates/l/singe-cublas.svg)

Safe Rust wrappers for cuBLAS, NVIDIA's dense linear algebra library, with optional cuBLASLt and cuBLASXt support.
A `Context` owns a cuBLAS handle in a `singe-cuda` context, and the BLAS routines take it with their device buffers.

This crate provides:

- **BLAS routines:** levels 1, 2, and 3 and the cuBLAS extensions in `blas`, generic over `f32`, `f64`, `Complex32`, and `Complex64`.
- **Typed GEMM:** column-major matrix views and strided batches described by a `GemmShape`, plus a `Gemm` operation that owns its buffers and can be recorded into a CUDA graph.
- **Scalars:** `Scalar` and `ScalarMut` hold host or device values and set the cuBLAS pointer mode to match.
- **cuBLASLt:** matmul descriptors, matrix layouts, algorithm heuristics, fused epilogues, and scaled FP8 and FP4 inputs, behind the `lt` feature.
- **cuBLASXt:** level 3 routines on host memory, split across several GPUs, behind the `xt` feature.

## Example

Multiply two column-major 2x2 matrices and read the product back:

```rust,no_run
use singe_cublas::{blas::level3::gemm, context::Context, scalar::Scalar, types::Operation};
use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let context = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroed(&cuda, 4)?;

    gemm(
        &context,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0),
        &a,
        2,
        &b,
        2,
        Scalar::host(&0.0),
        &mut c,
        2,
    )?;

    assert_eq!(c.copy_to_host_vec()?, [19.0, 43.0, 22.0, 50.0]);
    Ok(())
}
```

`gemm` checks the shape, the leading dimensions, and the CUDA context of every operand, then waits for the handle's stream before it returns.
Each routine also has an `unsafe` `<operation>_async` form that only enqueues the work and can be captured into a CUDA graph; its caller keeps the operands alive until the work finishes.

## Features

- `lt`: the `lt` module, for cuBLASLt.
- `xt`: the `xt` module, for cuBLASXt.

## More examples

The examples are grouped by library: [classic cuBLAS](./examples/cublas), [cuBLASLt](./examples/lt) (run with `--features lt`), and [cuBLASXt](./examples/xt) (run with `--features xt`).

- [`cublas_level3_gemm`](./examples/cublas/cublas_level3_gemm.rs): matrix multiplication with classic BLAS.
- [`cublas_extension_gemm_strided_batched_ex`](./examples/cublas/cublas_extension_gemm_strided_batched_ex.rs): a batch of matrix products with an explicit compute type.
- [`lt_sgemm`](./examples/lt/lt_sgemm.rs): cuBLASLt descriptors, algorithm heuristics, and workspace.
- [`lt_epilogue`](./examples/lt/lt_epilogue.rs): a matmul with a fused epilogue.
- [`lt_fp8_matmul`](./examples/lt/lt_fp8_matmul.rs): FP8 matrix multiplication with scaling factors.
- [`xt_sgemm`](./examples/xt/xt_sgemm.rs): cuBLASXt matrix multiplication of host arrays.
