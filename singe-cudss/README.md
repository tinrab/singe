singe-cudss
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-cudss.svg)](https://crates.io/crates/singe-cudss)
[![Documentation](https://docs.rs/singe-cudss/badge.svg)](https://docs.rs/singe-cudss)
![License](https://img.shields.io/crates/l/singe-cudss.svg)

`singe-cudss` provides safe Rust wrappers for [NVIDIA cuDSS](https://docs.nvidia.com/cuda/cudss/), the CUDA Direct Sparse Solver library.

## Examples

Runnable examples are available under [`examples/`](examples/).

Examples were ported from NVIDIA's [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/241afd3c7a8d1d756100139a3f629df5475fcf5c/cuDSS) repository.

### Example: 5x5 symmetric positive-definite sparse system

This example is based on the official C++ sample [cuDSS/simple/simple.cpp](https://github.com/NVIDIA/CUDALibrarySamples/blob/241afd3c7a8d1d756100139a3f629df5475fcf5c/cuDSS/simple/simple.cpp).

```rust
use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
};
use singe_cudss::{
    config::Config,
    context::Context,
    error::Result,
    matrix::*,
    types::*,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream()?;
    let context = Context::create(&cuda)?;
    context.set_stream(&stream)?;

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[
        4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0,
    ])?;
    let b_values = DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let x_values = DeviceMemory::<f64>::zeroes((n * nrhs) as usize)?;

    let a = Matrix::create_csr(
        CsrMatrixDescriptor {
            rows: n,
            columns: n,
            nnz,
            matrix_type: MatrixType::Spd,
            view_type: MatrixViewType::Upper,
            index_base: IndexBase::Zero,
        },
        &row_offsets,
        None,
        &col_indices,
        &a_values,
    )?;
    let b = Matrix::create_dense(
        DenseMatrixDescriptor::new(n, nrhs, n),
        &b_values,
    )?;
    let mut x = Matrix::create_dense(
        DenseMatrixDescriptor::new(n, nrhs, n),
        &x_values,
    )?;

    let config = Config::create()?;
    let mut data = context.create_data()?;

    context.execute(Phase::ANALYSIS, &config, &mut data, &a, &mut x, &b)?;
    context.execute(Phase::FACTORIZATION, &config, &mut data, &a, &mut x, &b)?;
    context.execute(Phase::SOLVE, &config, &mut data, &a, &mut x, &b)?;

    stream.synchronize()?;

    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        assert!((value - expected).abs() <= 2.0e-15);
    }

    Ok(())
}
```

## Notes

cuDSS 0.8 is documented as a preview release, so upstream APIs and behavior may change.
