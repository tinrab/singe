# singe-cusparse

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cusparse.svg)](https://crates.io/crates/singe-cusparse)
[![Documentation](https://docs.rs/singe-cusparse/badge.svg)](https://docs.rs/singe-cusparse)
![License](https://img.shields.io/crates/l/singe-cusparse.svg)

Safe Rust wrappers for NVIDIA cuSPARSE, for sparse linear algebra on the GPU.
A `Context` owns a cuSPARSE handle in a `singe-cuda` context, and the descriptors in `matrix` and `vector` borrow device buffers.
Creating and configuring descriptors is safe; every operation and workspace-size query is `unsafe`, because a descriptor does not check that its sparse indices are in bounds or its values initialized.

This crate provides:

- **Sparse formats:** CSR, CSC, COO, BSR, blocked-ELL, and sliced-ELL matrix descriptors, dense descriptors, and conversions between them.
- **Products:** sparse matrix-vector (`spmv`), sparse-dense matrix-matrix (`spmm`), sparse-sparse (`spgemm`), and sampled dense-dense (`sddmm`) multiplication.
- **Custom operators:** `SpMmOpPlan` compiles a sparse-dense product with operators supplied as LTO-IR.
- **Triangular solves:** `spsv` and `spsm` for one or several right-hand sides, with reusable analysis.
- **Vectors:** sparse and dense vector descriptors, with gather and scatter between them.

## Example

Compute `y = A * x` for a 3x3 CSR matrix and read `y` back:

```rust,no_run
use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceOutputView};
use singe_cusparse::{
    context::Context,
    matrix::SparseMatrixDescriptor,
    scalar::Scalar,
    spmv::{spmv_async, spmv_workspace_size},
    types::{IndexBase, Operation, SpMvAlgorithm},
    vector::DenseVectorDescriptor,
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let context = Context::create(&cuda)?;

    // A = [1 0 2; 0 3 0; 4 0 5] in CSR, and x = [1, 2, 3].
    let mut row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 3, 5])?;
    let mut column_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 0, 2])?;
    let mut values = DeviceMemory::from_slice(&cuda, &[1.0_f32, 2.0, 3.0, 4.0, 5.0])?;
    let mut x = DeviceMemory::from_slice(&cuda, &[1.0_f32, 2.0, 3.0])?;
    let mut y = DeviceMemory::<f32>::zeroed(&cuda, 3)?;

    {
        let matrix = SparseMatrixDescriptor::create_csr(
            &context,
            3,
            3,
            5,
            Some(DeviceOutputView::from_output(&mut row_offsets)),
            Some(DeviceOutputView::from_output(&mut column_indices)),
            Some(DeviceOutputView::from_output(&mut values)),
            IndexBase::Zero,
        )?;
        let x_vector =
            DenseVectorDescriptor::create(&context, 3, DeviceOutputView::from_output(&mut x))?;
        let mut y_vector =
            DenseVectorDescriptor::create(&context, 3, DeviceOutputView::from_output(&mut y))?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;

        // SAFETY: the descriptors cover the initialized device buffers above, and the CSR indices are in bounds.
        let workspace_size = unsafe {
            spmv_workspace_size::<f32>(
                &context,
                Operation::NonTranspose,
                Scalar::host(&alpha),
                &matrix,
                &x_vector,
                Scalar::host(&beta),
                &mut y_vector,
                SpMvAlgorithm::Default,
            )?
        };
        let mut workspace = DeviceMemory::<u8>::zeroed(&cuda, workspace_size.max(1))?;
        context.sync_scope(|| {
            // SAFETY: as above, and `sync_scope` waits for the stream before the borrowed buffers are released.
            unsafe {
                spmv_async::<f32>(
                    &context,
                    Operation::NonTranspose,
                    Scalar::host(&alpha),
                    &matrix,
                    &x_vector,
                    Scalar::host(&beta),
                    &mut y_vector,
                    SpMvAlgorithm::Default,
                    Some(DeviceOutputView::from_output(&mut workspace)),
                )
            }
        })?;
    }

    assert_eq!(y.copy_to_host_vec()?, [7.0, 6.0, 19.0]);
    Ok(())
}
```

`spmv_async` checks that the descriptors, scalars, and workspace belong to the handle's CUDA context, enqueues the product, and returns without waiting.
cuSPARSE operations have no waiting forms: `Context::sync_scope` runs a closure of `_async` calls, rejects stream capture, and waits for the handle's stream even when the closure fails.
The `_async` functions can also be called during stream capture, to record them into a CUDA graph.

## More examples

- [`spmv_csr`](./examples/spmv_csr.rs): `y = alpha * A * x + beta * y` with a 4x4 CSR matrix.
