# singe-cusolver

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cusolver.svg)](https://crates.io/crates/singe-cusolver)
[![Documentation](https://docs.rs/singe-cusolver/badge.svg)](https://docs.rs/singe-cusolver)
![License](https://img.shields.io/crates/l/singe-cusolver.svg)

Safe Rust wrappers for NVIDIA cuSOLVER, for matrix factorizations, linear solves, eigenvalue problems, and singular value decompositions on the GPU.

This crate provides:

- **Dense factorizations and solves:** Cholesky, LU, QR, and symmetric indefinite factorizations with their solves, plus triangular inversion and tridiagonal and bidiagonal reductions, in `dense`.
- **Eigenvalue solvers:** symmetric, generalized symmetric, and general problems, such as `Syevd`, `Syevj`, `Sygvd`, and `Geev`, including batched forms.
- **Singular value decompositions:** `Gesvd`, `Gesvdj`, `Gesvdp`, `Gesvdr`, and `GesvdaStridedBatched`.
- **Iterative refinement:** mixed-precision linear and least-squares solvers in `irs`, with their residual history.
- **Layouts:** borrowed column-major matrices, vectors, and host and device workspaces in `layout`.

## Example

LU-factorize a 3x3 matrix and solve `A * x = b`:

```rust,no_run
use std::mem::MaybeUninit;

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceOutputView};
use singe_cusolver::{
    context::Context,
    dense::{Getrf, GetrfBindings, Getrs, GetrsBindings},
    layout::{ByteWorkspaceMut, MatrixLayout, MatrixMut},
    params::Params,
    types::Operation,
};

const N: usize = 3;

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let context = Context::create(&cuda)?;
    let params = Params::create()?;

    // Column-major storage of
    //     | 1 2 3 |
    // A = | 4 5 6 |
    //     | 7 8 10|
    let mut a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0])?;
    let mut b = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0])?;
    let mut pivots = DeviceMemory::<i64>::zeroed(&cuda, N)?;
    let mut info = DeviceMemory::<i32>::zeroed(&cuda, 1)?;

    let getrf = Getrf::<f64>::new(N, N);
    let sizes = getrf.workspace_size(&context, &params, MatrixLayout::new(N))?;
    let mut device_workspace = DeviceMemory::<u8>::zeroed(&cuda, sizes.device_bytes)?;
    let mut host_workspace = vec![MaybeUninit::<u8>::uninit(); sizes.host_bytes];

    getrf.execute(
        &context,
        GetrfBindings {
            a: MatrixMut::new(&mut a, N),
            pivots: Some(DeviceOutputView::from_output(&mut pivots)),
            workspace: ByteWorkspaceMut::new(&mut device_workspace, &mut host_workspace),
            dev_info: DeviceOutputView::from_output(&mut info),
        },
        &params,
    )?;
    assert_eq!(info.copy_to_host_vec()?, [0]);

    let factor = MatrixMut::new(&mut a, N);
    Getrs::<f64>::new(Operation::NonTranspose, N, 1).execute(
        &context,
        GetrsBindings {
            factor: factor.as_ref(),
            pivots: pivots.view(),
            rhs: MatrixMut::new(&mut b, N),
            dev_info: DeviceOutputView::from_output(&mut info),
        },
        &params,
    )?;
    assert_eq!(info.copy_to_host_vec()?, [0]);

    let x = b.copy_to_host_vec()?;
    let expected = [-1.0 / 3.0, 2.0 / 3.0, 0.0];
    assert!(x.iter().zip(expected).all(|(x, e)| (x - e).abs() < 1.0e-9));
    Ok(())
}
```

Each operation is a typed description, such as `dense::Getrf`, that reports its workspace sizes and runs with a bindings struct of borrowed matrices, outputs, and workspaces.

`execute` waits for the context's stream, even when enqueuing fails.
Most operations also have an `unsafe` `execute_async` that only enqueues the work and can be captured into a CUDA graph; the eigenvalue solves that read back a count have none.
A finished operation can still have failed numerically, so check the `dev_info` it writes before using its other outputs.

## More examples

- [`getrf`](./examples/getrf.rs): the example above, with a residual check on the host.
