# singe-cudss

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cudss.svg)](https://crates.io/crates/singe-cudss)
[![Documentation](https://docs.rs/singe-cudss/badge.svg)](https://docs.rs/singe-cudss)
![License](https://img.shields.io/crates/l/singe-cudss.svg)

Safe Rust wrappers for NVIDIA cuDSS, the CUDA direct sparse solver library.

This crate provides:

- **Matrices:** CSR and dense descriptors over borrowed device storage, including uniform and non-uniform batches, with read-only inputs kept apart from write-only outputs.
- **Solver settings:** `Config` with typed parameters for the algorithms, hybrid CPU and GPU execution and memory modes, and the devices of a multi-GPU solve.
- **Solver state:** typed queries such as `info`, `lu_nnz`, `memory_estimates`, and `schur_shape`.
- **Multi-GPU and multi-node solves:** handles over several GPUs, plus the communication and threading layers for multi-node multi-GPU (MGMN) and multithreaded modes.
- **Memory handlers:** custom device allocators for the memory cuDSS allocates.

## Example

Solve a 5x5 symmetric positive-definite system and check the solution:

```rust,no_run
use singe_cuda::{
    context::Context as CudaContext, memory::DeviceMemory, stream::StreamOptions,
    view::DeviceOutputView,
};
use singe_cudss::{
    config::Config,
    context::Context,
    matrix::{CsrMatrixDescriptor, DenseMatrixDescriptor, Matrix},
    solver_state::SolverState,
    types::{IndexBase, MatrixType, MatrixViewType, Phase},
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream(StreamOptions::default())?;
    let mut context = Context::create(&cuda)?;
    context.set_stream(Some(&stream))?;

    // The upper triangle of the matrix in CSR, and the right-hand side.
    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
    let column_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let values = DeviceMemory::from_slice(&cuda, &[4.0_f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let rhs = DeviceMemory::from_slice(&cuda, &[7.0_f64, 12.0, 25.0, 4.0, 13.0])?;
    let mut solution = DeviceMemory::<f64>::zeroed(&cuda, 5)?;

    let config = Config::create()?;
    let mut state = SolverState::create(&context)?;
    {
        let a = Matrix::create_csr(
            CsrMatrixDescriptor {
                rows: 5,
                columns: 5,
                nnz: 8,
                matrix_type: MatrixType::Spd,
                view_type: MatrixViewType::Upper,
                index_base: IndexBase::Zero,
            },
            row_offsets.view(),
            None,
            column_indices.view(),
            values.view(),
        )?;
        let b = Matrix::create_dense(DenseMatrixDescriptor::new(5, 1, 5), rhs.view())?;
        let mut x = Matrix::create_dense_output(
            DenseMatrixDescriptor::new(5, 1, 5),
            DeviceOutputView::from_output(&mut solution),
        )?;

        context.sync_scope(|| {
            // SAFETY: the buffers were written synchronously before this call, the CSR indices are valid,
            // and `sync_scope` waits for the stream before the matrices and their buffers are released.
            unsafe {
                context.execute_async(Phase::ANALYSIS, &config, &mut state, &a, &mut x, &b)?;
                context.execute_async(Phase::FACTORIZATION, &config, &mut state, &a, &mut x, &b)?;
                context.execute_async(Phase::SOLVE, &config, &mut state, &a, &mut x, &b)
            }
        })?;
    }
    assert_eq!(state.info()?, 0);

    let x = solution.copy_to_host_vec()?;
    for (value, expected) in x.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0]) {
        assert!((value - expected).abs() < 1.0e-12);
    }
    Ok(())
}
```

A `Context` owns a cuDSS handle in a `singe-cuda` context and runs the analysis, factorization, and solve phases on matrices that borrow device buffers.
`SolverState` carries the analysis, the factors, and the results from one phase to the next.

`execute_async` checks that the solver state and the matrices belong to the context, enqueues the phase on the context's stream, and returns without waiting.
`Context::sync_scope` runs a closure of `execute_async` calls, rejects stream capture, and waits for the installed stream even when the closure fails; it does not wait for the per-device streams of a multi-GPU handle.
After the solve, `SolverState::info` returns zero if it succeeded; check it before reading the solution.

## More examples

- [`simple`](./examples/simple.rs): a 5x5 symmetric positive-definite system.
- [`simple_complex`](./examples/simple_complex.rs): a complex symmetric positive-definite system.
- [`simple_batch`](./examples/simple_batch.rs) and [`simple_uniform_batch`](./examples/simple_uniform_batch.rs): non-uniform and uniform batches of systems.
- [`simple_high_precision`](./examples/simple_high_precision.rs): a system in extended precision.
- [`simple_schur_complement`](./examples/simple_schur_complement.rs): a dense Schur complement.
- [`simple_hybrid_execute_mode`](./examples/simple_hybrid_execute_mode.rs) and [`simple_hybrid_memory_mode`](./examples/simple_hybrid_memory_mode.rs): hybrid CPU and GPU execution and memory.
- [`simple_memory_handler`](./examples/simple_memory_handler.rs): a custom device memory handler.
- [`simple_mg_mode`](./examples/simple_mg_mode.rs): a solve across several GPUs.
- [`simple_multithreaded_mode`](./examples/simple_multithreaded_mode.rs): a solve with an optional threading layer, set with `Context::set_threading_layer`.
- [`simple_matrix_market`](./examples/simple_matrix_market.rs): a system read from Matrix Market files.

The [examples directory](./examples) also covers MGMN setup, reordering, residuals, and the matrix helper APIs.
