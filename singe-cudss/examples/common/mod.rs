#![allow(dead_code)]

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    stream::{Stream, StreamOptions},
    view::DeviceOutputView,
};
use singe_cudss::{
    config::Config,
    context::Context,
    error::{Error, Result},
    matrix::{CsrMatrixDescriptor, DenseMatrixDescriptor, Matrix, MatrixIndex},
    solver_state::SolverState,
    types::{DataTypeLike, Fp64Mp2, IndexBase, MatrixType, MatrixViewType, Phase},
};

pub(crate) fn create_context() -> Result<(Arc<CudaContext>, Stream, Context)> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream(StreamOptions::default())?;
    let mut cudss = Context::create(&cuda)?;
    cudss.set_stream(Some(&stream))?;
    Ok((cuda, stream, cudss))
}

pub(crate) fn mp2(value: f64) -> Fp64Mp2 {
    Fp64Mp2::from_f64(value)
}

pub(crate) fn dense_matrix<T: DataTypeLike>(
    rows: i64,
    columns: i64,
    leading_dim: i64,
    values: &DeviceMemory<T>,
) -> Result<Matrix<'_>> {
    Matrix::create_dense(
        DenseMatrixDescriptor::new(rows, columns, leading_dim),
        values.view(),
    )
}

pub(crate) fn spd_csr_matrix<'a, O, I, T>(
    rows: i64,
    columns: i64,
    nnz: i64,
    row_offsets: &'a DeviceMemory<O>,
    column_indices: &'a DeviceMemory<I>,
    values: &'a DeviceMemory<T>,
) -> Result<Matrix<'a>>
where
    O: MatrixIndex,
    I: MatrixIndex,
    T: DataTypeLike,
{
    Matrix::create_csr(
        CsrMatrixDescriptor {
            rows,
            columns,
            nnz,
            matrix_type: MatrixType::Spd,
            view_type: MatrixViewType::Upper,
            index_base: IndexBase::Zero,
        },
        row_offsets.view(),
        None,
        column_indices.view(),
        values.view(),
    )
}

pub(crate) unsafe fn solve(
    streams: &[Stream],
    context: &Context,
    config: &Config,
    data: &mut SolverState<'_>,
    a: &Matrix<'_>,
    x: &mut Matrix<'_>,
    b: &Matrix<'_>,
) -> Result<()> {
    drain(streams, || {
        // SAFETY: the caller guarantees `execute_async`'s contract for these matrices, and `drain` waits for every stream in `streams` before returning.
        unsafe {
            context.execute_async(Phase::ANALYSIS, config, data, a, x, b)?;
            context.execute_async(Phase::FACTORIZATION, config, data, a, x, b)?;
            context.execute_async(Phase::SOLVE, config, data, a, x, b)
        }
    })?;
    check_info(data)
}

pub(crate) fn dense_output<T: DataTypeLike>(
    rows: i64,
    columns: i64,
    leading_dim: i64,
    values: DeviceOutputView<'_, T>,
) -> Result<Matrix<'_>> {
    Matrix::create_dense_output(
        DenseMatrixDescriptor::new(rows, columns, leading_dim),
        values,
    )
}

/// Completes a raw phase on all explicitly selected streams before returning.
///
/// # Safety
/// Follow `Context::execute_async`'s matrix, phase, ordering, and access contracts.
/// `streams` must cover all native work and must be the streams selected on context.
pub(crate) unsafe fn execute(
    streams: &[Stream],
    context: &Context,
    phase: Phase,
    config: &Config,
    data: &mut SolverState<'_>,
    a: &Matrix<'_>,
    x: &mut Matrix<'_>,
    b: &Matrix<'_>,
) -> Result<()> {
    // SAFETY: the caller guarantees `execute_async`'s contract and that `streams` are the context's selected streams, which `drain` waits for.
    drain(streams, || unsafe {
        context.execute_async(phase, config, data, a, x, b)
    })?;
    if phase.intersects(Phase::FACTORIZATION | Phase::REFACTORIZATION | Phase::SOLVE) {
        check_info(data)?;
    }
    Ok(())
}

fn check_info(data: &SolverState<'_>) -> Result<()> {
    let info = data.info()?;
    if info != 0 {
        return Err(Error::OperationFailed {
            operation: "cudss execute".into(),
            info,
        });
    }
    Ok(())
}

fn drain<R>(streams: &[Stream], action: impl FnOnce() -> Result<R>) -> Result<R> {
    match streams.split_first() {
        Some((stream, rest)) => stream.sync_scope(|_| Ok(drain(rest, action)))?,
        None => action(),
    }
}
