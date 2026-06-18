#![allow(dead_code)]

use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext, memory::DeviceMemory, stream::Stream, types::DevicePtr,
};
use singe_cudss::{
    config::Config,
    context::Context,
    data::Data,
    error::Result,
    matrix::{CsrMatrixDescriptor, DenseMatrixDescriptor, Matrix, MatrixIndex},
    types::{DataTypeLike, Fp64Mp2, IndexBase, MatrixType, MatrixViewType, Phase},
};

pub fn create_context() -> Result<(Arc<CudaContext>, Stream, Context)> {
    let cuda = CudaContext::create()?;
    let stream = cuda.create_stream()?;
    let cudss = Context::create(&cuda)?;
    cudss.set_stream(&stream)?;
    Ok((cuda, stream, cudss))
}

pub fn mp2(value: f64) -> Fp64Mp2 {
    Fp64Mp2::from_f64(value)
}

pub fn dense_matrix<T: DataTypeLike>(
    rows: i64,
    columns: i64,
    leading_dim: i64,
    values: &DeviceMemory<T>,
) -> Result<Matrix<'_>> {
    Matrix::create_dense(
        DenseMatrixDescriptor::new(rows, columns, leading_dim),
        values,
    )
}

pub fn spd_csr_matrix<'a, O, I, T>(
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
        row_offsets,
        None,
        column_indices,
        values,
    )
}

pub fn solve(
    context: &Context,
    config: &Config,
    data: &mut Data,
    a: &Matrix<'_>,
    x: &mut Matrix<'_>,
    b: &Matrix<'_>,
) -> Result<()> {
    context.execute(Phase::ANALYSIS, config, data, a, x, b)?;
    context.execute(Phase::FACTORIZATION, config, data, a, x, b)?;
    context.execute(Phase::SOLVE, config, data, a, x, b)
}

pub fn ptr_at<T>(memory: &DeviceMemory<T>, offset: usize) -> DevicePtr {
    let ptr = unsafe { memory.as_ptr().add(offset) };
    DevicePtr::from_raw(ptr.cast_mut().cast())
}
