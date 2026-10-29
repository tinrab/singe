use std::mem::MaybeUninit;

mod common;

use std::slice;

use singe_cuda::types::DevicePtr;
use singe_cudss::{
    config::Config,
    error::Result,
    matrix::{CsrMatrixDescriptor, DenseMatrixDescriptor, Matrix},
    types::{ConfigParameter, DataType, IndexBase, Layout, MatrixType, MatrixViewType},
};

use crate::common::{create_context, solve};

fn host_ptr<T>(values: &[T]) -> DevicePtr {
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    unsafe { DevicePtr::from_raw(values.as_ptr().cast_mut().cast()) }
}

fn host_mut_ptr<T>(values: &mut [T]) -> DevicePtr {
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    unsafe { DevicePtr::from_raw(values.as_mut_ptr().cast()) }
}

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solving with hybrid execute mode and host buffers");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = [0_i32, 2, 4, 6, 7, 8];
    let col_indices = [0_i32, 2, 1, 2, 2, 4, 3, 4];
    let a_values = [4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0];
    let b_values = [7.0f64, 12.0, 25.0, 4.0, 13.0];
    let mut x_values = vec![MaybeUninit::<f64>::uninit(); (n * nrhs) as usize];

    let mut config = Config::create()?;
    let hybrid_execute_mode = 1i32;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::HybridExecuteMode, &hybrid_execute_mode) }?;
    let mut data = context.create_data()?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    let b = unsafe {
        Matrix::create_dense_raw(
            &cuda,
            DenseMatrixDescriptor {
                rows: n,
                columns: nrhs,
                leading_dim: n,
                layout: Layout::ColumnMajor,
            },
            host_ptr(&b_values),
            DataType::F64,
        )
    }?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    let mut x = unsafe {
        Matrix::create_dense_raw(
            &cuda,
            DenseMatrixDescriptor {
                rows: n,
                columns: nrhs,
                leading_dim: n,
                layout: Layout::ColumnMajor,
            },
            host_mut_ptr(&mut x_values),
            DataType::F64,
        )
    }?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    let a = unsafe {
        Matrix::create_csr_raw(
            &cuda,
            CsrMatrixDescriptor {
                rows: n,
                columns: n,
                nnz,
                matrix_type: MatrixType::Spd,
                view_type: MatrixViewType::Upper,
                index_base: IndexBase::Zero,
            },
            host_ptr(&row_offsets),
            DevicePtr::null(),
            host_ptr(&col_indices),
            host_ptr(&a_values),
            DataType::I32,
            DataType::I32,
            DataType::F64,
        )
    }?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        solve(
            slice::from_ref(&stream),
            &context,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;
    stream.synchronize()?;

    drop(x);
    // SAFETY: the completed full solve produced every compact host element.
    let x_values: Vec<_> = x_values
        .into_iter()
        .map(|value| unsafe { value.assume_init() })
        .collect();
    for (index, value) in x_values.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        println!("x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
