mod common;

use singe_cuda::{context::Context, memory::DeviceMemory, types::DevicePtr};
use singe_cudss::{
    error::Result,
    matrix::{CsrMatrixDescriptor, Matrix},
    types::{DataType, IndexBase, MatrixFormat, MatrixType, MatrixViewType},
};

fn main() -> Result<()> {
    let cuda = Context::create()?;
    println!("cuDSS example: sparse matrix helper APIs");
    println!("cuDSS version: {}", singe_cudss::version()?);

    let rows = 5i64;
    let cols = rows;
    let nnz = 9i64;
    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 8, 9])?;
    let col_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 1, 1, 2, 2, 3, 3, 4, 4])?;
    let values =
        DeviceMemory::from_slice(&cuda, &[0.0f64, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])?;

    let replacement_row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 8, 9])?;
    let replacement_col_indices =
        DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 3, 2, 4, 3, 4, 4])?;
    let replacement_values = DeviceMemory::from_slice(
        &cuda,
        &[-4.0f64, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
    )?;

    let mut matrix = Matrix::create_csr(
        CsrMatrixDescriptor {
            rows,
            columns: cols,
            nnz,
            matrix_type: MatrixType::Symmetric,
            view_type: MatrixViewType::Upper,
            index_base: IndexBase::Zero,
        },
        row_offsets.view(),
        None,
        col_indices.view(),
        values.view(),
    )?;

    let info = matrix.csr_info()?;
    assert_eq!(info.rows, rows);
    assert_eq!(info.columns, cols);
    assert_eq!(info.nnz, nnz);
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(info.row_start, unsafe {
        DevicePtr::from_raw(row_offsets.as_mut_ptr().cast())
    });
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(info.column_indices, unsafe {
        DevicePtr::from_raw(col_indices.as_mut_ptr().cast())
    });
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(info.values, unsafe {
        DevicePtr::from_raw(values.as_mut_ptr().cast())
    });
    assert_eq!(info.value_type, DataType::F64);
    assert_eq!(info.matrix_type, MatrixType::Symmetric);
    println!("csr_info matched the original descriptor");

    matrix.set_values(replacement_values.view())?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(matrix.csr_info()?.values, unsafe {
        DevicePtr::from_raw(replacement_values.as_mut_ptr().cast())
    });
    println!("set_values replaced the sparse value buffer");

    matrix.set_csr_values(
        replacement_row_offsets.view(),
        None,
        replacement_col_indices.view(),
        replacement_values.view(),
    )?;
    let info = matrix.csr_info()?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(info.row_start, unsafe {
        DevicePtr::from_raw(replacement_row_offsets.as_mut_ptr().cast())
    });
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(info.column_indices, unsafe {
        DevicePtr::from_raw(replacement_col_indices.as_mut_ptr().cast())
    });
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    assert_eq!(info.values, unsafe {
        DevicePtr::from_raw(replacement_values.as_mut_ptr().cast())
    });
    assert_eq!(info.nnz, nnz);
    println!("set_csr_pointers replaced the CSR buffers; cuDSS keeps reporting the original nnz");

    assert!(matrix.format()?.contains(MatrixFormat::CSR));
    println!("format reports a CSR matrix");
    println!("Example PASSED");
    Ok(())
}
