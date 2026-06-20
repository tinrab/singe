mod common;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudss::{
    error::Result,
    matrix::{BatchCsrMatrixDescriptor, BatchDenseMatrixDescriptor, Matrix},
    memory::DevicePointerTable,
    types::{IndexBase, Layout, MatrixFormat, MatrixType, MatrixViewType},
};

fn main() -> Result<()> {
    println!("cuDSS example: batch sparse and dense matrix helper APIs");

    let rows = [5_i32, 6];
    let cols = rows;
    let nnz = [8_i32, 11];
    let nrhs = [1_i32, 1];

    let row_offsets = [
        DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?,
        DeviceMemory::from_slice(&[0_i32, 2, 4, 7, 8, 10, 11])?,
    ];
    let col_indices = [
        DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?,
        DeviceMemory::from_slice(&[0_i32, 5, 1, 4, 2, 4, 5, 3, 4, 5, 5])?,
    ];
    let values = [
        DeviceMemory::from_slice(&[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?,
        DeviceMemory::from_slice(&[3.0f64, 1.0, 2.0, 1.0, 6.0, 2.0, 2.0, 5.0, 7.0, 3.0, 8.0])?,
    ];
    let rhs = [
        DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0])?,
        DeviceMemory::from_slice(&[9.0f64, 9.0, 40.0, 20.0, 61.0, 70.0])?,
    ];

    let replacement_row_offsets = [
        DeviceMemory::from_slice(&[0_i32, 2, 4, 5, 6, 7])?,
        DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 9, 10])?,
    ];
    let replacement_col_indices = [
        DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 3, 4])?,
        DeviceMemory::from_slice(&[0_i32, 5, 1, 4, 2, 4, 3, 4, 5, 5])?,
    ];
    let replacement_values = [
        DeviceMemory::from_slice(&[-4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 2.0, 0.0])?,
        DeviceMemory::from_slice(&[-3.0f64, 1.0, 2.0, 1.0, 6.0, 2.0, 5.0, 7.0, 3.0, 8.0, 0.0])?,
    ];
    let replacement_rhs = [
        DeviceMemory::from_slice(&[-2.0f64, 12.0, 25.0, 4.0, 13.0])?,
        DeviceMemory::from_slice(&[-5.0f64, 9.0, 40.0, 20.0, 61.0, 70.0])?,
    ];

    let row_offset_ptrs = DeviceMemory::from_slice(
        &row_offsets
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let col_index_ptrs = DeviceMemory::from_slice(
        &col_indices
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let value_ptrs = DeviceMemory::from_slice(
        &values
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let rhs_ptrs = DeviceMemory::from_slice(
        &rhs.each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;

    let replacement_row_offset_ptrs = DeviceMemory::from_slice(
        &replacement_row_offsets
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let replacement_col_index_ptrs = DeviceMemory::from_slice(
        &replacement_col_indices
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let replacement_value_ptrs = DeviceMemory::from_slice(
        &replacement_values
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let replacement_rhs_ptrs = DeviceMemory::from_slice(
        &replacement_rhs
            .each_ref()
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let row_offset_table =
        unsafe { DevicePointerTable::<i32>::from_device_memory(&row_offset_ptrs) };
    let col_index_table = unsafe { DevicePointerTable::<i32>::from_device_memory(&col_index_ptrs) };
    let value_table = unsafe { DevicePointerTable::<f64>::from_device_memory(&value_ptrs) };
    let rhs_table = unsafe { DevicePointerTable::<f64>::from_device_memory(&rhs_ptrs) };
    let replacement_row_offset_table =
        unsafe { DevicePointerTable::<i32>::from_device_memory(&replacement_row_offset_ptrs) };
    let replacement_col_index_table =
        unsafe { DevicePointerTable::<i32>::from_device_memory(&replacement_col_index_ptrs) };
    let replacement_value_table =
        unsafe { DevicePointerTable::<f64>::from_device_memory(&replacement_value_ptrs) };
    let replacement_rhs_table =
        unsafe { DevicePointerTable::<f64>::from_device_memory(&replacement_rhs_ptrs) };

    let mut b = Matrix::create_batch_dense(
        BatchDenseMatrixDescriptor {
            rows: &rows,
            columns: &nrhs,
            leading_dim: &rows,
            layout: Layout::ColumnMajor,
        },
        rhs_table,
    )?;

    let mut a = Matrix::create_batch_csr(
        BatchCsrMatrixDescriptor {
            rows: &rows,
            columns: &cols,
            nnz: &nnz,
            matrix_type: MatrixType::Spd,
            view_type: MatrixViewType::Upper,
            index_base: IndexBase::Zero,
        },
        row_offset_table,
        None,
        col_index_table,
        value_table,
    )?;

    let a_info = a.batch_csr_info()?;
    assert_eq!(a_info.rows, rows.map(i64::from));
    assert_eq!(a_info.columns, cols.map(i64::from));
    assert_eq!(a_info.nnz, nnz.map(i64::from));
    assert_eq!(a_info.row_start, unsafe {
        DevicePtr::from_raw(row_offset_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(a_info.column_indices, unsafe {
        DevicePtr::from_raw(col_index_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(a_info.values, unsafe {
        DevicePtr::from_raw(value_ptrs.as_mut_ptr().cast())
    });

    let b_info = b.batch_dense_info()?;
    assert_eq!(b_info.rows, rows.map(i64::from));
    assert_eq!(b_info.columns, nrhs.map(i64::from));
    assert_eq!(b_info.leading_dim, rows.map(i64::from));
    assert_eq!(b_info.values, unsafe {
        DevicePtr::from_raw(rhs_ptrs.as_mut_ptr().cast())
    });
    println!("batch info matched the original descriptors");

    a.set_batch_values(replacement_value_table)?;
    b.set_batch_values(replacement_rhs_table)?;
    assert_eq!(a.batch_csr_info()?.values, unsafe {
        DevicePtr::from_raw(replacement_value_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(b.batch_dense_info()?.values, unsafe {
        DevicePtr::from_raw(replacement_rhs_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(replacement_values[0].copy_to_host_vec()?[0], -4.0);
    assert_eq!(replacement_values[1].copy_to_host_vec()?[0], -3.0);
    assert_eq!(replacement_rhs[0].copy_to_host_vec()?[0], -2.0);
    assert_eq!(replacement_rhs[1].copy_to_host_vec()?[0], -5.0);
    println!("set_batch_values replaced sparse and dense value pointer arrays");

    a.set_batch_csr_values(
        replacement_row_offset_table,
        None,
        replacement_col_index_table,
        replacement_value_table,
    )?;
    let a_info = a.batch_csr_info()?;
    assert_eq!(a_info.row_start, unsafe {
        DevicePtr::from_raw(replacement_row_offset_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(a_info.column_indices, unsafe {
        DevicePtr::from_raw(replacement_col_index_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(a_info.values, unsafe {
        DevicePtr::from_raw(replacement_value_ptrs.as_mut_ptr().cast())
    });
    assert_eq!(a_info.nnz, nnz.map(i64::from));
    println!("set_batch_csr_pointers replaced CSR pointer arrays; cuDSS keeps original nnz");

    assert!(
        a.format()?
            .contains(MatrixFormat::CSR | MatrixFormat::BATCH)
    );
    assert!(
        b.format()?
            .contains(MatrixFormat::DENSE | MatrixFormat::BATCH)
    );
    println!("format reports CSR batch and dense batch matrices");

    println!("Example PASSED");
    Ok(())
}
