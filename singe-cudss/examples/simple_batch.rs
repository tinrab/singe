mod common;

use std::slice;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudss::{
    config::Config,
    error::Result,
    matrix::{BatchCsrMatrixDescriptor, BatchDenseMatrixDescriptor, Matrix},
    memory::DevicePointerTable,
    types::{IndexBase, Layout, MatrixType, MatrixViewType},
};

use crate::common::{create_context, solve};

fn step<T>(name: &str, result: Result<T>) -> Result<T> {
    result.inspect_err(|error| eprintln!("{name} failed: {error}"))
}

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solving a non-uniform batch of two SPD systems");

    let rows = [5_i32, 6];
    let cols = rows;
    let nnz = [8_i32, 11];
    let nrhs = [1_i32, 1];

    let row_offsets = [
        DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?,
        DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 7, 8, 10, 11])?,
    ];
    let col_indices = [
        DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?,
        DeviceMemory::from_slice(&cuda, &[0_i32, 5, 1, 4, 2, 4, 5, 3, 4, 5, 5])?,
    ];
    let a_values = [
        DeviceMemory::from_slice(&cuda, &[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?,
        DeviceMemory::from_slice(
            &cuda,
            &[3.0f64, 1.0, 2.0, 1.0, 6.0, 2.0, 2.0, 5.0, 7.0, 3.0, 8.0],
        )?,
    ];
    let b_values = [
        DeviceMemory::from_slice(&cuda, &[7.0f64, 12.0, 25.0, 4.0, 13.0])?,
        DeviceMemory::from_slice(&cuda, &[9.0f64, 9.0, 40.0, 20.0, 61.0, 70.0])?,
    ];
    let x_values = [
        DeviceMemory::<f64>::create_uninit(&cuda, (rows[0] * nrhs[0]) as usize)?,
        DeviceMemory::<f64>::create_uninit(&cuda, (rows[1] * nrhs[1]) as usize)?,
    ];

    let row_offset_ptrs = DeviceMemory::from_slice(
        &cuda,
        &row_offsets
            .each_ref()
            // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let col_index_ptrs = DeviceMemory::from_slice(
        &cuda,
        &col_indices
            .each_ref()
            // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let a_value_ptrs = DeviceMemory::from_slice(
        &cuda,
        &a_values
            .each_ref()
            // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let b_value_ptrs = DeviceMemory::from_slice(
        &cuda,
        &b_values
            .each_ref()
            // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    let x_value_ptrs = DeviceMemory::from_slice(
        &cuda,
        &x_values
            .each_ref()
            // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
            .map(|memory| unsafe { DevicePtr::from_raw(memory.as_mut_ptr().cast()) }),
    )?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    let row_offset_table =
        unsafe { DevicePointerTable::<i32>::from_device_memory(&row_offset_ptrs) }?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    let col_index_table =
        unsafe { DevicePointerTable::<i32>::from_device_memory(&col_index_ptrs) }?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    let a_value_table = unsafe { DevicePointerTable::<f64>::from_device_memory(&a_value_ptrs) }?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    let b_value_table = unsafe { DevicePointerTable::<f64>::from_device_memory(&b_value_ptrs) }?;
    // SAFETY: the address comes from a live allocation; wrapping it does not dereference it.
    let x_value_table = unsafe { DevicePointerTable::<f64>::from_device_memory(&x_value_ptrs) }?;

    let config = Config::create()?;
    let mut data = context.create_data()?;

    let b = step(
        "create batch dense b",
        Matrix::create_batch_dense(
            BatchDenseMatrixDescriptor {
                rows: &rows,
                columns: &nrhs,
                leading_dim: &rows,
                layout: Layout::ColumnMajor,
            },
            b_value_table,
        ),
    )?;
    let mut x = step(
        "create batch dense x",
        Matrix::create_batch_dense(
            BatchDenseMatrixDescriptor {
                rows: &cols,
                columns: &nrhs,
                leading_dim: &cols,
                layout: Layout::ColumnMajor,
            },
            x_value_table,
        ),
    )?;
    let a = step(
        "create batch csr a",
        Matrix::create_batch_csr(
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
            a_value_table,
        ),
    )?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    step("solve batch", unsafe {
        solve(
            slice::from_ref(&stream),
            &context,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    })?;
    stream.synchronize()?;

    drop(x);
    for (batch, solution) in x_values.into_iter().enumerate() {
        // SAFETY: the checked completed batch solve produced each compact buffer.
        let solution = unsafe { solution.assume_init() }.copy_to_host_vec()?;
        for (index, value) in solution.iter().copied().enumerate() {
            let expected = (index + 1) as f64;
            println!("batch {batch}: x[{index}] = {value:.4}, expected {expected:.4}");
            assert!((value - expected).abs() <= 2.0e-15);
        }
    }

    println!("Example PASSED");
    Ok(())
}
