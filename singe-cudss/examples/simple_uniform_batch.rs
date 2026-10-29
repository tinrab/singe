mod common;

use std::slice;

use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};
use singe_cudss::{
    config::Config,
    error::Result,
    types::{ConfigParameter, Phase},
};

use crate::common::{
    create_context, dense_matrix, dense_output, execute, ptr_at, solve, spd_csr_matrix,
};

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solving a uniform batch of two SPD systems");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;
    let batch_size = 2i32;

    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(
        &cuda,
        &[
            4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 6.0, 2.0, 4.0, 8.0,
        ],
    )?;
    let b_values = DeviceMemory::from_slice(
        &cuda,
        &[7.0f64, 12.0, 25.0, 4.0, 13.0, 13.0, 15.0, 29.0, 8.0, 14.0],
    )?;
    let mut x_values =
        DeviceMemory::<f64>::create_uninit(&cuda, (batch_size as i64 * n * nrhs) as usize)?;

    let mut config = Config::create()?;
    let mut data = context.create_data()?;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::UniformBatchSize, &batch_size) }?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_output(n, nrhs, n, x_values.output())?;
    let mut a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

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
    cuda.synchronize()?;

    drop(x);
    // SAFETY: successful completed solve produced both compact batch solutions.
    let mut x_values = unsafe { x_values.assume_init() };
    let x_host = x_values.copy_to_host_vec()?;
    for (batch, expected_values) in [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
        .into_iter()
        .enumerate()
    {
        for (index, expected) in expected_values.into_iter().enumerate() {
            let value = x_host[batch * n as usize + index];
            println!("batch {batch}: x[{index}] = {value:.4}, expected {expected:.4}");
            assert!((value - expected).abs() <= 2.0e-15);
        }
    }

    let batch_index = 1i32;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::UniformBatchIndex, &batch_index) }?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        a.set_values_raw(ptr_at(&a_values, batch_index as usize * nnz as usize))?;
    }
    let mut x = dense_output::<f64>(n, nrhs, n, DeviceOutputView::from_output(&mut x_values))?;

    let mut b_one = b;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        b_one.set_values_raw(ptr_at(
            &b_values,
            batch_index as usize * (n * nrhs) as usize,
        ))?;
    }

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::FACTORIZATION,
            &config,
            &mut data,
            &a,
            &mut x,
            &b_one,
        )
    }?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::SOLVE,
            &config,
            &mut data,
            &a,
            &mut x,
            &b_one,
        )
    }?;
    cuda.synchronize()?;

    drop(x);
    let x_host = x_values.copy_to_host_vec()?;
    for (index, expected) in [5.0, 4.0, 3.0, 2.0, 1.0].into_iter().enumerate() {
        let value = x_host[index];
        println!("selected batch: x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
