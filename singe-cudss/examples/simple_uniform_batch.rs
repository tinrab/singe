mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{config::Config, error::Result, types::ConfigParameter};

use crate::common::*;

fn main() -> Result<()> {
    println!("cuDSS example: solving a uniform batch of two SPD systems");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;
    let batch_size = 2i32;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[
        4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 6.0, 2.0, 4.0, 8.0,
    ])?;
    let b_values =
        DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0, 13.0, 15.0, 29.0, 8.0, 14.0])?;
    let x_values = DeviceMemory::<f64>::zeroes((batch_size as i64 * n * nrhs) as usize)?;

    let (cuda, _stream, context) = create_context()?;
    let mut config = Config::create()?;
    let mut data = context.create_data()?;
    config.set(ConfigParameter::UniformBatchSize, &batch_size)?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let mut a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    solve(&context, &config, &mut data, &a, &mut x, &b)?;
    cuda.synchronize()?;

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
    config.set(ConfigParameter::UniformBatchIndex, &batch_index)?;
    unsafe {
        a.set_values_raw(ptr_at(&a_values, batch_index as usize * nnz as usize))?;
    }
    x.set_values(&x_values)?;

    let mut b_one = b;
    unsafe {
        b_one.set_values_raw(ptr_at(
            &b_values,
            batch_index as usize * (n * nrhs) as usize,
        ))?;
    }

    context.execute(
        singe_cudss::types::Phase::FACTORIZATION,
        &config,
        &mut data,
        &a,
        &mut x,
        &b_one,
    )?;
    context.execute(
        singe_cudss::types::Phase::SOLVE,
        &config,
        &mut data,
        &a,
        &mut x,
        &b_one,
    )?;
    cuda.synchronize()?;

    let x_host = x_values.copy_to_host_vec()?;
    for (index, expected) in [5.0, 4.0, 3.0, 2.0, 1.0].into_iter().enumerate() {
        let value = x_host[index];
        println!("selected batch: x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
