mod common;

use std::slice;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{
    config::Config,
    error::Result,
    types::{ConfigParameter, DataParameter, Phase},
};

use crate::common::{create_context, dense_matrix, dense_output, execute, spd_csr_matrix};

fn main() -> Result<()> {
    let (cuda, stream, context) = create_context()?;
    println!("cuDSS example: solving with hybrid memory mode");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&cuda, &[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&cuda, &[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&cuda, &[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let mut x_values = DeviceMemory::<f64>::create_uninit(&cuda, (n * nrhs) as usize)?;

    let mut config = Config::create()?;
    let mut data = context.create_data()?;

    let hybrid_memory_mode = 1i32;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe { config.set_raw(ConfigParameter::HybridMemoryMode, &hybrid_memory_mode) }?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_output(n, nrhs, n, x_values.output())?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        execute(
            slice::from_ref(&stream),
            &context,
            Phase::ANALYSIS,
            &config,
            &mut data,
            &a,
            &mut x,
            &b,
        )
    }?;

    // SAFETY: the value pointer and size describe a local of the parameter type.
    let min_device_memory: i64 = unsafe { data.get_raw(DataParameter::HybridDeviceMemoryMin) }?;
    println!("minimum device memory for hybrid mode: {min_device_memory} bytes");

    let device_memory_limit = 40_i64 * 1024;
    // SAFETY: the value pointer and size describe a local of the parameter type.
    unsafe {
        config.set_raw(
            ConfigParameter::HybridDeviceMemoryLimit,
            &device_memory_limit,
        )
    }?;

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
            &b,
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
            &b,
        )
    }?;
    cuda.synchronize()?;

    drop(x);
    // SAFETY: checked completed production initialized this compact output.
    let x_values = unsafe { x_values.assume_init() };
    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        println!("x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
