mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudss::{
    config::Config,
    error::Result,
    types::{ConfigParameter, DataParameter, Phase},
};

use crate::common::*;

fn main() -> Result<()> {
    println!("cuDSS example: solving with hybrid memory mode");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let x_values = DeviceMemory::<f64>::zeroes((n * nrhs) as usize)?;

    let (cuda, _stream, context) = create_context()?;
    let mut config = Config::create()?;
    let mut data = context.create_data()?;

    let hybrid_memory_mode = 1i32;
    config.set(ConfigParameter::HybridMemoryMode, &hybrid_memory_mode)?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    context.execute(Phase::ANALYSIS, &config, &mut data, &a, &mut x, &b)?;

    let min_device_memory: i64 = data.get(DataParameter::HybridDeviceMemoryMin)?;
    println!("minimum device memory for hybrid mode: {min_device_memory} bytes");

    let device_memory_limit = 40_i64 * 1024;
    config.set(
        ConfigParameter::HybridDeviceMemoryLimit,
        &device_memory_limit,
    )?;

    context.execute(Phase::FACTORIZATION, &config, &mut data, &a, &mut x, &b)?;
    context.execute(Phase::SOLVE, &config, &mut data, &a, &mut x, &b)?;
    cuda.synchronize()?;

    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        println!("x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
