mod common;

use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};
use singe_cudss::{
    config::Config,
    context::Context,
    error::Result,
    types::{ConfigParameter, DataParameter},
};

use crate::common::*;

fn main() -> Result<()> {
    let device_count = Device::count()?;
    if device_count < 2 {
        println!(
            "Example SKIPPED: multi-GPU mode needs at least 2 visible CUDA devices, got {device_count}"
        );
        return Ok(());
    }

    println!("cuDSS example: solving with multi-GPU mode on {device_count} devices");

    let n = 5i64;
    let nrhs = 1i64;
    let nnz = 8i64;

    let row_offsets = DeviceMemory::from_slice(&[0_i32, 2, 4, 6, 7, 8])?;
    let col_indices = DeviceMemory::from_slice(&[0_i32, 2, 1, 2, 2, 4, 3, 4])?;
    let a_values = DeviceMemory::from_slice(&[4.0f64, 1.0, 3.0, 2.0, 5.0, 1.0, 1.0, 2.0])?;
    let b_values = DeviceMemory::from_slice(&[7.0f64, 12.0, 25.0, 4.0, 13.0])?;
    let x_values = DeviceMemory::<f64>::zeroes((n * nrhs) as usize)?;

    let devices: Vec<_> = (0..device_count).collect();
    let primary_context = CudaContext::create_for_device(Device::new(devices[0]))?;
    let device_contexts = devices
        .iter()
        .copied()
        .map(|device| CudaContext::create_for_device(Device::new(device)))
        .collect::<singe_cuda::error::Result<Vec<_>>>()?;
    let streams = device_contexts
        .iter()
        .map(|context| context.create_stream())
        .collect::<singe_cuda::error::Result<Vec<_>>>()?;

    let context = Context::create_multi_gpu(&primary_context, &devices)?;
    let stream_refs: Vec<_> = streams.iter().collect();
    context.set_multi_gpu_streams(&stream_refs)?;

    let mut config = Config::create()?;
    config.set(ConfigParameter::DeviceCount, &device_count)?;
    config.set_slice(ConfigParameter::DeviceIndices, &devices)?;
    let mut data = context.create_data()?;

    let b = dense_matrix(n, nrhs, n, &b_values)?;
    let mut x = dense_matrix(n, nrhs, n, &x_values)?;
    let a = spd_csr_matrix(n, n, nnz, &row_offsets, &col_indices, &a_values)?;

    solve(&context, &config, &mut data, &a, &mut x, &b)?;

    for device in &devices {
        Device::new(*device).set_current()?;
        let min_memory: i64 = data.get(DataParameter::HybridDeviceMemoryMin)?;
        println!("device {device}: hybrid min device memory = {min_memory} bytes");
    }
    Device::new(devices[0]).set_current()?;

    for stream in &streams {
        stream.synchronize()?;
    }

    let x = x_values.copy_to_host_vec()?;
    for (index, value) in x.iter().copied().enumerate() {
        let expected = (index + 1) as f64;
        println!("x[{index}] = {value:.4}, expected {expected:.4}");
        assert!((value - expected).abs() <= 2.0e-15);
    }

    println!("Example PASSED");
    Ok(())
}
