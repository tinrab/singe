use singe_cublas::{blas::level3::cgemm3m, context::Context, error::Result, types::Operation};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex32,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[
        Complex32::new(1.1, 1.2),
        Complex32::new(3.5, 3.6),
        Complex32::new(2.3, 2.4),
        Complex32::new(4.7, 4.8),
    ])?;
    let b = DeviceMemory::from_slice(&[
        Complex32::new(1.1, 1.2),
        Complex32::new(3.5, 3.6),
        Complex32::new(2.3, 2.4),
        Complex32::new(4.7, 4.8),
    ])?;
    let mut c = DeviceMemory::<Complex32>::zeroes(4)?;

    cgemm3m(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &Complex32::new(1.0, 1.0),
        &a,
        2,
        &b,
        2,
        &Complex32::new(0.0, 0.0),
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    singe_core::assert_complex_close!(result[0], Complex32::new(-20.14, 18.50), 1.0e-3);
    singe_core::assert_complex_close!(result[1], Complex32::new(-43.18, 40.58), 1.0e-3);
    singe_core::assert_complex_close!(result[2], Complex32::new(-28.78, 26.66), 1.0e-3);
    singe_core::assert_complex_close!(result[3], Complex32::new(-63.34, 60.26), 1.0e-3);

    println!("gemm3m result: {result:?}");
    Ok(())
}
