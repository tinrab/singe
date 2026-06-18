use singe_cublas::{blas::level2::dgemv, context::Context, error::Result, types::Operation};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let x = DeviceMemory::from_slice(&[5.0_f64, 6.0])?;
    let mut y = DeviceMemory::<f64>::zeroes(2)?;

    dgemv(
        &cublas,
        Operation::NonTranspose,
        2,
        2,
        &1.0,
        &a,
        2,
        &x,
        1,
        &0.0,
        &mut y,
        1,
    )?;

    println!("gemv result: {:?}", y.copy_to_host_vec()?);
    Ok(())
}
