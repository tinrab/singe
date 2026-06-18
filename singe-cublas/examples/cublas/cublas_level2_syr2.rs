use singe_cublas::{blas::level2::dsyr2, context::Context, error::Result, types::FillMode};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let mut a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let x = DeviceMemory::from_slice(&[5.0_f64, 6.0])?;
    let y = DeviceMemory::from_slice(&[7.0_f64, 8.0])?;

    dsyr2(&cublas, FillMode::Upper, 2, &1.0, &x, 1, &y, 1, &mut a, 2)?;

    println!("syr2 result: {:?}", a.copy_to_host_vec()?);
    Ok(())
}
