use singe_cublas::{blas::level2::ger, context::Context, scalar::Scalar};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let mut a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
    let x = DeviceMemory::from_slice(&cuda, &[5.0_f64, 6.0])?;
    let y = DeviceMemory::from_slice(&cuda, &[7.0_f64, 8.0])?;

    ger(
        &cublas,
        2,
        2,
        Scalar::host(&2.0_f64),
        &x,
        1,
        &y,
        1,
        &mut a,
        2,
    )?;

    println!("ger result: {:?}", a.copy_to_host_vec()?);
    Ok(())
}
