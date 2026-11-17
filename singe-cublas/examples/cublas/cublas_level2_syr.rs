use singe_cublas::{blas::level2::syr, context::Context, scalar::Scalar, types::FillMode};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let mut a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 3.0, 4.0])?;
    let x = DeviceMemory::from_slice(&cuda, &[5.0_f64, 6.0])?;

    syr(
        &cublas,
        FillMode::Upper,
        2,
        Scalar::host(&1.0_f64),
        &x,
        1,
        &mut a,
        2,
    )?;

    println!("syr result: {:?}", a.copy_to_host_vec()?);
    Ok(())
}
