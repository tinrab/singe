use singe_cublas::{blas::level2::gemv, context::Context, scalar::Scalar, types::Operation};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
    let x = DeviceMemory::from_slice(&cuda, &[5.0_f64, 6.0])?;
    let mut y = DeviceMemory::<f64>::zeroed(&cuda, 2)?;

    gemv(
        &cublas,
        Operation::NonTranspose,
        2,
        2,
        Scalar::host(&1.0_f64),
        &a,
        2,
        &x,
        1,
        Scalar::host(&0.0_f64),
        &mut y,
        1,
    )?;

    println!("gemv result: {:?}", y.copy_to_host_vec()?);
    Ok(())
}
