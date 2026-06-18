use singe_cublas::{blas::level2::zhpr2, context::Context, error::Result, types::FillMode};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let mut ap = DeviceMemory::from_slice(&[
        Complex64::new(1.1, 1.2),
        Complex64::new(3.5, 3.6),
        Complex64::new(4.7, 4.8),
    ])?;
    let x = DeviceMemory::from_slice(&[Complex64::new(5.1, 6.2), Complex64::new(7.3, 8.4)])?;
    let y = DeviceMemory::from_slice(&[Complex64::new(1.1, 2.2), Complex64::new(3.3, 4.4)])?;

    zhpr2(
        &cublas,
        FillMode::Upper,
        2,
        &Complex64::new(1.0, 1.0),
        &x,
        1,
        &y,
        1,
        &mut ap,
    )?;

    println!("hpr2 result: {:?}", ap.copy_to_host_vec()?);
    Ok(())
}
