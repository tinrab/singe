use singe_cublas::{blas::level2::zhpmv, context::Context, error::Result, types::FillMode};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let ap = DeviceMemory::from_slice(&[
        Complex64::new(1.1, 1.2),
        Complex64::new(2.3, 2.4),
        Complex64::new(4.7, 4.8),
    ])?;
    let x = DeviceMemory::from_slice(&[Complex64::new(5.1, 6.2), Complex64::new(7.3, 8.4)])?;
    let mut y = DeviceMemory::<Complex64>::zeroes(2)?;

    zhpmv(
        &cublas,
        FillMode::Upper,
        2,
        &Complex64::new(1.0, 1.0),
        &ap,
        &x,
        1,
        &Complex64::new(0.0, 0.0),
        &mut y,
        1,
    )?;

    println!("hpmv result: {:?}", y.copy_to_host_vec()?);
    Ok(())
}
