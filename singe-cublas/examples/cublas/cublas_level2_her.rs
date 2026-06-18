use singe_cublas::{blas::level2::zher, context::Context, error::Result, types::FillMode};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let mut a = DeviceMemory::from_slice(&[
        Complex64::new(1.1, 1.2),
        Complex64::new(3.5, 3.6),
        Complex64::new(2.3, 2.4),
        Complex64::new(4.7, 4.8),
    ])?;
    let x = DeviceMemory::from_slice(&[Complex64::new(5.1, 6.2), Complex64::new(7.3, 8.4)])?;

    zher(&cublas, FillMode::Upper, 2, &1.0, &x, 1, &mut a, 2)?;

    println!("her result: {:?}", a.copy_to_host_vec()?);
    Ok(())
}
