use singe_cublas::{blas::level2::hpmv, context::Context, scalar::Scalar, types::FillMode};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let ap = DeviceMemory::from_slice(
        &cuda,
        &[
            Complex64::new(1.1, 1.2),
            Complex64::new(2.3, 2.4),
            Complex64::new(4.7, 4.8),
        ],
    )?;
    let x = DeviceMemory::from_slice(&cuda, &[Complex64::new(5.1, 6.2), Complex64::new(7.3, 8.4)])?;
    let mut y = DeviceMemory::<Complex64>::zeroed(&cuda, 2)?;

    hpmv(
        &cublas,
        FillMode::Upper,
        2,
        Scalar::host(&Complex64::new(1.0, 1.0)),
        &ap,
        &x,
        1,
        Scalar::host(&Complex64::new(0.0, 0.0)),
        &mut y,
        1,
    )?;

    println!("hpmv result: {:?}", y.copy_to_host_vec()?);
    Ok(())
}
