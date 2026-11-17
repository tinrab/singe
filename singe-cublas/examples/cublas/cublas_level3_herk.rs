use singe_cublas::{
    blas::level3::herk,
    context::Context,
    scalar::Scalar,
    types::{FillMode, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(
        &cuda,
        &[
            Complex64::new(1.1, 1.2),
            Complex64::new(3.5, 3.6),
            Complex64::new(2.3, 2.4),
            Complex64::new(4.7, 4.8),
        ],
    )?;
    let mut c = DeviceMemory::<Complex64>::zeroed(&cuda, 4)?;

    herk(
        &cublas,
        FillMode::Upper,
        Operation::NonTranspose,
        2,
        2,
        Scalar::host(&1.0_f64),
        &a,
        2,
        Scalar::host(&0.0_f64),
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    singe_core::assert_complex_close!(result[0], Complex64::new(13.70, 0.0), 1.0e-10);
    singe_core::assert_complex_close!(result[1], Complex64::new(0.0, 0.0), 1.0e-10);
    singe_core::assert_complex_close!(result[2], Complex64::new(30.50, 0.48), 1.0e-10);
    singe_core::assert_complex_close!(result[3], Complex64::new(70.34, 0.0), 1.0e-10);

    println!("herk result: {result:?}");
    Ok(())
}
