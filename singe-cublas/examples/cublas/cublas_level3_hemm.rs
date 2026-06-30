use singe_cublas::{
    blas::level3::zhemm,
    context::Context,
    error::Result,
    types::{FillMode, SideMode},
};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[
        Complex64::new(1.1, 1.2),
        Complex64::new(3.5, 3.6),
        Complex64::new(2.3, 2.4),
        Complex64::new(4.7, 4.8),
    ])?;
    let b = DeviceMemory::from_slice(&[
        Complex64::new(1.1, 1.2),
        Complex64::new(3.5, 3.6),
        Complex64::new(2.3, 2.4),
        Complex64::new(4.7, 4.8),
    ])?;
    let mut c = DeviceMemory::<Complex64>::zeroes(4)?;

    zhemm(
        &cublas,
        SideMode::Left,
        FillMode::Upper,
        2,
        2,
        &Complex64::new(1.0, 1.0),
        &a,
        2,
        &b,
        2,
        &Complex64::new(0.0, 0.0),
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    singe_core::assert_complex_close!(result[0], Complex64::new(-17.38, 18.62), 1.0e-10);
    singe_core::assert_complex_close!(result[1], Complex64::new(4.82, 38.90), 1.0e-10);
    singe_core::assert_complex_close!(result[2], Complex64::new(-23.14, 26.78), 1.0e-10);
    singe_core::assert_complex_close!(result[3], Complex64::new(10.58, 55.70), 1.0e-10);

    println!("hemm result: {result:?}");
    Ok(())
}
