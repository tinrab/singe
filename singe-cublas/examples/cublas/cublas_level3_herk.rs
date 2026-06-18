use singe_cublas::{
    blas::level3::zherk,
    context::Context,
    error::Result,
    types::{FillMode, Operation},
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
    let mut c = DeviceMemory::<Complex64>::zeroes(4)?;

    zherk(
        &cublas,
        FillMode::Upper,
        Operation::NonTranspose,
        2,
        2,
        &1.0,
        &a,
        2,
        &0.0,
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert_complex64_close(result[0], Complex64::new(13.70, 0.0));
    assert_complex64_close(result[1], Complex64::new(0.0, 0.0));
    assert_complex64_close(result[2], Complex64::new(30.50, 0.48));
    assert_complex64_close(result[3], Complex64::new(70.34, 0.0));

    println!("herk result: {result:?}");
    Ok(())
}

fn assert_complex64_close(actual: Complex64, expected: Complex64) {
    assert!((actual.re - expected.re).abs() < 1e-10);
    assert!((actual.im - expected.im).abs() < 1e-10);
}
