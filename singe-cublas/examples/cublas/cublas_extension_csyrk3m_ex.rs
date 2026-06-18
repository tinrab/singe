use singe_cublas::{
    blas::level3::csyrk3m_ex,
    context::Context,
    error::Result,
    types::{FillMode, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::Complex32,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[
        Complex32::new(1.1, 1.2),
        Complex32::new(3.5, 3.6),
        Complex32::new(3.5, 3.6),
        Complex32::new(4.7, 4.8),
    ])?;
    let mut c = DeviceMemory::<Complex32>::zeroes(4)?;

    csyrk3m_ex(
        &cublas,
        FillMode::Upper,
        Operation::NonTranspose,
        2,
        2,
        &Complex32::new(1.0, 1.0),
        &a,
        DataType::ComplexF32,
        2,
        &Complex32::new(0.0, 0.0),
        &mut c,
        DataType::ComplexF32,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert_complex32_close(result[0], Complex32::new(-28.78, 26.90));
    assert_complex32_close(result[1], Complex32::new(0.0, 0.0));
    assert_complex32_close(result[2], Complex32::new(-43.18, 40.58));
    assert_complex32_close(result[3], Complex32::new(-71.98, 68.66));

    println!("csyrk3m_ex result: {result:?}");
    Ok(())
}

fn assert_complex32_close(actual: Complex32, expected: Complex32) {
    assert!((actual.re - expected.re).abs() < 1e-3);
    assert!((actual.im - expected.im).abs() < 1e-3);
}
