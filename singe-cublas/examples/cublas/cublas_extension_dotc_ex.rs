use singe_cublas::{blas::level1::dotc_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::Complex64,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&[
        Complex64::new(1.1, 1.2),
        Complex64::new(2.3, 2.4),
        Complex64::new(3.5, 3.6),
        Complex64::new(4.7, 4.8),
    ])?;
    let y = DeviceMemory::from_slice(&[
        Complex64::new(5.1, 5.2),
        Complex64::new(6.3, 6.4),
        Complex64::new(7.5, 7.6),
        Complex64::new(8.7, 8.8),
    ])?;
    let mut result = Complex64::new(0.0, 0.0);

    dotc_ex(
        &cublas,
        &x,
        DataType::ComplexF64,
        1,
        &y,
        DataType::ComplexF64,
        1,
        &mut result,
        DataType::ComplexF64,
        DataType::ComplexF64,
    )?;

    assert!((result.re - 178.44).abs() < 1.0e-9);
    assert!((result.im + 1.60).abs() < 1.0e-9);

    println!("dotc_ex result: {result:?}");
    Ok(())
}
