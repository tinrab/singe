use singe_cublas::{blas::level1::axpy_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
    let alpha = 2.1_f64;

    axpy_ex(
        &cublas,
        &alpha,
        DataType::F64,
        &x,
        DataType::F64,
        1,
        &mut y,
        DataType::F64,
        1,
        DataType::F64,
    )?;

    let result = y.copy_to_host_vec()?;
    for (actual, expected) in result.iter().zip([7.1, 10.2, 13.3, 16.4]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    println!("axpy result: {result:?}");
    Ok(())
}
