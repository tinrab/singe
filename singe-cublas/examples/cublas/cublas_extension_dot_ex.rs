use singe_cublas::{blas::level1::dot_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let y = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
    let mut result = 0.0_f64;

    dot_ex(
        &cublas,
        &x,
        DataType::F64,
        1,
        &y,
        DataType::F64,
        1,
        &mut result,
        DataType::F64,
        DataType::F64,
    )?;
    assert_eq!(result, 70.0);

    println!("dot_ex result: {result}");
    Ok(())
}
