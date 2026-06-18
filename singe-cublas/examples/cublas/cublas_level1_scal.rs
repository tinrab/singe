use singe_cublas::{blas::level1::scal_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let alpha = 2.2_f64;

    scal_ex(
        &cublas,
        &alpha,
        DataType::F64,
        &mut x,
        DataType::F64,
        1,
        DataType::F64,
    )?;

    let result = x.copy_to_host_vec()?;
    for (actual, expected) in result.iter().zip([2.2, 4.4, 6.6, 8.8]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    println!("scaled vector: {result:?}");
    Ok(())
}
