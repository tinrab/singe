use singe_cublas::{blas::level1::copy_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::<f64>::zeroes(4)?;

    copy_ex(&cublas, &x, DataType::F64, 1, &mut y, DataType::F64, 1)?;

    let result = y.copy_to_host_vec()?;
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);

    println!("copy result: {result:?}");
    Ok(())
}
