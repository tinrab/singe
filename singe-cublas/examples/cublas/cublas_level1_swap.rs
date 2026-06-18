use singe_cublas::{blas::level1::swap_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;

    swap_ex(&cublas, &mut x, DataType::F64, 1, &mut y, DataType::F64, 1)?;

    let x_result = x.copy_to_host_vec()?;
    let y_result = y.copy_to_host_vec()?;
    assert_eq!(x_result, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y_result, vec![1.0, 2.0, 3.0, 4.0]);

    println!("swap x: {x_result:?}");
    println!("swap y: {y_result:?}");
    Ok(())
}
