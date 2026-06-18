use singe_cublas::{blas::level1::rot_ex, context::Context, error::Result};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
    let c = 2.1_f64;
    let s = 1.2_f64;

    rot_ex(
        &cublas,
        &mut x,
        DataType::F64,
        1,
        &mut y,
        DataType::F64,
        1,
        &c,
        &s,
        DataType::F64,
        DataType::F64,
    )?;

    let x_result = x.copy_to_host_vec()?;
    let y_result = y.copy_to_host_vec()?;
    for (actual, expected) in x_result.iter().zip([8.1, 11.4, 14.7, 18.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    for (actual, expected) in y_result.iter().zip([9.3, 10.2, 11.1, 12.0]) {
        assert!((actual - expected).abs() < 1.0e-12);
    }

    println!("rot_ex x: {x_result:?}");
    println!("rot_ex y: {y_result:?}");
    Ok(())
}
