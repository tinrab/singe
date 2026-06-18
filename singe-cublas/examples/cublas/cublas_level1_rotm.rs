use singe_cublas::{blas::level1::drotm, context::Context, error::Result};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut x = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
    let param = [1.0_f64, 5.0, 6.0, 7.0, 8.0];

    drotm(&cublas, &mut x, 1, &mut y, 1, &param)?;

    let x_result = x.copy_to_host_vec()?;
    let y_result = y.copy_to_host_vec()?;
    assert_eq!(x_result, vec![10.0, 16.0, 22.0, 28.0]);
    assert_eq!(y_result, vec![39.0, 46.0, 53.0, 60.0]);

    println!("rotm x: {x_result:?}");
    println!("rotm y: {y_result:?}");
    Ok(())
}
