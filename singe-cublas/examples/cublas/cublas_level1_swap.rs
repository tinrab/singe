use singe_cublas::{blas::level1::swap, context::Context};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut x = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::from_slice(&cuda, &[5.0_f64, 6.0, 7.0, 8.0])?;

    swap(&cublas, 4, &mut x, 1, &mut y, 1)?;

    let x_result = x.copy_to_host_vec()?;
    let y_result = y.copy_to_host_vec()?;
    assert_eq!(x_result, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y_result, vec![1.0, 2.0, 3.0, 4.0]);

    println!("swap x: {x_result:?}");
    println!("swap y: {y_result:?}");
    Ok(())
}
