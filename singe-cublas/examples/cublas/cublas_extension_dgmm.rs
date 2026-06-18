use singe_cublas::{blas::level3::ddgmm, context::Context, error::Result, types::SideMode};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let x = DeviceMemory::from_slice(&[5.0_f64, 6.0])?;
    let mut c = DeviceMemory::<f64>::zeroes(4)?;

    ddgmm(&cublas, SideMode::Left, 2, 2, &a, 2, &x, 1, &mut c, 2)?;

    let result = c.copy_to_host_vec()?;
    assert_eq!(result, vec![5.0, 18.0, 10.0, 24.0]);
    println!("dgmm result: {result:?}");
    Ok(())
}
