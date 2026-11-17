use singe_cublas::{blas::level1::copy, context::Context};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut y = DeviceMemory::<f64>::zeroed(&cuda, 4)?;

    copy(&cublas, 4, &x, 1, &mut y, 1)?;

    let result = y.copy_to_host_vec()?;
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);

    println!("copy result: {result:?}");
    Ok(())
}
