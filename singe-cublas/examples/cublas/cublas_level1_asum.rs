use singe_cublas::{blas::level1::asum, context::Context, scalar::ScalarMut};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut result = 0.0_f64;

    asum(&cublas, 4, &x, 1, ScalarMut::host(&mut result))?;
    assert_eq!(result, 10.0);

    println!("asum: {result}");
    Ok(())
}
