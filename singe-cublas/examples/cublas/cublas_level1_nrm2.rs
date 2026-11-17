use singe_cublas::{blas::level1::nrm2, context::Context, scalar::ScalarMut};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut result = 0.0_f64;

    nrm2(&cublas, 4, &x, 1, ScalarMut::host(&mut result))?;
    assert!((result - 5.477_225_575_051_661).abs() < 1.0e-12);

    println!("nrm2: {result}");
    Ok(())
}
