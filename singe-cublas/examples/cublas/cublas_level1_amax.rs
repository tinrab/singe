use singe_cublas::{blas::level1::iamax, context::Context, scalar::ScalarMut};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let x = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0, 4.0])?;
    let mut index = 0_i64;

    iamax(&cublas, 4, &x, 1, ScalarMut::host(&mut index))?;
    assert_eq!(index, 4);

    println!("iamax index: {index}");
    Ok(())
}
