use singe_cublas::{blas::level3::dtrttp, context::Context, error::Result, types::FillMode};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 0.0, 3.0, 2.0])?;
    let mut ap = DeviceMemory::<f64>::zeroes(3)?;

    dtrttp(&cublas, FillMode::Upper, 2, &a, 2, &mut ap)?;

    let result = ap.copy_to_host_vec()?;
    assert_eq!(result, vec![1.0, 3.0, 2.0]);
    println!("trttp result: {result:?}");
    Ok(())
}
