use singe_cublas::{blas::level3::dtpttr, context::Context, error::Result, types::FillMode};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    // Packed upper-triangular storage for [[1, 3], [0, 2]].
    let ap = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0])?;
    let mut a = DeviceMemory::from_slice(&[0.0_f64, 0.0, 0.0, 0.0])?;

    dtpttr(&cublas, FillMode::Upper, 2, &ap, &mut a, 2)?;

    let result = a.copy_to_host_vec()?;
    assert_eq!(result, vec![1.0, 0.0, 3.0, 2.0]);
    println!("tpttr result: {result:?}");
    Ok(())
}
