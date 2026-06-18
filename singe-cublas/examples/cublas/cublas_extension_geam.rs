use singe_cublas::{blas::level3::dgeam, context::Context, error::Result, types::Operation};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroes(4)?;

    dgeam(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        &1.0,
        &a,
        2,
        &2.0,
        &b,
        2,
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert_eq!(result, vec![11.0, 17.0, 14.0, 20.0]);
    println!("geam result: {result:?}");
    Ok(())
}
