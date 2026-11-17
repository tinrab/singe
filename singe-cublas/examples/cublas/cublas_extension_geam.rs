use singe_cublas::{blas::level3::geam, context::Context, scalar::Scalar, types::Operation};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroed(&cuda, 4)?;

    geam(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        Scalar::host(&1.0),
        &a,
        2,
        Scalar::host(&2.0),
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
