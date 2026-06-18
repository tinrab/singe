use singe_cublas::{
    blas::level3::dsyrk,
    context::Context,
    error::Result,
    types::{FillMode, Operation},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 3.0, 4.0])?;
    let mut c = DeviceMemory::<f64>::zeroes(4)?;

    dsyrk(
        &cublas,
        FillMode::Upper,
        Operation::Transpose,
        2,
        2,
        &1.0,
        &a,
        2,
        &0.0,
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert_eq!(result, vec![10.0, 0.0, 15.0, 25.0]);
    println!("syrk result: {result:?}");
    Ok(())
}
