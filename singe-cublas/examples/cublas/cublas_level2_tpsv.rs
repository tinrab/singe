use singe_cublas::{
    blas::level2::dtpsv,
    context::Context,
    error::Result,
    types::{DiagonalType, FillMode, Operation},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let ap = DeviceMemory::from_slice(&[1.0_f64, 2.0, 4.0])?;
    let mut x = DeviceMemory::from_slice(&[5.0_f64, 6.0])?;

    dtpsv(
        &cublas,
        FillMode::Upper,
        Operation::NonTranspose,
        DiagonalType::NonUnit,
        2,
        &ap,
        &mut x,
        1,
    )?;

    println!("tpsv result: {:?}", x.copy_to_host_vec()?);
    Ok(())
}
