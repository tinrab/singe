use singe_cublas::{
    blas::level3::dtrmm,
    context::Context,
    error::Result,
    types::{DiagonalType, FillMode, Operation, SideMode},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroes(4)?;

    dtrmm(
        &cublas,
        SideMode::Left,
        FillMode::Upper,
        Operation::NonTranspose,
        DiagonalType::NonUnit,
        2,
        2,
        &1.0,
        &a,
        2,
        &b,
        2,
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert_eq!(result, vec![19.0, 28.0, 22.0, 32.0]);
    println!("trmm result: {result:?}");
    Ok(())
}
