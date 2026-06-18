use singe_cublas::{
    blas::level3::dtrsm,
    context::Context,
    error::Result,
    types::{DiagonalType, FillMode, Operation, SideMode},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let mut b = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;

    dtrsm(
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
        &mut b,
        2,
    )?;

    let result = b.copy_to_host_vec()?;
    assert_eq!(result, vec![1.5, 1.75, 2.0, 2.0]);
    println!("trsm result: {result:?}");
    Ok(())
}
