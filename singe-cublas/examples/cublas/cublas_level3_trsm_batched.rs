use singe_cublas::{
    blas::level3::trsm_batched,
    context::Context,
    scalar::Scalar,
    types::{DiagonalType, FillMode, Operation, SideMode},
};
use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    view::{DeviceInputView, DeviceOutputView},
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a0 = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0])?;
    let a1 = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut b0 = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let mut b1 = DeviceMemory::from_slice(&cuda, &[9.0_f64, 11.0, 10.0, 12.0])?;

    trsm_batched(
        &cublas,
        SideMode::Left,
        FillMode::Upper,
        Operation::NonTranspose,
        DiagonalType::NonUnit,
        2,
        2,
        Scalar::host(&1.0),
        &[
            DeviceInputView::from_buffer(&a0),
            DeviceInputView::from_buffer(&a1),
        ],
        2,
        &mut [
            DeviceOutputView::from_output(&mut b0),
            DeviceOutputView::from_output(&mut b1),
        ],
        2,
    )?;

    assert_eq!(b0.copy_to_host_vec()?, vec![1.5, 1.75, 2.0, 2.0]);
    assert_eq!(b1.copy_to_host_vec()?, vec![0.15, 1.375, 0.2, 1.5]);

    println!("trsm_batched b0: {:?}", b0.copy_to_host_vec()?);
    println!("trsm_batched b1: {:?}", b1.copy_to_host_vec()?);
    Ok(())
}
