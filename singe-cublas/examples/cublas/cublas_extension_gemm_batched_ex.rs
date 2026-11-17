use singe_cublas::{
    blas::extension::gemm_batched_ex,
    context::Context,
    scalar::Scalar,
    types::{ComputeType, GemmAlgorithm, Operation},
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
    let b0 = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let b1 = DeviceMemory::from_slice(&cuda, &[9.0_f64, 11.0, 10.0, 12.0])?;
    let mut c0 = DeviceMemory::<f64>::zeroed(&cuda, 4)?;
    let mut c1 = DeviceMemory::<f64>::zeroed(&cuda, 4)?;

    gemm_batched_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0_f64),
        &[
            DeviceInputView::from_buffer(&a0),
            DeviceInputView::from_buffer(&a1),
        ],
        2,
        &[
            DeviceInputView::from_buffer(&b0),
            DeviceInputView::from_buffer(&b1),
        ],
        2,
        Scalar::host(&0.0_f64),
        &mut [
            DeviceOutputView::from_output(&mut c0),
            DeviceOutputView::from_output(&mut c1),
        ],
        2,
        ComputeType::F64,
        GemmAlgorithm::Default,
    )?;

    assert_eq!(c0.copy_to_host_vec()?, vec![19.0, 43.0, 22.0, 50.0]);
    assert_eq!(c1.copy_to_host_vec()?, vec![111.0, 151.0, 122.0, 166.0]);

    println!("gemm_batched_ex c0: {:?}", c0.copy_to_host_vec()?);
    println!("gemm_batched_ex c1: {:?}", c1.copy_to_host_vec()?);
    Ok(())
}
