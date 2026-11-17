use singe_cublas::{
    blas::level3::{GemmGroup, gemm_grouped_batched},
    context::Context,
    gemm::GemmShape,
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
    let a2 = DeviceMemory::from_slice(&cuda, &[1.0_f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0])?;
    let b0 = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0])?;
    let b1 = DeviceMemory::from_slice(&cuda, &[9.0_f64, 11.0, 10.0, 12.0])?;
    let b2 =
        DeviceMemory::from_slice(&cuda, &[4.0_f64, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0])?;
    let mut c0 = DeviceMemory::<f64>::zeroed(&cuda, 4)?;
    let mut c1 = DeviceMemory::<f64>::zeroed(&cuda, 4)?;
    let mut c2 = DeviceMemory::<f64>::zeroed(&cuda, 9)?;

    gemm_grouped_batched(
        &cublas,
        &[
            GemmGroup {
                shape: GemmShape::new(2, 2, 2),
                alpha: 1.0,
                beta: 0.0,
                lda: 2,
                ldb: 2,
                ldc: 2,
                count: 2,
            },
            GemmGroup {
                shape: GemmShape::new(3, 3, 3),
                alpha: 1.0,
                beta: 0.0,
                lda: 3,
                ldb: 3,
                ldc: 3,
                count: 1,
            },
        ],
        &[
            DeviceInputView::from_buffer(&a0),
            DeviceInputView::from_buffer(&a1),
            DeviceInputView::from_buffer(&a2),
        ],
        &[
            DeviceInputView::from_buffer(&b0),
            DeviceInputView::from_buffer(&b1),
            DeviceInputView::from_buffer(&b2),
        ],
        &mut [
            DeviceOutputView::from_output(&mut c0),
            DeviceOutputView::from_output(&mut c1),
            DeviceOutputView::from_output(&mut c2),
        ],
    )?;

    assert_eq!(c0.copy_to_host_vec()?, vec![19.0, 43.0, 22.0, 50.0]);
    assert_eq!(c1.copy_to_host_vec()?, vec![111.0, 151.0, 122.0, 166.0]);
    assert_eq!(
        c2.copy_to_host_vec()?,
        vec![48.0, 111.0, 174.0, 54.0, 126.0, 198.0, 60.0, 141.0, 222.0]
    );

    println!("grouped batched c0: {:?}", c0.copy_to_host_vec()?);
    println!("grouped batched c1: {:?}", c1.copy_to_host_vec()?);
    println!("grouped batched c2: {:?}", c2.copy_to_host_vec()?);
    Ok(())
}
