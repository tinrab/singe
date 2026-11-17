use singe_cublas::{
    blas::level3::gemm_strided_batched, context::Context, scalar::Scalar, types::Operation,
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0])?;
    let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0])?;
    let mut c = DeviceMemory::<f64>::zeroed(&cuda, 8)?;

    gemm_strided_batched(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        Scalar::host(&1.0),
        &a,
        2,
        4,
        &b,
        2,
        4,
        Scalar::host(&0.0),
        &mut c,
        2,
        4,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert_eq!(
        result,
        vec![19.0, 43.0, 22.0, 50.0, 111.0, 151.0, 122.0, 166.0]
    );

    println!("gemm_strided_batched result: {result:?}");
    Ok(())
}
