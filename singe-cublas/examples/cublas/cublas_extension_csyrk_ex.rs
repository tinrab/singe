use singe_cublas::{
    blas::extension::csyrk_ex,
    context::Context,
    scalar::Scalar,
    types::{FillMode, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory, types::Complex32,
};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(
        &cuda,
        &[
            Complex32::new(1.1, 1.2),
            Complex32::new(3.5, 3.6),
            Complex32::new(3.5, 3.6),
            Complex32::new(4.7, 4.8),
        ],
    )?;
    let mut c = DeviceMemory::<Complex32>::zeroed(&cuda, 4)?;

    csyrk_ex(
        &cublas,
        FillMode::Upper,
        Operation::NonTranspose,
        2,
        2,
        Scalar::host(&Complex32::new(1.0, 1.0)),
        &a,
        2,
        Scalar::host(&Complex32::new(0.0, 0.0)),
        &mut c,
        2,
    )?;

    let result = c.copy_to_host_vec()?;
    assert!(
        result
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    println!("csyrk_ex result: {result:?}");
    Ok(())
}
