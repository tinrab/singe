use singe_cublas::{
    blas::extension::gemm_ex,
    context::Context,
    scalar::Scalar,
    types::{ComputeType, EmulationStrategy, GemmAlgorithm, Operation},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let mut cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(
        &cuda,
        &[
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
    )?;
    let b = DeviceMemory::from_slice(
        &cuda,
        &[
            1.0_f32, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0,
            16.0,
        ],
    )?;
    let mut c_native = DeviceMemory::<f32>::zeroed(&cuda, 16)?;
    let mut c_emulated = DeviceMemory::<f32>::zeroed(&cuda, 16)?;

    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        4,
        4,
        4,
        Scalar::host(&1.0_f32),
        &a,
        4,
        &b,
        4,
        Scalar::host(&0.0_f32),
        &mut c_native,
        4,
        ComputeType::F32,
        GemmAlgorithm::Default,
    )?;

    cublas.set_emulation_strategy(EmulationStrategy::Eager)?;
    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        4,
        4,
        4,
        Scalar::host(&1.0_f32),
        &a,
        4,
        &b,
        4,
        Scalar::host(&0.0_f32),
        &mut c_emulated,
        4,
        ComputeType::F32EmulatedBf16x9,
        GemmAlgorithm::Default,
    )?;

    println!("gemm_ex native: {:?}", c_native.copy_to_host_vec()?);
    println!("gemm_ex bf16x9: {:?}", c_emulated.copy_to_host_vec()?);
    Ok(())
}
