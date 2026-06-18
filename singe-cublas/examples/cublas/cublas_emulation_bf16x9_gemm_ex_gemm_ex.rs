use singe_cublas::{
    blas::level3::gemm_ex,
    context::Context,
    error::Result,
    types::{ComputeType, EmulationStrategy, GemmAlgorithm, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ])?;
    let b = DeviceMemory::from_slice(&[
        1.0_f32, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
    ])?;
    let mut c_native = DeviceMemory::<f32>::zeroes(16)?;
    let mut c_emulated = DeviceMemory::<f32>::zeroes(16)?;

    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        4,
        4,
        4,
        &1.0_f32,
        &a,
        DataType::F32,
        4,
        &b,
        DataType::F32,
        4,
        &0.0_f32,
        &mut c_native,
        DataType::F32,
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
        &1.0_f32,
        &a,
        DataType::F32,
        4,
        &b,
        DataType::F32,
        4,
        &0.0_f32,
        &mut c_emulated,
        DataType::F32,
        4,
        ComputeType::F32EmulatedBf16x9,
        GemmAlgorithm::Default,
    )?;

    println!("gemm_ex native: {:?}", c_native.copy_to_host_vec()?);
    println!("gemm_ex bf16x9: {:?}", c_emulated.copy_to_host_vec()?);
    Ok(())
}
