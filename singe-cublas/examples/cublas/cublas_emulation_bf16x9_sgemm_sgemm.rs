use singe_cublas::{
    blas::level3::sgemm,
    context::Context,
    error::Result,
    types::{EmulationStrategy, MathMode, Operation},
};
use singe_cuda::{context::Context as CudaContext, device::Device, memory::DeviceMemory};

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

    sgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        4,
        4,
        4,
        &1.0_f32,
        &a,
        4,
        &b,
        4,
        &0.0_f32,
        &mut c_native,
        4,
    )?;

    cublas.set_math_mode(MathMode::FP32_EMULATED_BF16X9)?;
    cublas.set_emulation_strategy(EmulationStrategy::Eager)?;
    sgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        4,
        4,
        4,
        &1.0_f32,
        &a,
        4,
        &b,
        4,
        &0.0_f32,
        &mut c_emulated,
        4,
    )?;

    println!("sgemm native: {:?}", c_native.copy_to_host_vec()?);
    println!("sgemm bf16x9: {:?}", c_emulated.copy_to_host_vec()?);
    Ok(())
}
