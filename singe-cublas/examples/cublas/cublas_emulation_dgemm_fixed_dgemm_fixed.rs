use singe_cublas::{
    blas::level3::dgemm,
    context::Context,
    error::Result,
    types::{EmulationStrategy, MathMode, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, device::Device, memory::DeviceMemory,
    types::EmulationMantissaControl,
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, 1 << 20)?;
    let cublas = Context::create(&cuda)?;
    // The workspace precedes the handle, so it survives handle destruction on every path.
    // The completed GEMM below releases its borrowed operands only after CUDA completion.
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        cublas.set_workspace(Some(workspace.output()))?;
    }

    let a = DeviceMemory::from_slice(&cuda, &[1.0_f64, 2.0, 3.0, 4.0])?;
    let b = DeviceMemory::from_slice(&cuda, &[5.0_f64, 6.0, 7.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroed(&cuda, 4)?;

    cublas.set_math_mode(MathMode::FP64_EMULATED_FIXED_POINT)?;
    cublas.set_emulation_strategy(EmulationStrategy::Eager)?;
    cublas.set_emulation_mantissa_control(EmulationMantissaControl::Fixed)?;
    cublas.set_fixed_point_emulation_max_mantissa_bit_count(55)?;

    dgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &1.0_f64,
        &a,
        2,
        &b,
        2,
        &0.0_f64,
        &mut c,
        2,
    )?;

    println!("dgemm fixed emulation: {:?}", c.copy_to_host_vec()?);
    Ok(())
}
