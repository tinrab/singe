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
    let cublas = Context::create(&cuda)?;
    let mut workspace = DeviceMemory::<u8>::create(1 << 20)?;
    cublas.set_workspace(Some(&mut workspace))?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?;
    let b = DeviceMemory::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?;
    let mut c = DeviceMemory::<f64>::zeroes(4)?;

    cublas.set_math_mode(MathMode::FP64_EMULATED_FIXED_POINT)?;
    cublas.set_emulation_strategy(EmulationStrategy::Eager)?;
    cublas.set_emulation_mantissa_control(EmulationMantissaControl::Dynamic)?;
    cublas.set_fixed_point_emulation_max_mantissa_bit_count(79)?;
    cublas.set_fixed_point_emulation_mantissa_bit_offset(-8)?;

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

    println!("dgemm dynamic emulation: {:?}", c.copy_to_host_vec()?);
    Ok(())
}
