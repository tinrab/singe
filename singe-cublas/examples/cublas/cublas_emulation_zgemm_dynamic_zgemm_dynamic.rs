use singe_cublas::{
    blas::level3::zgemm,
    context::Context,
    error::Result,
    types::{EmulationStrategy, MathMode, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    device::Device,
    memory::DeviceMemory,
    types::{Complex64, EmulationMantissaControl},
};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut workspace = DeviceMemory::<u8>::create(1 << 20)?;
    cublas.set_workspace(Some(&mut workspace))?;

    let a = DeviceMemory::from_slice(&[
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -2.0),
        Complex64::new(3.0, -3.0),
        Complex64::new(4.0, 4.0),
    ])?;
    let b = DeviceMemory::from_slice(&[
        Complex64::new(5.0, 5.0),
        Complex64::new(6.0, -6.0),
        Complex64::new(7.0, -7.0),
        Complex64::new(8.0, 8.0),
    ])?;
    let mut c = DeviceMemory::<Complex64>::zeroes(4)?;

    cublas.set_math_mode(MathMode::FP64_EMULATED_FIXED_POINT)?;
    cublas.set_emulation_strategy(EmulationStrategy::Eager)?;
    cublas.set_emulation_mantissa_control(EmulationMantissaControl::Dynamic)?;
    cublas.set_fixed_point_emulation_max_mantissa_bit_count(79)?;
    cublas.set_fixed_point_emulation_mantissa_bit_offset(-8)?;

    zgemm(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &Complex64::new(1.0, -1.0),
        &a,
        2,
        &b,
        2,
        &Complex64::new(0.0, 0.0),
        &mut c,
        2,
    )?;

    println!("zgemm dynamic emulation: {:?}", c.copy_to_host_vec()?);
    Ok(())
}
