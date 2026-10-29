use singe_cublas::{
    blas::level3::gemm_ex,
    context::Context,
    error::Result,
    types::{ComputeType, EmulationStrategy, GemmAlgorithm, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
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

    cublas.set_emulation_strategy(EmulationStrategy::Eager)?;
    cublas.set_emulation_mantissa_control(EmulationMantissaControl::Dynamic)?;
    cublas.set_fixed_point_emulation_max_mantissa_bit_count(79)?;
    cublas.set_fixed_point_emulation_mantissa_bit_offset(-8)?;

    gemm_ex(
        &cublas,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &1.0_f64,
        &a,
        DataType::F64,
        2,
        &b,
        DataType::F64,
        2,
        &0.0_f64,
        &mut c,
        DataType::F64,
        2,
        ComputeType::F64EmulatedFixedPoint,
        GemmAlgorithm::Default,
    )?;

    println!("gemm_ex dynamic emulation: {:?}", c.copy_to_host_vec()?);
    Ok(())
}
