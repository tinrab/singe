use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::{EmulationDescriptor, MatrixLayout},
        matmul::{MatmulDescriptor, MatmulPreference, matmul, matmul_algorithm_heuristics},
        types::EmulationDescAttribute,
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    memory::DeviceMemory,
    types::{EmulationMantissaControl, EmulationSpecialValuesSupport, EmulationStrategy},
};

const M: u64 = 2;
const N: u64 = 2;
const K: u64 = 2;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&[1.0_f64, 3.0, 2.0, 4.0])?;
    let b = DeviceMemory::from_slice(&[5.0_f64, 7.0, 6.0, 8.0])?;
    let c = DeviceMemory::<f64>::zeroes((M * N) as usize)?;
    let mut d = DeviceMemory::<f64>::zeroes((M * N) as usize)?;

    let a_layout = MatrixLayout::create(DataType::F64, M, K, M as i64)?;
    let b_layout = MatrixLayout::create(DataType::F64, K, N, K as i64)?;
    let c_layout = MatrixLayout::create(DataType::F64, M, N, M as i64)?;
    let d_layout = MatrixLayout::create(DataType::F64, M, N, M as i64)?;

    let mut emulation = EmulationDescriptor::create()?;
    emulation.set_strategy(EmulationStrategy::Performant)?;
    emulation.set_special_values_support(
        EmulationSpecialValuesSupport::INFINITY | EmulationSpecialValuesSupport::NAN,
    )?;
    emulation.set_mantissa_control(EmulationMantissaControl::Dynamic)?;
    emulation.set_attribute(
        EmulationDescAttribute::FixedPointMaxMantissaBitCount,
        &52_i32,
    )?;
    emulation.set_mantissa_bit_offset(0)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F64EmulatedFixedPoint, DataType::F64)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_emulation_descriptor(&emulation)?;

    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(WORKSPACE_BYTES)?;
    let heuristics = matmul_algorithm_heuristics(
        &lt,
        &desc,
        &a_layout,
        &b_layout,
        &c_layout,
        &d_layout,
        &preference,
        1,
    )?;

    let mut workspace = DeviceMemory::<u8>::create(heuristics[0].workspace_size)?;
    let alpha = 1.0_f64;
    let beta = 0.0_f64;
    matmul(
        &lt,
        &desc,
        &alpha,
        &a,
        &a_layout,
        &b,
        &b_layout,
        &beta,
        &c,
        &c_layout,
        &mut d,
        &d_layout,
        Some(&heuristics[0].algorithm),
        Some(&mut workspace),
        None,
    )?;

    let result = d.copy_to_host_vec()?;
    println!("lt_dgemm_emulated result: {result:?}");
    Ok(())
}
