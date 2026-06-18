use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, check_matmul_algorithm, matmul, matmul_algorithm},
        types::{MatmulAlgorithmConfigAttribute, ReductionScheme},
    },
    types::{ComputeType, Operation},
};
use singe_cublas_sys as sys;
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
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

    let mut desc = MatmulDescriptor::create(ComputeType::F64, DataType::F64)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

    let mut algorithm = matmul_algorithm(
        &lt,
        ComputeType::F64,
        DataType::F64,
        DataType::F64,
        DataType::F64,
        DataType::F64,
        DataType::F64,
        10,
    )?;

    let tile = sys::cublasLtMatmulTile_t::CUBLASLT_MATMUL_TILE_16x16;
    let split_k = 2_i32;
    algorithm.set_config_attribute(MatmulAlgorithmConfigAttribute::TileId, &tile)?;
    algorithm.set_config_attribute(MatmulAlgorithmConfigAttribute::SplitKCount, &split_k)?;
    algorithm.set_config_attribute(
        MatmulAlgorithmConfigAttribute::ReductionScheme,
        &sys::cublasLtReductionScheme_t::from(ReductionScheme::InPlace),
    )?;

    let heuristic = check_matmul_algorithm(
        &lt, &desc, &a_layout, &b_layout, &c_layout, &d_layout, &algorithm,
    )?;
    let mut workspace = DeviceMemory::<u8>::create(WORKSPACE_BYTES.max(heuristic.workspace_size))?;

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
        Some(&algorithm),
        Some(&mut workspace),
        None,
    )?;

    let result = d.copy_to_host_vec()?;
    println!("lt_dgemm_preset_algo result: {result:?}");
    Ok(())
}
