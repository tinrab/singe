use std::error::Error;

use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{
            MatmulDescriptor, check_matmul_algorithm, matmul, matmul_algorithm,
            matmul_algorithm_ids,
        },
        types::{MatmulAlgorithmCapAttribute, MatmulAlgorithmConfigAttribute},
    },
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    event::EventRecordFlags,
    memory::DeviceMemory,
    types::{bf16, f8e4m3},
};

const M: u64 = 64;
const N: u64 = 64;
const K: u64 = 64;
const WORKSPACE_BYTES: usize = 1 << 20;
const REPEATS: usize = 2;

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let stream = cuda.create_stream()?;
    let start = cuda.create_event()?;
    let stop = cuda.create_event()?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&vec![f8e4m3::from_bits(1); (M * K) as usize])?;
    let b = DeviceMemory::from_slice(&vec![f8e4m3::from_bits(1); (K * N) as usize])?;
    let c = DeviceMemory::<bf16>::zeroes((M * N) as usize)?;
    let mut d = DeviceMemory::<f8e4m3>::zeroes((M * N) as usize)?;
    let a_scale = DeviceMemory::from_slice(&[1.0_f32])?;
    let b_scale = DeviceMemory::from_slice(&[1.0_f32])?;
    let c_scale = DeviceMemory::from_slice(&[1.0_f32])?;
    let d_scale = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut amax_d = DeviceMemory::<f32>::zeroes(1)?;

    let a_layout = MatrixLayout::create(DataType::F8E4M3, M, K, M as i64)?;
    let b_layout = MatrixLayout::create(DataType::F8E4M3, K, N, K as i64)?;
    let c_layout = MatrixLayout::create(DataType::Bf16, M, N, M as i64)?;
    let d_layout = MatrixLayout::create(DataType::F8E4M3, M, N, M as i64)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::Transpose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_a_scale_pointer(&a_scale)?;
    desc.set_b_scale_pointer(&b_scale)?;
    desc.set_c_scale_pointer(&c_scale)?;
    desc.set_d_scale_pointer(&d_scale)?;
    desc.set_amax_d_pointer(&mut amax_d)?;

    let alpha = 2.0_f32;
    let beta = 0.0_f32;
    let ids = matmul_algorithm_ids(
        &lt,
        ComputeType::F32,
        DataType::F32,
        DataType::F8E4M3,
        DataType::F8E4M3,
        DataType::Bf16,
        DataType::F8E4M3,
        8,
    )?;

    let mut best = None;
    for id in ids {
        let mut algorithm = matmul_algorithm(
            &lt,
            ComputeType::F32,
            DataType::F32,
            DataType::F8E4M3,
            DataType::F8E4M3,
            DataType::Bf16,
            DataType::F8E4M3,
            id,
        )?;
        let splitk_support =
            algorithm.cap_attribute::<i32>(MatmulAlgorithmCapAttribute::SplitKSupport)? != 0;
        for splitk in [1_i32, 2_i32] {
            if splitk > 1 && !splitk_support {
                continue;
            }
            algorithm.set_config_attribute(MatmulAlgorithmConfigAttribute::SplitKCount, &splitk)?;
            let heuristic = match check_matmul_algorithm(
                &lt, &desc, &a_layout, &b_layout, &c_layout, &d_layout, &algorithm,
            ) {
                Ok(heuristic) if heuristic.workspace_size <= WORKSPACE_BYTES => heuristic,
                _ => continue,
            };

            let mut workspace = DeviceMemory::<u8>::create(heuristic.workspace_size)?;
            start.record(&stream, EventRecordFlags::DEFAULT)?;
            for _ in 0..REPEATS {
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
                    Some(&stream),
                )?;
            }
            stop.record(&stream, EventRecordFlags::DEFAULT)?;
            stop.synchronize()?;
            let elapsed = stop.elapsed_time_since(&start)? / REPEATS as f32;
            if best.is_none_or(|(_, _, best_time)| elapsed < best_time) {
                best = Some((id, splitk, elapsed));
            }
        }
    }

    println!("lt_fp8_custom_find best candidate: {best:?}");
    Ok(())
}
