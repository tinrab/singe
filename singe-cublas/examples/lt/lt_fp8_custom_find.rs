use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{
            MatmulDescriptor, check_matmul_algorithm, matmul_algorithm, matmul_algorithm_ids,
            matmul_async,
        },
    },
    scalar::Scalar,
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    event::{EventFlags, EventRecordFlags},
    memory::DeviceMemory,
    stream::StreamOptions,
    types::{bf16, f8e4m3},
    view::DeviceInputView,
};

const M: u64 = 64;
const N: u64 = 64;
const K: u64 = 64;
const WORKSPACE_BYTES: usize = 1 << 20;
const REPEATS: usize = 2;

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let stream = cuda.create_stream(StreamOptions::default())?;
    let start = cuda.create_event(EventFlags::DEFAULT)?;
    let stop = cuda.create_event(EventFlags::DEFAULT)?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &vec![f8e4m3::from_bits(1); usize::try_from(M * K)?])?;
    let b = DeviceMemory::from_slice(&cuda, &vec![f8e4m3::from_bits(1); usize::try_from(K * N)?])?;
    let c = DeviceMemory::<bf16>::zeroed(&cuda, usize::try_from(M * N)?)?;
    let mut d = DeviceMemory::<f8e4m3>::zeroed(&cuda, usize::try_from(M * N)?)?;
    let a_scale = DeviceMemory::from_slice(&cuda, &[1.0_f32])?;
    let b_scale = DeviceMemory::from_slice(&cuda, &[1.0_f32])?;
    let c_scale = DeviceMemory::from_slice(&cuda, &[1.0_f32])?;
    let d_scale = DeviceMemory::from_slice(&cuda, &[1.0_f32])?;
    let mut amax_d = DeviceMemory::<f32>::zeroed(&cuda, 1)?;

    let a_layout = MatrixLayout::create(DataType::F8E4M3, M, K, i64::try_from(M)?)?;
    let b_layout = MatrixLayout::create(DataType::F8E4M3, K, N, i64::try_from(K)?)?;
    let c_layout = MatrixLayout::create(DataType::Bf16, M, N, i64::try_from(M)?)?;
    let d_layout = MatrixLayout::create(DataType::F8E4M3, M, N, i64::try_from(M)?)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::Transpose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    // SAFETY: each scale is one `f32` tensor-wide scalar, `amax_d` holds the one `f32` amax that the 32-bit float compute type writes, and all outlive every timed matmul, each of which is waited for before they drop.
    unsafe {
        desc.set_a_scale_pointer(&a_scale)?;
        desc.set_b_scale_pointer(&b_scale)?;
        desc.set_c_scale_pointer(&c_scale)?;
        desc.set_d_scale_pointer(&d_scale)?;
        desc.set_amax_d_pointer(&mut amax_d)?;
    }

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

    // The `bf16` layout tells cuBLASLt how to read `C`; the view only shares `D`'s element type.
    let c_view = DeviceInputView::from_buffer(&c).cast::<f8e4m3>()?;
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
        let splitk_support = algorithm.supports_split_k()?;
        for splitk in [1_i32, 2_i32] {
            if splitk > 1 && !splitk_support {
                continue;
            }

            algorithm.set_split_k_count(splitk as usize)?;
            let heuristic = match check_matmul_algorithm(
                &lt, &desc, &a_layout, &b_layout, &c_layout, &d_layout, &algorithm,
            ) {
                Ok(heuristic) if heuristic.workspace_size <= WORKSPACE_BYTES => heuristic,
                _ => continue,
            };

            let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, heuristic.workspace_size)?;
            start.record(&stream, EventRecordFlags::DEFAULT)?;
            for _ in 0..REPEATS {
                // SAFETY: descriptors match these buffers, which remain live until the completion wait below.
                unsafe {
                    matmul_async(
                        &lt,
                        &desc,
                        Scalar::host(&alpha),
                        &a,
                        &a_layout,
                        &b,
                        &b_layout,
                        Scalar::host(&beta),
                        Some(c_view),
                        &c_layout,
                        &mut d,
                        &d_layout,
                        Some(&algorithm),
                        Some(workspace.output()),
                        Some(&stream),
                    )
                }?;
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
