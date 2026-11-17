use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{
            MatmulDescriptor, MatmulPreference, check_matmul_algorithm, matmul_algorithm,
            matmul_algorithm_ids, matmul_async,
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

    let a = DeviceMemory::from_slice(&cuda, &vec![1.0_f32; usize::try_from(M * K)?])?;
    let b = DeviceMemory::from_slice(&cuda, &vec![1.0_f32; usize::try_from(K * N)?])?;
    let c = DeviceMemory::<f32>::zeroed(&cuda, usize::try_from(M * N)?)?;
    let mut d = DeviceMemory::<f32>::zeroed(&cuda, usize::try_from(M * N)?)?;

    let a_layout = MatrixLayout::create(DataType::F32, M, K, i64::try_from(M)?)?;
    let b_layout = MatrixLayout::create(DataType::F32, K, N, i64::try_from(K)?)?;
    let c_layout = MatrixLayout::create(DataType::F32, M, N, i64::try_from(M)?)?;
    let d_layout = MatrixLayout::create(DataType::F32, M, N, i64::try_from(M)?)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

    let mut preference = MatmulPreference::create()?;
    preference.set_max_workspace_bytes(WORKSPACE_BYTES)?;
    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    let ids = matmul_algorithm_ids(
        &lt,
        ComputeType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        8,
    )?;

    let mut best = None;
    for id in ids {
        let mut algorithm = matmul_algorithm(
            &lt,
            ComputeType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
            DataType::F32,
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
                        Some(DeviceInputView::from_buffer(&c)),
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

    println!("lt_sgemm_custom_find best candidate: {best:?}");
    Ok(())
}
