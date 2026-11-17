use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul_algorithm_heuristics, matmul_async},
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
const REPEATS: usize = 3;

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
    let heuristics = matmul_algorithm_heuristics(
        &lt,
        &desc,
        &a_layout,
        &b_layout,
        &c_layout,
        &d_layout,
        &preference,
        8,
    )?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    let mut best_time = f32::INFINITY;
    let mut best_index = None;

    for (index, heuristic) in heuristics.iter().enumerate() {
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
                    Some(&heuristic.algorithm),
                    Some(workspace.output()),
                    Some(&stream),
                )
            }?;
        }

        stop.record(&stream, EventRecordFlags::DEFAULT)?;
        stop.synchronize()?;
        let elapsed = stop.elapsed_time_since(&start)? / REPEATS as f32;
        if elapsed < best_time {
            best_time = elapsed;
            best_index = Some(index);
        }
    }

    println!(
        "lt_sgemm_simple_auto_tuning best heuristic: {best_index:?}, average time: {best_time:.4} ms"
    );
    Ok(())
}
