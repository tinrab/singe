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
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::f16, view::DeviceInputView,
};

const M: u64 = 2;
const N: u64 = 2;
const K: u64 = 2;
const BATCH_COUNT: i32 = 2;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let one = f16::from_f32(1.0);
    let two = f16::from_f32(2.0);
    let three = f16::from_f32(3.0);
    let four = f16::from_f32(4.0);

    let a = DeviceMemory::from_slice(&cuda, &[one, three, two, four, two, four, one, three])?;
    let b = DeviceMemory::from_slice(&cuda, &[one, one, one, one, one, two, three, four])?;
    let c = DeviceMemory::<f16>::zeroed(&cuda, usize::try_from(M * N * BATCH_COUNT as u64)?)?;
    let mut d = DeviceMemory::<f16>::zeroed(&cuda, usize::try_from(M * N * BATCH_COUNT as u64)?)?;

    let mut a_layout = MatrixLayout::create(DataType::F16, M, K, i64::try_from(M)?)?;
    a_layout.set_batch_count(BATCH_COUNT)?;
    a_layout.set_strided_batch_offset(i64::try_from(M * K)?)?;

    let mut b_layout = MatrixLayout::create(DataType::F16, K, N, i64::try_from(K)?)?;
    b_layout.set_batch_count(BATCH_COUNT)?;
    b_layout.set_strided_batch_offset(i64::try_from(K * N)?)?;

    let mut c_layout = MatrixLayout::create(DataType::F16, M, N, i64::try_from(M)?)?;
    c_layout.set_batch_count(BATCH_COUNT)?;
    c_layout.set_strided_batch_offset(i64::try_from(M * N)?)?;

    let mut d_layout = MatrixLayout::create(DataType::F16, M, N, i64::try_from(M)?)?;
    d_layout.set_batch_count(BATCH_COUNT)?;
    d_layout.set_strided_batch_offset(i64::try_from(M * N)?)?;

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
        1,
    )?;

    let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, heuristics[0].workspace_size)?;
    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    // SAFETY: layouts match the retained buffers and scalars; the default stream is drained below.
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
            Some(&heuristics[0].algorithm),
            Some(workspace.output()),
            None,
        )
    }?;
    cuda.synchronize()?;

    let result = d.copy_to_host_vec()?;
    println!("lt_hshgemm_strided_batch_simple result: {result:?}");
    Ok(())
}
