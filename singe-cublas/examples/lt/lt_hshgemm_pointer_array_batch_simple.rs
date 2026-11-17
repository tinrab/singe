use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, matmul_async},
        types::BatchMode,
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

    let a_batches = [
        DeviceMemory::from_slice(&cuda, &[one, three, two, four])?,
        DeviceMemory::from_slice(&cuda, &[two, four, one, three])?,
    ];
    let b_batches = [
        DeviceMemory::from_slice(&cuda, &[one, one, one, one])?,
        DeviceMemory::from_slice(&cuda, &[one, two, three, four])?,
    ];
    let c_batches = [
        DeviceMemory::<f16>::zeroed(&cuda, usize::try_from(M * N)?)?,
        DeviceMemory::<f16>::zeroed(&cuda, usize::try_from(M * N)?)?,
    ];
    let mut d_batches = [
        DeviceMemory::<f16>::zeroed(&cuda, usize::try_from(M * N)?)?,
        DeviceMemory::<f16>::zeroed(&cuda, usize::try_from(M * N)?)?,
    ];

    let a_ptrs = DeviceMemory::from_slice(
        &cuda,
        &a_batches
            .iter()
            .map(DeviceMemory::as_device_ptr)
            .collect::<Vec<_>>(),
    )?;
    let b_ptrs = DeviceMemory::from_slice(
        &cuda,
        &b_batches
            .iter()
            .map(DeviceMemory::as_device_ptr)
            .collect::<Vec<_>>(),
    )?;
    let c_ptrs = DeviceMemory::from_slice(
        &cuda,
        &c_batches
            .iter()
            .map(DeviceMemory::as_device_ptr)
            .collect::<Vec<_>>(),
    )?;
    let mut d_ptrs = DeviceMemory::from_slice(
        &cuda,
        &d_batches
            .iter_mut()
            .map(|batch| batch.as_device_ptr())
            .collect::<Vec<_>>(),
    )?;
    let mut workspace = DeviceMemory::<u8>::create_uninit(&cuda, WORKSPACE_BYTES)?;

    let mut a_layout = MatrixLayout::create(DataType::F16, M, K, i64::try_from(M)?)?;
    a_layout.set_batch_count(BATCH_COUNT)?;
    a_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut b_layout = MatrixLayout::create(DataType::F16, K, N, i64::try_from(K)?)?;
    b_layout.set_batch_count(BATCH_COUNT)?;
    b_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut c_layout = MatrixLayout::create(DataType::F16, M, N, i64::try_from(M)?)?;
    c_layout.set_batch_count(BATCH_COUNT)?;
    c_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut d_layout = MatrixLayout::create(DataType::F16, M, N, i64::try_from(M)?)?;
    d_layout.set_batch_count(BATCH_COUNT)?;
    d_layout.set_batch_mode(BatchMode::PointerArray)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    // SAFETY: layouts match the retained buffers and scalars; the default stream is drained below.
    unsafe {
        matmul_async(
            &lt,
            &desc,
            Scalar::host(&alpha),
            &a_ptrs,
            &a_layout,
            &b_ptrs,
            &b_layout,
            Scalar::host(&beta),
            Some(DeviceInputView::from_buffer(&c_ptrs)),
            &c_layout,
            &mut d_ptrs,
            &d_layout,
            None,
            Some(workspace.output()),
            None,
        )
    }?;
    cuda.synchronize()?;

    for (index, batch) in d_batches.iter().enumerate() {
        println!(
            "lt_hshgemm_pointer_array_batch_simple batch {index}: {:?}",
            batch.copy_to_host_vec()?
        );
    }
    Ok(())
}
