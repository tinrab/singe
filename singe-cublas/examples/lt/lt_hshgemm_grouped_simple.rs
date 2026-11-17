use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul_algorithm_heuristics, matmul_async},
        types::PointerMode,
    },
    scalar::Scalar,
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    types::f16, view::DeviceInputView,
};

const GROUP_COUNT: usize = 2;
const M: i64 = 2;
const N: i64 = 2;
const K: i64 = 2;
const WORKSPACE_BYTES: usize = 1 << 20;

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let rows = DeviceMemory::from_slice(&cuda, &[M, M])?;
    let cols = DeviceMemory::from_slice(&cuda, &[N, N])?;
    let reduction = DeviceMemory::from_slice(&cuda, &[K, K])?;
    let lda = DeviceMemory::from_slice(&cuda, &[M, M])?;
    let ldb = DeviceMemory::from_slice(&cuda, &[K, K])?;
    let ldc = DeviceMemory::from_slice(&cuda, &[M, M])?;
    let ldd = DeviceMemory::from_slice(&cuda, &[M, M])?;

    let a_batches = [
        DeviceMemory::from_slice(&cuda, &vec![f16::from_f32(1.0); usize::try_from(M * K)?])?,
        DeviceMemory::from_slice(&cuda, &vec![f16::from_f32(2.0); usize::try_from(M * K)?])?,
    ];
    let b_batches = [
        DeviceMemory::from_slice(&cuda, &vec![f16::from_f32(1.0); usize::try_from(K * N)?])?,
        DeviceMemory::from_slice(&cuda, &vec![f16::from_f32(3.0); usize::try_from(K * N)?])?,
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

    let alpha = DeviceMemory::from_slice(&cuda, &[1.0_f32, 1.0])?;
    let beta = DeviceMemory::from_slice(&cuda, &[0.0_f32, 0.0])?;

    // SAFETY: the dimension arrays describe the matrices the pointer arrays address and outlive the matmul, which `cuda.synchronize()` drains before they drop.
    let (a_layout, b_layout, c_layout, d_layout) = unsafe {
        (
            MatrixLayout::create_grouped(DataType::F16, &rows, &reduction, &lda)?,
            MatrixLayout::create_grouped(DataType::F16, &reduction, &cols, &ldb)?,
            MatrixLayout::create_grouped(DataType::F16, &rows, &cols, &ldc)?,
            MatrixLayout::create_grouped(DataType::F16, &rows, &cols, &ldd)?,
        )
    };

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_pointer_mode(PointerMode::Device)?;
    desc.set_alpha_batch_stride(1)?;
    desc.set_beta_batch_stride(1)?;

    let mut preference = MatmulPreference::create()?;
    preference.set_grouped_desc_d_average_rows(u32::try_from(M)?)?;
    preference.set_grouped_desc_d_average_cols(u32::try_from(N)?)?;
    preference.set_grouped_average_reduction_dim(u32::try_from(K)?)?;
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

    // Each scalar tracks only its first element, while cuBLASLt reads one per group through the batch stride.
    // SAFETY: layouts match the retained buffers and scalars; the default stream is drained below.
    unsafe {
        matmul_async(
            &lt,
            &desc,
            Scalar::device(&alpha)?,
            &a_ptrs,
            &a_layout,
            &b_ptrs,
            &b_layout,
            Scalar::device(&beta)?,
            Some(DeviceInputView::from_buffer(&c_ptrs)),
            &c_layout,
            &mut d_ptrs,
            &d_layout,
            Some(&heuristics[0].algorithm),
            Some(workspace.output()),
            None,
        )
    }?;
    cuda.synchronize()?;

    println!("lt_hshgemm_grouped_simple groups: {GROUP_COUNT}");
    Ok(())
}
