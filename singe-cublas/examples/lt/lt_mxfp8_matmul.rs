use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::MatrixLayout,
        matmul::{MatmulDescriptor, MatmulPreference, matmul_algorithm_heuristics, matmul_async},
        types::MatrixScale,
    },
    scalar::Scalar,
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext,
    data_type::DataType,
    device::Device,
    memory::DeviceMemory,
    types::{bf16, f8e4m3, f8ue8m0},
    view::DeviceInputView,
};

const M: u64 = 32;
const N: u64 = 32;
const K: u64 = 32;
const WORKSPACE_BYTES: usize = 1 << 20;

// cuBLASLt stores block scales in 128x4 tiles, so `outer` rounds up to 128 and the `inner.div_ceil(block)` scales per row round up to 4.
fn block_scale_len(inner: u64, outer: u64, block: u64) -> anyhow::Result<usize> {
    Ok(usize::try_from(
        outer.next_multiple_of(128) * inner.div_ceil(block).next_multiple_of(4),
    )?)
}

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &vec![f8e4m3::from_bits(1); usize::try_from(M * K)?])?;
    let b = DeviceMemory::from_slice(&cuda, &vec![f8e4m3::from_bits(1); usize::try_from(K * N)?])?;
    let c = DeviceMemory::<bf16>::zeroed(&cuda, usize::try_from(M * N)?)?;
    let mut d = DeviceMemory::<f8e4m3>::zeroed(&cuda, usize::try_from(M * N)?)?;

    let a_scale = DeviceMemory::from_slice(
        &cuda,
        &vec![f8ue8m0::from_bits(1); block_scale_len(K, M, 32)?],
    )?;
    let b_scale = DeviceMemory::from_slice(
        &cuda,
        &vec![f8ue8m0::from_bits(1); block_scale_len(K, N, 32)?],
    )?;
    let mut d_out_scale = DeviceMemory::<f8ue8m0>::zeroed(&cuda, block_scale_len(M, N, 32)?)?;

    let a_layout = MatrixLayout::create(DataType::F8E4M3, M, K, i64::try_from(M)?)?;
    let b_layout = MatrixLayout::create(DataType::F8E4M3, K, N, i64::try_from(K)?)?;
    let c_layout = MatrixLayout::create(DataType::Bf16, M, N, i64::try_from(M)?)?;
    let d_layout = MatrixLayout::create(DataType::F8E4M3, M, N, i64::try_from(M)?)?;

    let mut desc = MatmulDescriptor::create(ComputeType::F32, DataType::F32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    desc.set_a_scale_mode(MatrixScale::Vector32UE8M0)?;
    desc.set_b_scale_mode(MatrixScale::Vector32UE8M0)?;
    desc.set_d_out_scale_mode(MatrixScale::Vector32UE8M0)?;
    // SAFETY: each scale buffer holds the full tiled tensor of 32-element block scales for its matrix, and all outlive the matmul, which `cuda.synchronize()` drains before they drop.
    unsafe {
        desc.set_a_scale_pointer(&a_scale)?;
        desc.set_b_scale_pointer(&b_scale)?;
        desc.set_d_out_scale_pointer(&mut d_out_scale)?;
    }

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
    // The `bf16` layout tells cuBLASLt how to read `C`; the view only shares `D`'s element type.
    let c_view = DeviceInputView::from_buffer(&c).cast::<f8e4m3>()?;
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
            Some(c_view),
            &c_layout,
            &mut d,
            &d_layout,
            Some(&heuristics[0].algorithm),
            Some(workspace.output()),
            None,
        )
    }?;
    cuda.synchronize()?;

    println!(
        "lt_mxfp8_matmul output elements: {}",
        d.copy_to_host_vec()?.len()
    );
    Ok(())
}
