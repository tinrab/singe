use singe_cublas::{
    lt::{
        context::Context as LtContext,
        descriptor::{MatrixLayout, MatrixTransformDescriptor},
        matmul::{MatmulDescriptor, matmul_async, matrix_transform_async},
        types::Order,
    },
    scalar::Scalar,
    types::{ComputeType, Operation},
};
use singe_cuda::{
    context::Context as CudaContext, data_type::DataType, device::Device, memory::DeviceMemory,
    view::DeviceInputView,
};

const M: u64 = 32;
const N: u64 = 32;
const K: u64 = 32;

fn round_up(value: u64, multiple: u64) -> u64 {
    value.div_ceil(multiple) * multiple
}

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let lt = LtContext::create(&cuda)?;

    let a = DeviceMemory::from_slice(&cuda, &vec![1_i8; usize::try_from(M * K)?])?;
    let b = DeviceMemory::from_slice(&cuda, &vec![1_i8; usize::try_from(K * N)?])?;
    let mut c = DeviceMemory::<i32>::zeroed(&cuda, usize::try_from(M * N)?)?;

    let a_transform_ld = 32 * M;
    let b_transform_ld = 32 * round_up(N, 8);
    let c_transform_ld = 32 * M;

    let mut a_transform = DeviceMemory::<i8>::zeroed(
        &cuda,
        usize::try_from(round_up(K, 32) / 32 * a_transform_ld)?,
    )?;
    let mut b_transform = DeviceMemory::<i8>::zeroed(
        &cuda,
        usize::try_from(round_up(K, 32) / 32 * b_transform_ld)?,
    )?;
    let mut c_transform = DeviceMemory::<i32>::zeroed(
        &cuda,
        usize::try_from(round_up(N, 32) / 32 * c_transform_ld)?,
    )?;

    let a_layout = MatrixLayout::create(DataType::I8, M, K, i64::try_from(M)?)?;
    let b_layout = MatrixLayout::create(DataType::I8, K, N, i64::try_from(K)?)?;
    let c_layout = MatrixLayout::create(DataType::I32, M, N, i64::try_from(M)?)?;

    let mut a_transform_layout =
        MatrixLayout::create(DataType::I8, M, K, i64::try_from(a_transform_ld)?)?;
    a_transform_layout.set_order(Order::Column32)?;

    let mut b_transform_layout =
        MatrixLayout::create(DataType::I8, N, K, i64::try_from(b_transform_ld)?)?;
    b_transform_layout.set_order(Order::Column4_4R2_8C)?;

    let mut c_transform_layout =
        MatrixLayout::create(DataType::I32, M, N, i64::try_from(c_transform_ld)?)?;
    c_transform_layout.set_order(Order::Column32)?;

    let mut transform_desc = MatrixTransformDescriptor::create(DataType::F32)?;
    let transform_alpha = 1.0_f32;
    let transform_beta = 0.0_f32;

    // SAFETY: the layouts describe the sizes of the buffers, all work runs on the default stream in order, and `cuda.synchronize()` drains it before the buffers drop.
    unsafe {
        matrix_transform_async(
            &lt,
            &transform_desc,
            Scalar::host(&transform_alpha),
            &a,
            &a_layout,
            Scalar::host(&transform_beta),
            None,
            &mut a_transform,
            &a_transform_layout,
            None,
        )
    }?;

    transform_desc.set_transpose_a(Operation::Transpose)?;
    // SAFETY: the layouts describe the sizes of the buffers, all work runs on the default stream in order, and `cuda.synchronize()` drains it before the buffers drop.
    unsafe {
        matrix_transform_async(
            &lt,
            &transform_desc,
            Scalar::host(&transform_alpha),
            &b,
            &b_layout,
            Scalar::host(&transform_beta),
            None,
            &mut b_transform,
            &b_transform_layout,
            None,
        )
    }?;

    let mut desc = MatmulDescriptor::create(ComputeType::I32, DataType::I32)?;
    desc.set_transpose_a(Operation::NonTranspose)?;
    desc.set_transpose_b(Operation::NonTranspose)?;
    let alpha = 1_i32;
    let beta = 0_i32;
    let c_accumulator = DeviceMemory::<i32>::zeroed(
        &cuda,
        usize::try_from(round_up(N, 32) / 32 * c_transform_ld)?,
    )?;
    // SAFETY: layouts match the retained buffers and scalars; the default stream is drained below.
    unsafe {
        matmul_async(
            &lt,
            &desc,
            Scalar::host(&alpha),
            &a_transform,
            &a_transform_layout,
            &b_transform,
            &b_transform_layout,
            Scalar::host(&beta),
            Some(DeviceInputView::from_buffer(&c_accumulator)),
            &c_transform_layout,
            &mut c_transform,
            &c_transform_layout,
            None,
            None,
            None,
        )
    }?;
    cuda.synchronize()?;

    transform_desc.set_transpose_a(Operation::NonTranspose)?;
    // SAFETY: the layouts describe the sizes of the buffers, all work runs on the default stream in order, and `cuda.synchronize()` drains it before the buffers drop.
    unsafe {
        matrix_transform_async(
            &lt,
            &transform_desc,
            Scalar::host(&transform_alpha),
            &c_transform,
            &c_transform_layout,
            Scalar::host(&transform_beta),
            None,
            &mut c,
            &c_layout,
            None,
        )
    }?;
    cuda.synchronize()?;

    let result = c.copy_to_host_vec()?;
    let expected = i32::try_from(K)?;
    assert!(result.iter().all(|&value| value == expected));
    println!("lt_igemm_tensor result[0]: {}", result[0]);
    Ok(())
}
