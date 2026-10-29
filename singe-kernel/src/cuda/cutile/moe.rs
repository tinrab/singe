use std::sync::Arc;

#[cfg(feature = "dtype_f8")]
use cutile::core::f8e4m3fn;
#[cfg(feature = "dtype_bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype_f16")]
use cutile::half::f16;
use cutile::{cuda_async::device_buffer::DevicePointer, cuda_core::Stream};
#[cfg(feature = "dtype_f8")]
use singe_cuda::device::Device as CudaDevice;

use crate::{
    cuda::{
        cutile::{
            DeviceOpExt, TileKernel,
            kernel::{common::fill_i32, moe as kernel_moe, moe_routing},
            utility::{checked_device_pointer, raw_vector_grid},
        },
        moe::MoeAlignBlockSizeConfig,
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedMoe {
    element_count: i32,
    columns: i32,
    reduction: i32,
    top_k: i32,
    block_size: i32,
    input_row_stride: i32,
    expert_stride: i32,
    weight_row_stride: i32,
    output_row_stride: i32,
    mul_routed_weight: i32,
}

#[cfg(feature = "dtype_f8")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedMoeBlockScaled {
    base: FusedMoe,
    group_n: i32,
    group_k: i32,
    k_groups: i32,
}

#[cfg(feature = "dtype_f8")]
impl FusedMoeBlockScaled {
    fn create(
        element_count: usize,
        columns: usize,
        reduction: usize,
        top_k: usize,
        block_size: usize,
        group_n: usize,
        group_k: usize,
        input_row_stride: usize,
        expert_stride: usize,
        weight_row_stride: usize,
        output_row_stride: usize,
        sorted_token_ids_len: usize,
        mul_routed_weight: bool,
    ) -> Result<Self> {
        if group_n == 0 || group_k == 0 {
            return Err(Error::InvalidLength);
        }
        let k_groups = ceil_div(reduction, group_k)?;
        Ok(Self {
            base: FusedMoe::create(
                element_count,
                columns,
                reduction,
                top_k,
                block_size,
                input_row_stride,
                expert_stride,
                weight_row_stride,
                output_row_stride,
                sorted_token_ids_len,
                mul_routed_weight,
            )?,
            group_n: checked_i32_value(group_n)?,
            group_k: checked_i32_value(group_k)?,
            k_groups: checked_i32_value(k_groups)?,
        })
    }
}

impl FusedMoe {
    fn create(
        element_count: usize,
        columns: usize,
        reduction: usize,
        top_k: usize,
        block_size: usize,
        input_row_stride: usize,
        expert_stride: usize,
        weight_row_stride: usize,
        output_row_stride: usize,
        sorted_token_ids_len: usize,
        mul_routed_weight: bool,
    ) -> Result<Self> {
        if element_count == 0
            || columns == 0
            || reduction == 0
            || top_k == 0
            || block_size == 0
            || input_row_stride == 0
            || expert_stride == 0
            || weight_row_stride == 0
            || output_row_stride < columns
            || sorted_token_ids_len == 0
        {
            return Err(Error::InvalidLength);
        }
        checked_i32_value(checked_element_count(sorted_token_ids_len, columns)?)?;
        Ok(Self {
            element_count: checked_i32_value(element_count)?,
            columns: checked_i32_value(columns)?,
            reduction: checked_i32_value(reduction)?,
            top_k: checked_i32_value(top_k)?,
            block_size: checked_i32_value(block_size)?,
            input_row_stride: checked_i32_value(input_row_stride)?,
            expert_stride: checked_i32_value(expert_stride)?,
            weight_row_stride: checked_i32_value(weight_row_stride)?,
            output_row_stride: checked_i32_value(output_row_stride)?,
            mul_routed_weight: i32::from(mul_routed_weight),
        })
    }
}

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn moe_align_block_size_i32(
    stream: &Arc<Stream>,
    sorted_token_ids: DevicePointer<i32>,
    expert_ids: DevicePointer<i32>,
    num_tokens_post_pad: DevicePointer<i32>,
    cumsum: DevicePointer<i32>,
    max_expert_count: DevicePointer<i32>,
    workspace: DevicePointer<i32>,
    workspace_len: usize,
    topk_ids: DevicePointer<i32>,
    element_count: usize,
    expert_count: usize,
    block_size: usize,
    sorted_token_ids_len: usize,
    expert_ids_len: usize,
) -> Result<()> {
    let config = MoeAlignBlockSizeConfig::create(element_count, expert_count, block_size)?;
    if sorted_token_ids_len < config.max_num_tokens_padded()?
        || expert_ids_len < config.max_num_blocks()?
        || workspace_len < config.workspace_len()?
    {
        return Err(Error::InvalidLength);
    }
    for pointer in [
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum,
        max_expert_count,
        workspace,
        topk_ids,
    ] {
        checked_device_pointer(&pointer)?;
    }
    let assignments = checked_i32_value(element_count)?;
    let experts = checked_i32_value(expert_count)?;
    let block = checked_i32_value(block_size)?;
    let chunks = checked_i32_value(element_count.div_ceil(256))?;
    let grid = (chunks as u32, experts as u32, 1);
    // SAFETY: checked ranges and the caller's allocation lifetimes cover the ordered stages.
    unsafe {
        fill_i32(
            sorted_token_ids,
            assignments,
            checked_i32_value(sorted_token_ids_len)?,
        )
        .grid(raw_vector_grid(sorted_token_ids_len)?)
        .enqueue_on(stream)?;
        moe_routing::count(workspace, topk_ids, assignments, chunks)
            .grid(grid)
            .enqueue_on(stream)?;
        moe_routing::prefix_chunks(workspace, chunks, experts)
            .grid((experts as u32, 1, 1))
            .enqueue_on(stream)?;
        moe_routing::prefix_experts(
            workspace,
            cumsum,
            num_tokens_post_pad,
            max_expert_count,
            chunks,
            experts,
            block,
        )
        .grid((1, 1, 1))
        .enqueue_on(stream)?;
        moe_routing::scatter(
            sorted_token_ids,
            expert_ids,
            topk_ids,
            workspace,
            cumsum,
            assignments,
            chunks,
            block,
        )
        .grid(grid)
        .enqueue_on(stream)?;
    }
    Ok(())
}

/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn fused_moe_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    weight: DevicePointer<f32>,
    routed_weight: DevicePointer<f32>,
    sorted_token_ids: DevicePointer<i32>,
    expert_ids: DevicePointer<i32>,
    num_tokens_post_pad: DevicePointer<i32>,
    element_count: usize,
    columns: usize,
    reduction: usize,
    top_k: usize,
    block_size: usize,
    input_row_stride: usize,
    expert_stride: usize,
    weight_row_stride: usize,
    output_row_stride: usize,
    sorted_token_ids_len: usize,
    mul_routed_weight: bool,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&input)?;
    checked_device_pointer(&weight)?;
    if mul_routed_weight {
        checked_device_pointer(&routed_weight)?;
    }
    checked_device_pointer(&sorted_token_ids)?;
    checked_device_pointer(&expert_ids)?;
    checked_device_pointer(&num_tokens_post_pad)?;

    let params = FusedMoe::create(
        element_count,
        columns,
        reduction,
        top_k,
        block_size,
        input_row_stride,
        expert_stride,
        weight_row_stride,
        output_row_stride,
        sorted_token_ids_len,
        mul_routed_weight,
    )?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel_moe::expert_product(
            out,
            input,
            weight,
            routed_weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            params.element_count,
            params.columns,
            params.reduction,
            params.top_k,
            params.block_size,
            params.input_row_stride,
            params.expert_stride,
            params.weight_row_stride,
            params.output_row_stride,
            params.mul_routed_weight,
            routed_weight,
            routed_weight,
            1i32,
            params.reduction,
            1i32,
        )
    }
    .generics(vec!["f32".into(), "f32".into(), "0".into()])
    .grid((
        u32::try_from(sorted_token_ids_len.div_ceil(block_size))
            .map_err(|_| Error::SizeOverflow)?,
        u32::try_from(columns.div_ceil(64)).map_err(|_| Error::SizeOverflow)?,
        u32::try_from(block_size.div_ceil(16)).map_err(|_| Error::SizeOverflow)?,
    ))
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(any(feature = "dtype_f16", feature = "dtype_bf16"))]
macro_rules! fused_moe_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        /// # Safety
        /// Pointers must reference correctly sized buffers on the stream's device.
        /// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
        pub unsafe fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            weight: DevicePointer<$ty>,
            routed_weight: DevicePointer<f32>,
            sorted_token_ids: DevicePointer<i32>,
            expert_ids: DevicePointer<i32>,
            num_tokens_post_pad: DevicePointer<i32>,
            element_count: usize,
            columns: usize,
            reduction: usize,
            top_k: usize,
            block_size: usize,
            input_row_stride: usize,
            expert_stride: usize,
            weight_row_stride: usize,
            output_row_stride: usize,
            sorted_token_ids_len: usize,
            mul_routed_weight: bool,
        ) -> Result<()> {
            checked_device_pointer(&out)?;
            checked_device_pointer(&input)?;
            checked_device_pointer(&weight)?;
            if mul_routed_weight {
                checked_device_pointer(&routed_weight)?;
            }
            checked_device_pointer(&sorted_token_ids)?;
            checked_device_pointer(&expert_ids)?;
            checked_device_pointer(&num_tokens_post_pad)?;

            let params = FusedMoe::create(
                element_count,
                columns,
                reduction,
                top_k,
                block_size,
                input_row_stride,
                expert_stride,
                weight_row_stride,
                output_row_stride,
                sorted_token_ids_len,
                mul_routed_weight,
            )?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                kernel_moe::expert_product(
                    out,
                    input,
                    weight,
                    routed_weight,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_pad,
                    params.element_count,
                    params.columns,
                    params.reduction,
                    params.top_k,
                    params.block_size,
                    params.input_row_stride,
                    params.expert_stride,
                    params.weight_row_stride,
                    params.output_row_stride,
                    params.mul_routed_weight,
                    routed_weight,
                    routed_weight,
                    1i32,
                    params.reduction,
                    1i32,
                )
            }
            .generics(vec![
                stringify!($ty).into(),
                stringify!($ty).into(),
                "0".into(),
            ])
            .grid((
                u32::try_from(sorted_token_ids_len.div_ceil(block_size))
                    .map_err(|_| Error::SizeOverflow)?,
                u32::try_from(columns.div_ceil(64)).map_err(|_| Error::SizeOverflow)?,
                u32::try_from(block_size.div_ceil(16)).map_err(|_| Error::SizeOverflow)?,
            ))
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype_f16")]
fused_moe_half_fn!(fused_moe_f16, f16, fused_moe_f16);

#[cfg(feature = "dtype_bf16")]
fused_moe_half_fn!(fused_moe_bf16, bf16, fused_moe_bf16);

#[cfg(feature = "dtype_f8")]
/// # Safety
/// Pointers must reference correctly sized buffers on the stream's device.
/// Keep allocations alive and order conflicting accesses until completion and throughout graph replay.
pub unsafe fn fused_moe_f8e4m3_block_scaled_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<u8>,
    weight: DevicePointer<u8>,
    input_scales: DevicePointer<f32>,
    weight_scales: DevicePointer<f32>,
    routed_weight: DevicePointer<f32>,
    sorted_token_ids: DevicePointer<i32>,
    expert_ids: DevicePointer<i32>,
    num_tokens_post_pad: DevicePointer<i32>,
    element_count: usize,
    columns: usize,
    reduction: usize,
    top_k: usize,
    block_size: usize,
    group_n: usize,
    group_k: usize,
    input_row_stride: usize,
    expert_stride: usize,
    weight_row_stride: usize,
    output_row_stride: usize,
    sorted_token_ids_len: usize,
    mul_routed_weight: bool,
) -> Result<()> {
    checked_device_pointer(&out)?;
    checked_device_pointer(&input)?;
    checked_device_pointer(&weight)?;
    checked_device_pointer(&input_scales)?;
    checked_device_pointer(&weight_scales)?;
    if mul_routed_weight {
        checked_device_pointer(&routed_weight)?;
    }
    checked_device_pointer(&sorted_token_ids)?;
    checked_device_pointer(&expert_ids)?;
    checked_device_pointer(&num_tokens_post_pad)?;

    let params = FusedMoeBlockScaled::create(
        element_count,
        columns,
        reduction,
        top_k,
        block_size,
        group_n,
        group_k,
        input_row_stride,
        expert_stride,
        weight_row_stride,
        output_row_stride,
        sorted_token_ids_len,
        mul_routed_weight,
    )?;
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let input = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(input.cu_deviceptr()) };
    // SAFETY: the address comes from a live device buffer; wrapping it does not dereference it.
    let weight = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(weight.cu_deviceptr()) };
    let properties = CudaDevice::new(stream.device().cu_device()).properties()?;
    let mode = if (properties.major, properties.minor) >= (8, 9) {
        1
    } else {
        -1
    };
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        kernel_moe::expert_product(
            out,
            input,
            weight,
            routed_weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            params.base.element_count,
            params.base.columns,
            params.base.reduction,
            params.base.top_k,
            params.base.block_size,
            params.base.input_row_stride,
            params.base.expert_stride,
            params.base.weight_row_stride,
            params.base.output_row_stride,
            params.base.mul_routed_weight,
            input_scales,
            weight_scales,
            params.group_n,
            params.group_k,
            params.k_groups,
        )
    }
    .generics(vec!["f8e4m3fn".into(), "f32".into(), mode.to_string()])
    .grid((
        u32::try_from(sorted_token_ids_len.div_ceil(block_size))
            .map_err(|_| Error::SizeOverflow)?,
        u32::try_from(columns.div_ceil(64)).map_err(|_| Error::SizeOverflow)?,
        u32::try_from(block_size.div_ceil(16)).map_err(|_| Error::SizeOverflow)?,
    ))
    .enqueue_on(stream)?;
    Ok(())
}

fn ceil_div(lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_add(rhs - 1)
        .ok_or(Error::SizeOverflow)?
        .checked_div(rhs)
        .ok_or(Error::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    use crate::{cpu::moe as cpu_moe, error::Result};

    #[test]
    fn test_moe_align_block_size_i32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let topk_ids_host = vec![2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3];
        let element_count = topk_ids_host.len();
        let expert_count = 5usize;
        let block_size = 4usize;
        let (expected_sorted, expected_experts, expected_total, expected_cumsum, expected_max) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, expert_count, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                element_count,
                expert_count,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;

        assert_eq!(
            sorted_token_ids.to_host_vec().sync_on(&stream)?,
            expected_sorted
        );
        assert_eq!(
            expert_ids.to_host_vec().sync_on(&stream)?,
            expected_experts_padded
        );
        assert_eq!(
            num_tokens_post_pad.to_host_vec().sync_on(&stream)?,
            vec![expected_total]
        );
        assert_eq!(cumsum.to_host_vec().sync_on(&stream)?, expected_cumsum);
        assert_eq!(
            max_expert_count.to_host_vec().sync_on(&stream)?,
            vec![expected_max]
        );
        Ok(())
    }

    #[test]
    fn test_fused_moe_f32_topk2() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (3usize, 2usize, 4usize, 5usize, 3usize, 2usize);
        let topk_ids_host = vec![1, 0, 2, 1, 3, 2];
        let input_host = (0..tokens * reduction)
            .map(|index| (index as f32 % 7.0) * 0.125 - 0.25)
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| (index as f32 % 11.0) * 0.0625 - 0.3125)
            .collect::<Vec<_>>();
        let routed_weight_host = (0..tokens * top_k)
            .map(|index| 0.5 + (index as f32 % 3.0) * 0.125)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f32(
            &input_host,
            &weight_host,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            true,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_f32(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                true,
            )
        }?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn test_fused_moe_f32_topk2_with_empty_experts() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (4usize, 2usize, 5usize, 4usize, 4usize, 3usize);
        let topk_ids_host = vec![0, 2, 4, 2, 0, 4, 2, 4];
        let input_host = (0..tokens * reduction)
            .map(|index| (index as f32 % 11.0) * 0.09375 - 0.375)
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| (index as f32 % 17.0) * 0.03125 - 0.25)
            .collect::<Vec<_>>();
        let routed_weight_host = (0..tokens * top_k)
            .map(|index| 0.375 + (index as f32 % 5.0) * 0.0625)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f32(
            &input_host,
            &weight_host,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            true,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_f32(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                true,
            )
        }?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype_f8")]
    #[test]
    fn test_fused_moe_f8e4m3_block_scaled_f32_topk2() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (3usize, 2usize, 4usize, 5usize, 5usize, 2usize);
        let (group_n, group_k) = (2usize, 3usize);
        let topk_ids_host = vec![1, 0, 2, 1, 3, 2];
        let input_host = vec![
            0x38u8, 0x40, 0x30, 0xb8, 0xc0, 0x40, 0x38, 0xb8, 0x30, 0x00, 0x30, 0xb8, 0x40, 0xc0,
            0x38,
        ];
        let weight_host = vec![
            0x38u8, 0x30, 0xb8, 0x40, 0xc0, 0x40, 0x38, 0x30, 0xb8, 0x00, 0xb8, 0x40, 0x38, 0x30,
            0xc0, 0x30, 0xb8, 0x40, 0x38, 0x00, 0xc0, 0x30, 0xb8, 0x40, 0x38, 0x30, 0x38, 0x40,
            0xb8, 0xc0, 0xb8, 0x30, 0x38, 0x40, 0x00, 0x40, 0xc0, 0x30, 0xb8, 0x38, 0x38, 0x40,
            0xc0, 0x30, 0xb8, 0x00, 0x30, 0x38, 0x40, 0xc0, 0xc0, 0xb8, 0x30, 0x38, 0x40, 0x40,
            0x38, 0x30, 0xb8, 0x00, 0x30, 0xb8, 0x40, 0x38, 0xc0, 0xb8, 0x40, 0x38, 0x30, 0x00,
            0x38, 0xc0, 0xb8, 0x30, 0x40, 0x30, 0x38, 0x40, 0xb8, 0xc0, 0x40, 0x30, 0xb8, 0x38,
            0x00, 0xb8, 0xc0, 0x38, 0x30, 0x40, 0x38, 0x30, 0x40, 0xc0, 0xb8, 0x00, 0x40, 0xb8,
            0x30, 0x38,
        ];
        let input_scales_host = vec![0.5f32, 2.0, 1.5, 0.75, 1.25, 0.25];
        let weight_scales_host = vec![
            1.0f32, 0.5, 2.0, 1.5, 0.75, 1.25, 0.25, 1.75, 0.5, 2.0, 1.0, 0.75, 1.25, 0.5, 1.5,
            0.75, 2.0, 0.25, 1.0, 1.25, 0.5, 1.5, 0.75, 2.0,
        ];
        let routed_weight_host = (0..tokens * top_k)
            .map(|index| 0.5 + (index as f32 % 3.0) * 0.125)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f8e4m3_block_scaled_f32(
            &input_host,
            &weight_host,
            &input_scales_host,
            &weight_scales_host,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            experts,
            columns,
            reduction,
            group_n,
            group_k,
            true,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let input_scales =
            api::copy_host_vec_to_device(&Arc::new(input_scales_host)).sync_on(&stream)?;
        let weight_scales =
            api::copy_host_vec_to_device(&Arc::new(weight_scales_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_f8e4m3_block_scaled_f32(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                input_scales.device_pointer(),
                weight_scales.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                group_n,
                group_k,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                true,
            )
        }?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype_f16")]
    #[test]
    fn test_fused_moe_f16_topk2() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (3usize, 2usize, 4usize, 5usize, 3usize, 2usize);
        let topk_ids_host = vec![1, 0, 2, 1, 3, 2];
        let input_host = (0..tokens * reduction)
            .map(|index| f16::from_f32((index as f32 % 7.0) * 0.125 - 0.25))
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| f16::from_f32((index as f32 % 11.0) * 0.0625 - 0.3125))
            .collect::<Vec<_>>();
        let routed_weight_host = (0..tokens * top_k)
            .map(|index| 0.5 + (index as f32 % 3.0) * 0.125)
            .collect::<Vec<_>>();
        let input_expected = input_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weight_expected = weight_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f32(
            &input_expected,
            &weight_expected,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            true,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_f16(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                true,
            )
        }?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype_bf16")]
    #[test]
    fn test_fused_moe_bf16_topk2() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (3usize, 2usize, 4usize, 5usize, 3usize, 2usize);
        let topk_ids_host = vec![1, 0, 2, 1, 3, 2];
        let input_host = (0..tokens * reduction)
            .map(|index| bf16::from_f32((index as f32 % 7.0) * 0.125 - 0.25))
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| bf16::from_f32((index as f32 % 11.0) * 0.0625 - 0.3125))
            .collect::<Vec<_>>();
        let routed_weight_host = (0..tokens * top_k)
            .map(|index| 0.5 + (index as f32 % 3.0) * 0.125)
            .collect::<Vec<_>>();
        let input_expected = input_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weight_expected = weight_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f32(
            &input_expected,
            &weight_expected,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            true,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_bf16(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                true,
            )
        }?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 4e-3);
        Ok(())
    }

    #[test]
    fn test_fused_moe_f32_topk1_without_routed_weight() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (5usize, 1usize, 3usize, 4usize, 5usize, 3usize);
        let topk_ids_host = vec![2, 0, 1, 2, 1];
        let input_host = (0..tokens * reduction)
            .map(|index| (index as f32 % 9.0) * 0.09375 - 0.375)
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| (index as f32 % 13.0) * 0.046875 - 0.25)
            .collect::<Vec<_>>();
        let routed_weight_host = vec![1.0f32; tokens * top_k];
        let expected = cpu_moe::fused_moe_f32(
            &input_host,
            &weight_host,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            false,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_f32(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                false,
            )
        }?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype_f16")]
    #[test]
    fn test_fused_moe_f16_topk1_without_routed_weight() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (5usize, 1usize, 3usize, 4usize, 5usize, 3usize);
        let topk_ids_host = vec![2, 0, 1, 2, 1];
        let input_host = (0..tokens * reduction)
            .map(|index| f16::from_f32((index as f32 % 9.0) * 0.09375 - 0.375))
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| f16::from_f32((index as f32 % 13.0) * 0.046875 - 0.25))
            .collect::<Vec<_>>();
        let routed_weight_host = vec![1.0f32; tokens * top_k];
        let input_expected = input_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weight_expected = weight_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f32(
            &input_expected,
            &weight_expected,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            false,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_f16(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                false,
            )
        }?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 1e-3);
        Ok(())
    }

    #[cfg(feature = "dtype_bf16")]
    #[test]
    fn test_fused_moe_bf16_topk1_without_routed_weight() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (tokens, top_k, experts, columns, reduction, block_size) =
            (5usize, 1usize, 3usize, 4usize, 5usize, 3usize);
        let topk_ids_host = vec![2, 0, 1, 2, 1];
        let input_host = (0..tokens * reduction)
            .map(|index| bf16::from_f32((index as f32 % 9.0) * 0.09375 - 0.375))
            .collect::<Vec<_>>();
        let weight_host = (0..experts * columns * reduction)
            .map(|index| bf16::from_f32((index as f32 % 13.0) * 0.046875 - 0.25))
            .collect::<Vec<_>>();
        let routed_weight_host = vec![1.0f32; tokens * top_k];
        let input_expected = input_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let weight_expected = weight_host
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        let expected = cpu_moe::fused_moe_f32(
            &input_expected,
            &weight_expected,
            &routed_weight_host,
            &topk_ids_host,
            tokens,
            top_k,
            columns,
            reduction,
            false,
        );
        let (expected_sorted, expected_experts, _, expected_cumsum, _) =
            cpu_moe::moe_align_block_size_i32(&topk_ids_host, experts, block_size);
        let max_expert_ids_len = cpu_moe::ceil_div(expected_sorted.len(), block_size);
        let mut expected_experts_padded = expected_experts;
        expected_experts_padded.resize(max_expert_ids_len, 0);

        let topk_ids = api::copy_host_vec_to_device(&Arc::new(topk_ids_host)).sync_on(&stream)?;
        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let weight = api::copy_host_vec_to_device(&Arc::new(weight_host)).sync_on(&stream)?;
        let routed_weight =
            api::copy_host_vec_to_device(&Arc::new(routed_weight_host)).sync_on(&stream)?;
        let sorted_token_ids = api::zeros::<i32>(&[expected_sorted.len()]).sync_on(&stream)?;
        let expert_ids = api::zeros::<i32>(&[expected_experts_padded.len()]).sync_on(&stream)?;
        let num_tokens_post_pad = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let cumsum = api::zeros::<i32>(&[expected_cumsum.len()]).sync_on(&stream)?;
        let max_expert_count = api::zeros::<i32>(&[1]).sync_on(&stream)?;
        let workspace_len = (expected_sorted.len().div_ceil(256) + 1) * (expected_cumsum.len() - 1);
        let workspace = api::zeros::<i32>(&[workspace_len]).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            super::moe_align_block_size_i32(
                &stream,
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                cumsum.device_pointer(),
                max_expert_count.device_pointer(),
                workspace.device_pointer(),
                workspace_len,
                topk_ids.device_pointer(),
                tokens * top_k,
                experts,
                block_size,
                expected_sorted.len(),
                expected_experts_padded.len(),
            )
        }?;
        // SAFETY: the test keeps these buffers alive and synchronizes before reading the results.
        unsafe {
            fused_moe_bf16(
                &stream,
                out.device_pointer(),
                input.device_pointer(),
                weight.device_pointer(),
                routed_weight.device_pointer(),
                sorted_token_ids.device_pointer(),
                expert_ids.device_pointer(),
                num_tokens_post_pad.device_pointer(),
                tokens * top_k,
                columns,
                reduction,
                top_k,
                block_size,
                reduction,
                columns * reduction,
                reduction,
                columns,
                expected_sorted.len(),
                false,
            )
        }?;

        let actual = out
            .to_host_vec()
            .sync_on(&stream)?
            .iter()
            .copied()
            .map(f32::from)
            .collect::<Vec<_>>();
        singe_core::assert_close!(&actual, &expected, 4e-3);
        Ok(())
    }
}
