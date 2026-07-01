use std::sync::Arc;

#[cfg(feature = "dtype-f8")]
use cutile::core::f8e4m3fn;
#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        kernel::moe as kernel_moe,
        utility::{checked_device_pointer, raw_vector_grid},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MoeAlignBlockSize {
    element_count: i32,
    expert_count: i32,
    block_size: i32,
    sorted_token_ids_len: i32,
    expert_ids_len: i32,
    output_len: i32,
    grid: (u32, u32, u32),
}

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
    output_len: i32,
    grid: (u32, u32, u32),
}

#[cfg(feature = "dtype-f8")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusedMoeBlockScaled {
    base: FusedMoe,
    group_n: i32,
    group_k: i32,
    k_groups: i32,
}

impl MoeAlignBlockSize {
    fn create(
        element_count: usize,
        expert_count: usize,
        block_size: usize,
        sorted_token_ids_len: usize,
        expert_ids_len: usize,
    ) -> Result<Self> {
        if element_count == 0
            || expert_count == 0
            || block_size == 0
            || sorted_token_ids_len == 0
            || expert_ids_len == 0
        {
            return Err(Error::InvalidLength);
        }
        let max_num_tokens_padded = element_count
            .checked_add(
                expert_count
                    .checked_mul(block_size - 1)
                    .ok_or(Error::SizeOverflow)?,
            )
            .ok_or(Error::SizeOverflow)?;
        if sorted_token_ids_len < max_num_tokens_padded {
            return Err(Error::InvalidLength);
        }
        let max_blocks = ceil_div(max_num_tokens_padded, block_size)?;
        if expert_ids_len < max_blocks {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(
            sorted_token_ids_len
                .max(expert_ids_len)
                .max(expert_count + 1),
            1,
        )?;
        Ok(Self {
            element_count: checked_i32_value(element_count)?,
            expert_count: checked_i32_value(expert_count)?,
            block_size: checked_i32_value(block_size)?,
            sorted_token_ids_len: checked_i32_value(sorted_token_ids_len)?,
            expert_ids_len: checked_i32_value(expert_ids_len)?,
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

#[cfg(feature = "dtype-f8")]
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
            || output_row_stride == 0
            || sorted_token_ids_len == 0
        {
            return Err(Error::InvalidLength);
        }
        let output_len = checked_element_count(sorted_token_ids_len, columns)?;
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
            output_len: checked_i32_value(output_len)?,
            grid: raw_vector_grid(output_len)?,
        })
    }
}

pub fn moe_align_block_size_i32(
    stream: &Arc<Stream>,
    sorted_token_ids: DevicePointer<i32>,
    expert_ids: DevicePointer<i32>,
    num_tokens_post_pad: DevicePointer<i32>,
    cumsum: DevicePointer<i32>,
    max_expert_count: DevicePointer<i32>,
    topk_ids: DevicePointer<i32>,
    element_count: usize,
    expert_count: usize,
    block_size: usize,
    sorted_token_ids_len: usize,
    expert_ids_len: usize,
) -> Result<()> {
    checked_device_pointer(sorted_token_ids)?;
    checked_device_pointer(expert_ids)?;
    checked_device_pointer(num_tokens_post_pad)?;
    checked_device_pointer(cumsum)?;
    checked_device_pointer(max_expert_count)?;
    checked_device_pointer(topk_ids)?;

    let params = MoeAlignBlockSize::create(
        element_count,
        expert_count,
        block_size,
        sorted_token_ids_len,
        expert_ids_len,
    )?;
    unsafe {
        kernel_moe::moe_align_block_size_i32(
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum,
            max_expert_count,
            params.element_count,
            params.expert_count,
            params.block_size,
            params.sorted_token_ids_len,
            params.expert_ids_len,
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

pub fn fused_moe_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(routed_weight)?;
    checked_device_pointer(sorted_token_ids)?;
    checked_device_pointer(expert_ids)?;
    checked_device_pointer(num_tokens_post_pad)?;

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
    unsafe {
        kernel_moe::fused_moe_f32(
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
            params.output_len,
        )
    }
    .grid(params.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(any(feature = "dtype-f16", feature = "dtype-bf16"))]
macro_rules! fused_moe_half_fn {
    ($name:ident, $ty:ty, $kernel:ident) => {
        pub fn $name(
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
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(weight)?;
            checked_device_pointer(routed_weight)?;
            checked_device_pointer(sorted_token_ids)?;
            checked_device_pointer(expert_ids)?;
            checked_device_pointer(num_tokens_post_pad)?;

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
            unsafe {
                kernel_moe::$kernel(
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
                    params.output_len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
fused_moe_half_fn!(fused_moe_f16, f16, fused_moe_f16);
#[cfg(feature = "dtype-bf16")]
fused_moe_half_fn!(fused_moe_bf16, bf16, fused_moe_bf16);

#[cfg(feature = "dtype-f8")]

pub fn fused_moe_f8e4m3_block_scaled_f32(
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
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(weight)?;
    checked_device_pointer(input_scales)?;
    checked_device_pointer(weight_scales)?;
    checked_device_pointer(routed_weight)?;
    checked_device_pointer(sorted_token_ids)?;
    checked_device_pointer(expert_ids)?;
    checked_device_pointer(num_tokens_post_pad)?;

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
    let input = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(input.cu_deviceptr()) };
    let weight = unsafe { DevicePointer::<f8e4m3fn>::from_cu_deviceptr(weight.cu_deviceptr()) };
    unsafe {
        kernel_moe::fused_moe_f8e4m3_block_scaled_f32(
            out,
            input,
            weight,
            input_scales,
            weight_scales,
            routed_weight,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            params.base.element_count,
            params.base.columns,
            params.base.reduction,
            params.base.top_k,
            params.base.block_size,
            params.group_n,
            params.group_k,
            params.k_groups,
            params.base.input_row_stride,
            params.base.expert_stride,
            params.base.weight_row_stride,
            params.base.output_row_stride,
            params.base.mul_routed_weight,
            params.base.output_len,
        )
    }
    .grid(params.base.grid)
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
    fn moe_align_block_size_i32() -> Result<()> {
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

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            element_count,
            expert_count,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;

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
    fn fused_moe_f32_topk2() -> Result<()> {
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
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn fused_moe_f32_topk2_with_empty_experts() -> Result<()> {
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
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn fused_moe_f8e4m3_block_scaled_f32_topk2() -> Result<()> {
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
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn fused_moe_f16_topk2() -> Result<()> {
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
        let out = api::zeros::<f16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

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

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn fused_moe_bf16_topk2() -> Result<()> {
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
        let out = api::zeros::<bf16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

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
    fn fused_moe_f32_topk1_without_routed_weight() -> Result<()> {
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
        let out = api::zeros::<f32>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn fused_moe_f16_topk1_without_routed_weight() -> Result<()> {
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
        let out = api::zeros::<f16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

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

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn fused_moe_bf16_topk1_without_routed_weight() -> Result<()> {
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
        let out = api::zeros::<bf16>(&[tokens * top_k * columns]).sync_on(&stream)?;

        moe_align_block_size_i32(
            &stream,
            sorted_token_ids.device_pointer(),
            expert_ids.device_pointer(),
            num_tokens_post_pad.device_pointer(),
            cumsum.device_pointer(),
            max_expert_count.device_pointer(),
            topk_ids.device_pointer(),
            tokens * top_k,
            experts,
            block_size,
            expected_sorted.len(),
            expected_experts_padded.len(),
        )?;
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
        )?;

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
