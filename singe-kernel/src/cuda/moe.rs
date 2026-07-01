//! Mixture-of-experts routing and alignment helpers.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::ensure_len_at_least,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeAlignBlockSizeConfig {
    pub element_count: usize,
    pub expert_count: usize,
    pub block_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedMoeConfig {
    pub element_count: usize,
    pub token_count: usize,
    pub experts: usize,
    pub columns: usize,
    pub reduction: usize,
    pub top_k: usize,
    pub block_size: usize,
    pub input_row_stride: usize,
    pub expert_stride: usize,
    pub weight_row_stride: usize,
    pub output_row_stride: usize,
    pub mul_routed_weight: bool,
}

#[cfg(feature = "dtype-f8")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedMoeBlockScaledConfig {
    pub base: FusedMoeConfig,
    pub group_n: usize,
    pub group_k: usize,
}

impl MoeAlignBlockSizeConfig {
    pub fn create(element_count: usize, expert_count: usize, block_size: usize) -> Result<Self> {
        let config = Self {
            element_count,
            expert_count,
            block_size,
        };
        validate_moe_align_block_size_config(config)?;
        Ok(config)
    }

    pub fn max_num_tokens_padded(self) -> Result<usize> {
        self.element_count
            .checked_add(
                self.expert_count
                    .checked_mul(self.block_size - 1)
                    .ok_or(Error::SizeOverflow)?,
            )
            .ok_or(Error::SizeOverflow)
    }

    pub fn max_num_blocks(self) -> Result<usize> {
        ceil_div(self.max_num_tokens_padded()?, self.block_size)
    }

    fn validate_lengths(
        self,
        sorted_token_ids_len: usize,
        expert_ids_len: usize,
        num_tokens_post_pad_len: usize,
        cumsum_len: usize,
        max_expert_count_len: usize,
        topk_ids_len: usize,
    ) -> Result<()> {
        validate_moe_align_block_size_config(self)?;
        ensure_len_at_least(topk_ids_len, self.element_count)?;
        ensure_len_at_least(sorted_token_ids_len, self.max_num_tokens_padded()?)?;
        ensure_len_at_least(expert_ids_len, self.max_num_blocks()?)?;
        ensure_len_at_least(num_tokens_post_pad_len, 1)?;
        ensure_len_at_least(cumsum_len, self.expert_count + 1)?;
        ensure_len_at_least(max_expert_count_len, 1)
    }
}

#[cfg(feature = "dtype-f8")]
impl FusedMoeBlockScaledConfig {
    pub fn contiguous(
        token_count: usize,
        experts: usize,
        columns: usize,
        reduction: usize,
        top_k: usize,
        block_size: usize,
        group_n: usize,
        group_k: usize,
        mul_routed_weight: bool,
    ) -> Result<Self> {
        let config = Self {
            base: FusedMoeConfig::contiguous(
                token_count,
                experts,
                columns,
                reduction,
                top_k,
                block_size,
                mul_routed_weight,
            )?,
            group_n,
            group_k,
        };
        validate_fused_moe_block_scaled_config(config)?;
        Ok(config)
    }

    pub fn k_groups(self) -> Result<usize> {
        ceil_div(self.base.reduction, self.group_k)
    }

    pub fn n_groups(self) -> Result<usize> {
        ceil_div(self.base.columns, self.group_n)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        input_len: usize,
        weight_len: usize,
        input_scales_len: usize,
        weight_scales_len: usize,
        routed_weight_len: usize,
        sorted_token_ids_len: usize,
        expert_ids_len: usize,
        num_tokens_post_pad_len: usize,
    ) -> Result<()> {
        validate_fused_moe_block_scaled_config(self)?;
        self.base.validate_lengths(
            out_len,
            input_len,
            weight_len,
            routed_weight_len,
            sorted_token_ids_len,
            expert_ids_len,
            num_tokens_post_pad_len,
        )?;
        let k_groups = self.k_groups()?;
        ensure_len_at_least(
            input_scales_len,
            self.base
                .token_count
                .checked_mul(k_groups)
                .ok_or(Error::SizeOverflow)?,
        )?;
        ensure_len_at_least(
            weight_scales_len,
            self.base
                .experts
                .checked_mul(self.n_groups()?)
                .and_then(|groups| groups.checked_mul(k_groups))
                .ok_or(Error::SizeOverflow)?,
        )
    }
}

impl FusedMoeConfig {
    pub fn contiguous(
        token_count: usize,
        experts: usize,
        columns: usize,
        reduction: usize,
        top_k: usize,
        block_size: usize,
        mul_routed_weight: bool,
    ) -> Result<Self> {
        let config = Self {
            element_count: token_count.checked_mul(top_k).ok_or(Error::SizeOverflow)?,
            token_count,
            experts,
            columns,
            reduction,
            top_k,
            block_size,
            input_row_stride: reduction,
            expert_stride: columns.checked_mul(reduction).ok_or(Error::SizeOverflow)?,
            weight_row_stride: reduction,
            output_row_stride: columns,
            mul_routed_weight,
        };
        validate_fused_moe_config(config)?;
        Ok(config)
    }

    fn validate_lengths(
        self,
        out_len: usize,
        input_len: usize,
        weight_len: usize,
        routed_weight_len: usize,
        sorted_token_ids_len: usize,
        expert_ids_len: usize,
        num_tokens_post_pad_len: usize,
    ) -> Result<()> {
        validate_fused_moe_config(self)?;
        ensure_len_at_least(
            out_len,
            self.element_count
                .checked_sub(1)
                .and_then(|row| row.checked_mul(self.output_row_stride))
                .and_then(|base| base.checked_add(self.columns))
                .ok_or(Error::SizeOverflow)?,
        )?;
        ensure_len_at_least(
            input_len,
            self.token_count
                .checked_sub(1)
                .and_then(|row| row.checked_mul(self.input_row_stride))
                .and_then(|base| base.checked_add(self.reduction))
                .ok_or(Error::SizeOverflow)?,
        )?;
        ensure_len_at_least(
            weight_len,
            self.experts
                .checked_sub(1)
                .and_then(|expert| expert.checked_mul(self.expert_stride))
                .and_then(|base| {
                    base.checked_add(
                        self.columns
                            .checked_sub(1)?
                            .checked_mul(self.weight_row_stride)?,
                    )
                })
                .and_then(|base| base.checked_add(self.reduction))
                .ok_or(Error::SizeOverflow)?,
        )?;
        let routed_weight_min_len = if self.mul_routed_weight {
            self.element_count
        } else {
            1
        };
        ensure_len_at_least(routed_weight_len, routed_weight_min_len)?;
        ensure_len_at_least(sorted_token_ids_len, 1)?;
        ensure_len_at_least(expert_ids_len, 1)?;
        ensure_len_at_least(num_tokens_post_pad_len, 1)
    }
}

pub fn moe_align_block_size_i32(
    stream: &Stream,
    sorted_token_ids: &mut impl DeviceSliceMut<i32>,
    expert_ids: &mut impl DeviceSliceMut<i32>,
    num_tokens_post_pad: &mut impl DeviceSliceMut<i32>,
    cumsum: &mut impl DeviceSliceMut<i32>,
    max_expert_count: &mut impl DeviceSliceMut<i32>,
    topk_ids: &impl DeviceSlice<i32>,
    config: MoeAlignBlockSizeConfig,
) -> Result<()> {
    config.validate_lengths(
        sorted_token_ids.len(),
        expert_ids.len(),
        num_tokens_post_pad.len(),
        cumsum.len(),
        max_expert_count.len(),
        topk_ids.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::moe::moe_align_block_size_i32(
        &stream,
        output_pointer(sorted_token_ids),
        output_pointer(expert_ids),
        output_pointer(num_tokens_post_pad),
        output_pointer(cumsum),
        output_pointer(max_expert_count),
        input_pointer(topk_ids),
        config.element_count,
        config.expert_count,
        config.block_size,
        sorted_token_ids.len(),
        expert_ids.len(),
    )
}

pub fn fused_moe_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    weight: &impl DeviceSlice<f32>,
    routed_weight: &impl DeviceSlice<f32>,
    sorted_token_ids: &impl DeviceSlice<i32>,
    expert_ids: &impl DeviceSlice<i32>,
    num_tokens_post_pad: &impl DeviceSlice<i32>,
    config: FusedMoeConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        input.len(),
        weight.len(),
        routed_weight.len(),
        sorted_token_ids.len(),
        expert_ids.len(),
        num_tokens_post_pad.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::moe::fused_moe_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(routed_weight),
        input_pointer(sorted_token_ids),
        input_pointer(expert_ids),
        input_pointer(num_tokens_post_pad),
        config.element_count,
        config.columns,
        config.reduction,
        config.top_k,
        config.block_size,
        config.input_row_stride,
        config.expert_stride,
        config.weight_row_stride,
        config.output_row_stride,
        sorted_token_ids.len(),
        config.mul_routed_weight,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn fused_moe_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    weight: &impl DeviceSlice<f16>,
    routed_weight: &impl DeviceSlice<f32>,
    sorted_token_ids: &impl DeviceSlice<i32>,
    expert_ids: &impl DeviceSlice<i32>,
    num_tokens_post_pad: &impl DeviceSlice<i32>,
    config: FusedMoeConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        input.len(),
        weight.len(),
        routed_weight.len(),
        sorted_token_ids.len(),
        expert_ids.len(),
        num_tokens_post_pad.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::moe::fused_moe_f16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(routed_weight),
        input_pointer(sorted_token_ids),
        input_pointer(expert_ids),
        input_pointer(num_tokens_post_pad),
        config.element_count,
        config.columns,
        config.reduction,
        config.top_k,
        config.block_size,
        config.input_row_stride,
        config.expert_stride,
        config.weight_row_stride,
        config.output_row_stride,
        sorted_token_ids.len(),
        config.mul_routed_weight,
    )
}

#[cfg(feature = "dtype-bf16")]
pub fn fused_moe_bf16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<bf16>,
    input: &impl DeviceSlice<bf16>,
    weight: &impl DeviceSlice<bf16>,
    routed_weight: &impl DeviceSlice<f32>,
    sorted_token_ids: &impl DeviceSlice<i32>,
    expert_ids: &impl DeviceSlice<i32>,
    num_tokens_post_pad: &impl DeviceSlice<i32>,
    config: FusedMoeConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        input.len(),
        weight.len(),
        routed_weight.len(),
        sorted_token_ids.len(),
        expert_ids.len(),
        num_tokens_post_pad.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::moe::fused_moe_bf16(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(routed_weight),
        input_pointer(sorted_token_ids),
        input_pointer(expert_ids),
        input_pointer(num_tokens_post_pad),
        config.element_count,
        config.columns,
        config.reduction,
        config.top_k,
        config.block_size,
        config.input_row_stride,
        config.expert_stride,
        config.weight_row_stride,
        config.output_row_stride,
        sorted_token_ids.len(),
        config.mul_routed_weight,
    )
}

#[cfg(feature = "dtype-f8")]

pub fn fused_moe_f8e4m3_block_scaled_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<u8>,
    weight: &impl DeviceSlice<u8>,
    input_scales: &impl DeviceSlice<f32>,
    weight_scales: &impl DeviceSlice<f32>,
    routed_weight: &impl DeviceSlice<f32>,
    sorted_token_ids: &impl DeviceSlice<i32>,
    expert_ids: &impl DeviceSlice<i32>,
    num_tokens_post_pad: &impl DeviceSlice<i32>,
    config: FusedMoeBlockScaledConfig,
) -> Result<()> {
    config.validate_lengths(
        out.len(),
        input.len(),
        weight.len(),
        input_scales.len(),
        weight_scales.len(),
        routed_weight.len(),
        sorted_token_ids.len(),
        expert_ids.len(),
        num_tokens_post_pad.len(),
    )?;
    let stream = borrowed_stream(stream)?;
    cutile::moe::fused_moe_f8e4m3_block_scaled_f32(
        &stream,
        output_pointer(out),
        input_pointer(input),
        input_pointer(weight),
        input_pointer(input_scales),
        input_pointer(weight_scales),
        input_pointer(routed_weight),
        input_pointer(sorted_token_ids),
        input_pointer(expert_ids),
        input_pointer(num_tokens_post_pad),
        config.base.element_count,
        config.base.columns,
        config.base.reduction,
        config.base.top_k,
        config.base.block_size,
        config.group_n,
        config.group_k,
        config.base.input_row_stride,
        config.base.expert_stride,
        config.base.weight_row_stride,
        config.base.output_row_stride,
        sorted_token_ids.len(),
        config.base.mul_routed_weight,
    )
}

fn validate_moe_align_block_size_config(config: MoeAlignBlockSizeConfig) -> Result<()> {
    if config.element_count == 0 || config.expert_count == 0 || config.block_size == 0 {
        return Err(Error::InvalidLength);
    }
    config.max_num_tokens_padded()?;
    config.max_num_blocks()?;
    Ok(())
}

fn validate_fused_moe_config(config: FusedMoeConfig) -> Result<()> {
    if config.element_count == 0
        || config.token_count == 0
        || config.experts == 0
        || config.columns == 0
        || config.reduction == 0
        || config.top_k == 0
        || config.block_size == 0
        || config.input_row_stride == 0
        || config.expert_stride == 0
        || config.weight_row_stride == 0
        || config.output_row_stride == 0
    {
        return Err(Error::InvalidLength);
    }
    if config.element_count
        != config
            .token_count
            .checked_mul(config.top_k)
            .ok_or(Error::SizeOverflow)?
    {
        return Err(Error::InvalidLength);
    }
    Ok(())
}

#[cfg(feature = "dtype-f8")]
fn validate_fused_moe_block_scaled_config(config: FusedMoeBlockScaledConfig) -> Result<()> {
    validate_fused_moe_config(config.base)?;
    if config.group_n == 0 || config.group_k == 0 {
        return Err(Error::InvalidLength);
    }
    config.k_groups()?;
    config.n_groups()?;
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
    use super::*;

    #[test]
    fn moe_align_block_size_validation_accepts_exact_max_lengths() -> Result<()> {
        let config = MoeAlignBlockSizeConfig::create(7, 4, 3)?;
        config.validate_lengths(
            config.max_num_tokens_padded()?,
            config.max_num_blocks()?,
            1,
            5,
            1,
            7,
        )
    }

    #[test]
    fn moe_align_block_size_validation_rejects_short_sorted_tokens() -> Result<()> {
        let config = MoeAlignBlockSizeConfig::create(7, 4, 3)?;
        assert!(matches!(
            config.validate_lengths(
                config.max_num_tokens_padded()? - 1,
                config.max_num_blocks()?,
                1,
                5,
                1,
                7
            ),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn moe_align_block_size_validation_rejects_zero_block_size() {
        assert!(matches!(
            MoeAlignBlockSizeConfig::create(7, 4, 0),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn fused_moe_validation_accepts_topk1_without_routed_weights() -> Result<()> {
        let config = FusedMoeConfig::contiguous(5, 3, 4, 6, 1, 2, false)?;
        config.validate_lengths(20, 30, 72, 1, 1, 1, 1)
    }

    #[test]
    fn fused_moe_validation_rejects_short_routed_weights_when_enabled() -> Result<()> {
        let config = FusedMoeConfig::contiguous(5, 3, 4, 6, 2, 2, true)?;
        assert!(matches!(
            config.validate_lengths(40, 30, 72, 9, 1, 1, 1),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[test]
    fn fused_moe_validation_rejects_inconsistent_element_count() {
        let mut config = FusedMoeConfig::contiguous(5, 3, 4, 6, 2, 2, true).unwrap();
        config.element_count -= 1;
        assert!(matches!(
            validate_fused_moe_config(config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn fused_moe_block_scaled_validation_accepts_exact_scale_lengths() -> Result<()> {
        let config = FusedMoeBlockScaledConfig::contiguous(5, 3, 7, 6, 2, 2, 4, 3, true)?;
        config.validate_lengths(70, 30, 126, 10, 12, 10, 1, 1, 1)
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn fused_moe_block_scaled_validation_rejects_short_weight_scales() -> Result<()> {
        let config = FusedMoeBlockScaledConfig::contiguous(5, 3, 7, 6, 2, 2, 4, 3, true)?;
        assert!(matches!(
            config.validate_lengths(70, 30, 126, 10, 11, 10, 1, 1, 1),
            Err(Error::LengthMismatch)
        ));
        Ok(())
    }

    #[cfg(feature = "dtype-f8")]
    #[test]
    fn fused_moe_block_scaled_validation_rejects_zero_group() {
        assert!(matches!(
            FusedMoeBlockScaledConfig::contiguous(5, 3, 7, 6, 2, 2, 0, 3, true),
            Err(Error::InvalidLength)
        ));
    }
}
