//! Mixture-of-experts routing and alignment helpers.

#[cfg(feature = "dtype_bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceOutputView},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{input_pointer, write_only_pointer},
    error::{Error, Result},
    utility::{checked_i32_value, ensure_len_at_least},
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

#[cfg(feature = "dtype_f8")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedMoeBlockScaledConfig {
    pub base: FusedMoeConfig,
    pub group_n: usize,
    pub group_k: usize,
}

impl MoeAlignBlockSizeConfig {
    /// Number of i32 scratch elements, initialized on every launch and replay.
    pub fn workspace_len(self) -> Result<usize> {
        let chunks = self.element_count.div_ceil(256);
        let length = self
            .expert_count
            .checked_mul(chunks.checked_add(1).ok_or(Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        checked_i32_value(length)?;
        Ok(length)
    }

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
        if self.element_count == 0 || self.expert_count == 0 {
            return Err(Error::InvalidLength);
        }
        let padding = self.block_size.checked_sub(1).ok_or(Error::InvalidLength)?;
        let length = self
            .element_count
            .checked_add(
                self.expert_count
                    .checked_mul(padding)
                    .ok_or(Error::SizeOverflow)?,
            )
            .ok_or(Error::SizeOverflow)?;
        checked_i32_value(length)?;
        checked_i32_value(
            self.expert_count
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?,
        )?;
        Ok(length)
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
        ensure_len_at_least(
            cumsum_len,
            self.expert_count
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?,
        )?;
        ensure_len_at_least(max_expert_count_len, 1)
    }
}

#[cfg(feature = "dtype_f8")]
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
            0
        };
        ensure_len_at_least(routed_weight_len, routed_weight_min_len)?;
        ensure_len_at_least(sorted_token_ids_len, 1)?;
        ensure_len_at_least(expert_ids_len, 1)?;
        ensure_len_at_least(num_tokens_post_pad_len, 1)
    }
}

/// Groups top-k assignments by expert, with block padding and sentinel token IDs.
/// Produces the full `max_num_tokens_padded()` sorted prefix, `expert_count + 1`
/// cumulative offsets, and one element in each scalar destination.
/// Only `num_tokens_post_pad[0] / block_size` expert IDs are produced; unused
/// expert capacity and every destination tail remain untouched.
///
/// # Safety
///
/// Top-k IDs must be valid expert indices when read. Order preceding producers
/// before use and retain all buffers in the stream's context through completion
/// on every path and throughout replay. Output ranges must be mutually disjoint,
/// exclusive, and disjoint from the input.
pub unsafe fn moe_align_block_size_i32(
    stream: &Stream,
    sorted_token_ids: &mut impl DeviceOutput<i32>,
    expert_ids: &mut impl DeviceOutput<i32>,
    num_tokens_post_pad: &mut impl DeviceOutput<i32>,
    cumsum: &mut impl DeviceOutput<i32>,
    max_expert_count: &mut impl DeviceOutput<i32>,
    workspace: &mut impl DeviceOutput<i32>,
    topk_ids: &impl DeviceInput<i32>,
    config: MoeAlignBlockSizeConfig,
) -> Result<()> {
    ensure_len_at_least(workspace.len(), config.workspace_len()?)?;
    config.validate_lengths(
        sorted_token_ids.len(),
        expert_ids.len(),
        num_tokens_post_pad.len(),
        cumsum.len(),
        max_expert_count.len(),
        topk_ids.len(),
    )?;
    let sorted_len = config.max_num_tokens_padded()?;
    let blocks = config.max_num_blocks()?;
    let cumsum_len = config
        .expert_count
        .checked_add(1)
        .ok_or(Error::SizeOverflow)?;
    let mut sorted_token_ids =
        DeviceOutputView::from_output(sorted_token_ids).slice(..sorted_len)?;
    let mut expert_ids = DeviceOutputView::from_output(expert_ids).slice(..blocks)?;
    let mut num_tokens_post_pad = DeviceOutputView::from_output(num_tokens_post_pad).slice(..1)?;
    let mut cumsum = DeviceOutputView::from_output(cumsum).slice(..cumsum_len)?;
    let mut max_expert_count = DeviceOutputView::from_output(max_expert_count).slice(..1)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::moe::moe_align_block_size_i32(
            &stream,
            write_only_pointer(&mut sorted_token_ids),
            write_only_pointer(&mut expert_ids),
            write_only_pointer(&mut num_tokens_post_pad),
            write_only_pointer(&mut cumsum),
            write_only_pointer(&mut max_expert_count),
            write_only_pointer(workspace),
            workspace.len(),
            input_pointer(topk_ids),
            config.element_count,
            config.expert_count,
            config.block_size,
            sorted_token_ids.len(),
            expert_ids.len(),
        )
    }
}

/// Writes routed expert products into the assignment rows selected by the token IDs.
/// Only `columns` elements of each selected row are produced; row padding and
/// unselected rows remain untouched. Complete logical output requires every
/// assignment to occur once in the active routing prefix.
///
/// # Safety
///
/// Input values must be valid when read and their producers ordered before use.
/// The padded count must be a block multiple within the sorted buffer. Active
/// token IDs must be distinct and in `0..element_count`, or equal the padding
/// sentinel `element_count`; active expert IDs must be in `0..experts`.
/// Retain every allocation in the stream's context through completion on every
/// path and throughout replay. The output must be exclusive and disjoint from inputs.
/// Routing weights are not read when `mul_routed_weight` is false and may be empty.
pub unsafe fn fused_moe_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    weight: &impl DeviceInput<f32>,
    routed_weight: &impl DeviceInput<f32>,
    sorted_token_ids: &impl DeviceInput<i32>,
    expert_ids: &impl DeviceInput<i32>,
    num_tokens_post_pad: &impl DeviceInput<i32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::moe::fused_moe_f32(
            &stream,
            write_only_pointer(out),
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
}

#[cfg(feature = "dtype_f16")]
/// Writes routed expert products into the assignment rows selected by the token IDs.
/// Only `columns` elements of each selected row are produced; row padding and
/// unselected rows remain untouched. Complete logical output requires every
/// assignment to occur once in the active routing prefix.
///
/// # Safety
///
/// Input values must be valid when read and their producers ordered before use.
/// The padded count must be a block multiple within the sorted buffer. Active
/// token IDs must be distinct and in `0..element_count`, or equal the padding
/// sentinel `element_count`; active expert IDs must be in `0..experts`.
/// Retain every allocation in the stream's context through completion on every
/// path and throughout replay. The output must be exclusive and disjoint from inputs.
/// Routing weights are not read when `mul_routed_weight` is false and may be empty.
pub unsafe fn fused_moe_f16(
    stream: &Stream,
    out: &mut impl DeviceOutput<f16>,
    input: &impl DeviceInput<f16>,
    weight: &impl DeviceInput<f16>,
    routed_weight: &impl DeviceInput<f32>,
    sorted_token_ids: &impl DeviceInput<i32>,
    expert_ids: &impl DeviceInput<i32>,
    num_tokens_post_pad: &impl DeviceInput<i32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::moe::fused_moe_f16(
            &stream,
            write_only_pointer(out),
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
}

#[cfg(feature = "dtype_bf16")]
/// Writes routed expert products into the assignment rows selected by the token IDs.
/// Only `columns` elements of each selected row are produced; row padding and
/// unselected rows remain untouched. Complete logical output requires every
/// assignment to occur once in the active routing prefix.
///
/// # Safety
///
/// Input values must be valid when read and their producers ordered before use.
/// The padded count must be a block multiple within the sorted buffer. Active
/// token IDs must be distinct and in `0..element_count`, or equal the padding
/// sentinel `element_count`; active expert IDs must be in `0..experts`.
/// Retain every allocation in the stream's context through completion on every
/// path and throughout replay. The output must be exclusive and disjoint from inputs.
/// Routing weights are not read when `mul_routed_weight` is false and may be empty.
pub unsafe fn fused_moe_bf16(
    stream: &Stream,
    out: &mut impl DeviceOutput<bf16>,
    input: &impl DeviceInput<bf16>,
    weight: &impl DeviceInput<bf16>,
    routed_weight: &impl DeviceInput<f32>,
    sorted_token_ids: &impl DeviceInput<i32>,
    expert_ids: &impl DeviceInput<i32>,
    num_tokens_post_pad: &impl DeviceInput<i32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::moe::fused_moe_bf16(
            &stream,
            write_only_pointer(out),
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
}

#[cfg(feature = "dtype_f8")]
/// Writes routed expert products into the assignment rows selected by the token IDs.
/// Only `columns` elements of each selected row are produced; row padding and
/// unselected rows remain untouched. Complete logical output requires every
/// assignment to occur once in the active routing prefix.
///
/// # Safety
///
/// Input values must be valid when read and their producers ordered before use.
/// The padded count must be a block multiple within the sorted buffer. Active
/// token IDs must be distinct and in `0..element_count`, or equal the padding
/// sentinel `element_count`; active expert IDs must be in `0..experts`.
/// Retain every allocation in the stream's context through completion on every
/// path and throughout replay. The output must be exclusive and disjoint from inputs.
/// Routing weights are not read when `mul_routed_weight` is false and may be empty.
pub unsafe fn fused_moe_f8e4m3_block_scaled_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<u8>,
    weight: &impl DeviceInput<u8>,
    input_scales: &impl DeviceInput<f32>,
    weight_scales: &impl DeviceInput<f32>,
    routed_weight: &impl DeviceInput<f32>,
    sorted_token_ids: &impl DeviceInput<i32>,
    expert_ids: &impl DeviceInput<i32>,
    num_tokens_post_pad: &impl DeviceInput<i32>,
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
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::moe::fused_moe_f8e4m3_block_scaled_f32(
            &stream,
            write_only_pointer(out),
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
        || config.output_row_stride < config.columns
    {
        return Err(Error::InvalidLength);
    }
    let output_span = (config.element_count - 1)
        .checked_mul(config.output_row_stride)
        .and_then(|value| value.checked_add(config.columns))
        .ok_or(Error::SizeOverflow)?;
    let input_span = (config.token_count - 1)
        .checked_mul(config.input_row_stride)
        .and_then(|value| value.checked_add(config.reduction))
        .ok_or(Error::SizeOverflow)?;
    let weight_span = (config.experts - 1)
        .checked_mul(config.expert_stride)
        .and_then(|value| {
            (config.columns - 1)
                .checked_mul(config.weight_row_stride)
                .and_then(|row| value.checked_add(row))
        })
        .and_then(|value| value.checked_add(config.reduction))
        .ok_or(Error::SizeOverflow)?;
    checked_i32_value(output_span)?;
    checked_i32_value(input_span)?;
    checked_i32_value(weight_span)?;
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

#[cfg(feature = "dtype_f8")]
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
    if rhs == 0 {
        return Err(Error::InvalidLength);
    }
    Ok(lhs.div_ceil(rhs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_align_block_size_validation_accepts_exact_max_lengths() -> Result<()> {
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
    fn test_moe_align_block_size_validation_rejects_short_sorted_tokens() -> Result<()> {
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
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_moe_align_block_size_validation_rejects_zero_block_size() {
        assert!(matches!(
            MoeAlignBlockSizeConfig::create(7, 4, 0),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn test_fused_moe_validation_accepts_topk1_without_routed_weights() -> Result<()> {
        let config = FusedMoeConfig::contiguous(5, 3, 4, 6, 1, 2, false)?;
        config.validate_lengths(20, 30, 72, 1, 1, 1, 1)
    }

    #[test]
    fn test_fused_moe_validation_rejects_short_routed_weights_when_enabled() -> Result<()> {
        let config = FusedMoeConfig::contiguous(5, 3, 4, 6, 2, 2, true)?;
        assert!(matches!(
            config.validate_lengths(40, 30, 72, 9, 1, 1, 1),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_fused_moe_validation_rejects_inconsistent_element_count() {
        let mut config = FusedMoeConfig::contiguous(5, 3, 4, 6, 2, 2, true).unwrap();
        config.element_count -= 1;
        assert!(matches!(
            validate_fused_moe_config(config),
            Err(Error::InvalidLength)
        ));
    }

    #[cfg(feature = "dtype_f8")]
    #[test]
    fn test_fused_moe_block_scaled_validation_accepts_exact_scale_lengths() -> Result<()> {
        let config = FusedMoeBlockScaledConfig::contiguous(5, 3, 7, 6, 2, 2, 4, 3, true)?;
        config.validate_lengths(70, 30, 126, 10, 12, 10, 1, 1, 1)
    }

    #[cfg(feature = "dtype_f8")]
    #[test]
    fn test_fused_moe_block_scaled_validation_rejects_short_weight_scales() -> Result<()> {
        let config = FusedMoeBlockScaledConfig::contiguous(5, 3, 7, 6, 2, 2, 4, 3, true)?;
        assert!(matches!(
            config.validate_lengths(70, 30, 126, 10, 11, 10, 1, 1, 1),
            Err(Error::LengthMismatch
                | Error::LengthMismatchFor { .. }
                | Error::LengthTooShort { .. }
                | Error::RankReachMismatch { .. })
        ));
        Ok(())
    }

    #[cfg(feature = "dtype_f8")]
    #[test]
    fn test_fused_moe_block_scaled_validation_rejects_zero_group() {
        assert!(matches!(
            FusedMoeBlockScaledConfig::contiguous(5, 3, 7, 6, 2, 2, 0, 3, true),
            Err(Error::InvalidLength)
        ));
    }
}
