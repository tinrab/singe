use crate::error::{Error, Result};

#[cfg(feature = "cutile")]
pub fn checked_i32_value(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::LengthExceedsI32)
}

pub fn checked_element_count(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols).ok_or(Error::SizeOverflow)
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_attention",
    feature = "kernel_audio",
    feature = "kernel_cast",
    feature = "kernel_conv",
    feature = "kernel_elementwise",
    feature = "kernel_fft",
    feature = "kernel_fused",
    feature = "kernel_gather",
    feature = "kernel_mask",
    feature = "kernel_matmul",
    feature = "kernel_normalization",
    feature = "kernel_pooling",
    feature = "kernel_positional",
    feature = "kernel_quantization",
    feature = "kernel_recurrent",
    feature = "kernel_reduction",
    feature = "kernel_scalar",
    feature = "kernel_shape",
    feature = "kernel_softmax",
    feature = "kernel_unary",
))]
#[allow(
    dead_code,
    reason = "unused under some kernel and dtype feature combinations"
)]
pub fn ensure_len(actual: usize, expected: usize) -> Result<()> {
    ensure_named_len("length", actual, expected)
}

#[cfg(any(
    feature = "kernel_activation",
    feature = "kernel_attention",
    feature = "kernel_audio",
    feature = "kernel_cast",
    feature = "kernel_conv",
    feature = "kernel_elementwise",
    feature = "kernel_fft",
    feature = "kernel_fused",
    feature = "kernel_gather",
    feature = "kernel_mask",
    feature = "kernel_matmul",
    feature = "kernel_normalization",
    feature = "kernel_pooling",
    feature = "kernel_positional",
    feature = "kernel_quantization",
    feature = "kernel_recurrent",
    feature = "kernel_reduction",
    feature = "kernel_scalar",
    feature = "kernel_shape",
    feature = "kernel_softmax",
    feature = "kernel_unary",
))]
#[allow(
    dead_code,
    reason = "unused under some kernel and dtype feature combinations"
)]
pub fn ensure_named_len(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::LengthMismatchFor {
            name: name.into(),
            expected,
            actual,
        })
    }
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_positional",
    feature = "kernel_recurrent",
    feature = "kernel_shape",
))]
pub fn checked_rank4_len(dimensions: [usize; 4]) -> Result<usize> {
    let len = checked_element_count(dimensions[0], dimensions[1])?;
    let len = checked_element_count(len, dimensions[2])?;
    checked_element_count(len, dimensions[3])
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
pub fn ensure_rank4_reach(
    input_len: usize,
    dimensions: [usize; 4],
    strides: [usize; 4],
) -> Result<()> {
    ensure_rank4_reach_named("input", input_len, dimensions, strides)
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
pub fn ensure_rank4_reach_named(
    name: &'static str,
    input_len: usize,
    dimensions: [usize; 4],
    strides: [usize; 4],
) -> Result<()> {
    ensure_rank_reach(name, input_len, dimensions, strides)
}

#[cfg(any(
    feature = "kernel_matmul",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
pub fn ensure_rank3_reach(
    input_len: usize,
    dimensions: [usize; 3],
    strides: [usize; 3],
) -> Result<()> {
    ensure_rank3_reach_named("input", input_len, dimensions, strides)
}

#[cfg(any(
    feature = "kernel_matmul",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
pub fn ensure_rank3_reach_named(
    name: &'static str,
    input_len: usize,
    dimensions: [usize; 3],
    strides: [usize; 3],
) -> Result<()> {
    ensure_rank_reach(name, input_len, dimensions, strides)
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
pub fn ensure_rank2_reach(
    input_len: usize,
    dimensions: [usize; 2],
    strides: [usize; 2],
) -> Result<()> {
    ensure_rank2_reach_named("input", input_len, dimensions, strides)
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
pub fn ensure_rank2_reach_named(
    name: &'static str,
    input_len: usize,
    dimensions: [usize; 2],
    strides: [usize; 2],
) -> Result<()> {
    ensure_rank_reach(name, input_len, dimensions, strides)
}

#[cfg(any(
    feature = "kernel_fused",
    feature = "kernel_matmul",
    feature = "kernel_positional",
    feature = "kernel_shape",
))]
fn ensure_rank_reach<const R: usize>(
    name: &'static str,
    input_len: usize,
    dimensions: [usize; R],
    strides: [usize; R],
) -> Result<()> {
    if dimensions.contains(&0) {
        return Ok(());
    }
    let mut max_offset = 0usize;
    for axis in 0..R {
        let span = dimensions[axis]
            .checked_sub(1)
            .ok_or(Error::SizeOverflow)?
            .checked_mul(strides[axis])
            .ok_or(Error::SizeOverflow)?;
        max_offset = max_offset.checked_add(span).ok_or(Error::SizeOverflow)?;
    }
    let required = max_offset.checked_add(1).ok_or(Error::SizeOverflow)?;
    if required <= input_len {
        Ok(())
    } else {
        Err(Error::RankReachMismatch {
            name: name.into(),
            rank: R,
            required,
            actual: input_len,
        })
    }
}

#[cfg(any(feature = "kernel_positional", feature = "kernel_shape"))]
pub fn checked_rank4_i32(values: [usize; 4]) -> Result<[i32; 4]> {
    Ok([
        checked_i32_value(values[0])?,
        checked_i32_value(values[1])?,
        checked_i32_value(values[2])?,
        checked_i32_value(values[3])?,
    ])
}

#[cfg(any(feature = "kernel_positional", feature = "kernel_shape"))]
pub fn checked_rank3_i32(values: [usize; 3]) -> Result<[i32; 3]> {
    Ok([
        checked_i32_value(values[0])?,
        checked_i32_value(values[1])?,
        checked_i32_value(values[2])?,
    ])
}

#[cfg(feature = "kernel_shape")]
pub fn checked_rank2_i32(values: [usize; 2]) -> Result<[i32; 2]> {
    Ok([checked_i32_value(values[0])?, checked_i32_value(values[1])?])
}

#[cfg(any(
    feature = "kernel_attention",
    feature = "kernel_matmul",
    feature = "kernel_moe",
    all(feature = "kernel_fused", feature = "kernel_quantization"),
    all(feature = "kernel_positional", feature = "dtype_f16"),
))]
pub fn ensure_len_at_least(actual: usize, minimum: usize) -> Result<()> {
    ensure_named_len_at_least("length", actual, minimum)
}

#[cfg(any(
    feature = "kernel_attention",
    feature = "kernel_matmul",
    feature = "kernel_moe",
    all(feature = "kernel_fused", feature = "kernel_quantization"),
    all(feature = "kernel_positional", feature = "dtype_f16"),
))]
pub fn ensure_named_len_at_least(name: &'static str, actual: usize, minimum: usize) -> Result<()> {
    if actual < minimum {
        Err(Error::LengthTooShort {
            name: name.into(),
            minimum,
            actual,
        })
    } else {
        Ok(())
    }
}
