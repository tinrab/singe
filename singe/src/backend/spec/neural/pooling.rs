use crate::{
    backend::spec::{validate_non_negative, validate_positive},
    error::{Error, Result},
    shape::{
        Shape,
        spatial::{Padding2d, Spatial2d, Spatial2dI64},
    },
    utility::{checked_add_i64, checked_mul_i64, checked_sub_i64},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolReduction {
    Maximum,
    Average,
}

/// Validated two-dimensional pooling dimensions for channel-height-width or batch-channel-height-width tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pool2dProblem {
    /// Inferred output shape with the same rank as the input.
    pub output_shape: Shape,
    /// Input and output rank, either three or four.
    pub rank: usize,
    /// Batch count from rank-four input, or one for rank-three input.
    pub batch: i64,
    /// Positive channel count.
    pub channels: i64,
    /// Positive input image height.
    pub input_height: i64,
    /// Positive input image width.
    pub input_width: i64,
    /// Inferred output height.
    pub output_height: i64,
    /// Inferred output width.
    pub output_width: i64,
    /// Positive pooling-window height.
    pub kernel_height: i64,
    /// Positive pooling-window width.
    pub kernel_width: i64,
    /// Positive vertical step between windows.
    pub stride_height: i64,
    /// Positive horizontal step between windows.
    pub stride_width: i64,
    /// Nonnegative padding above the input.
    pub pad_height_start: i64,
    /// Nonnegative padding below the input.
    pub pad_height_end: i64,
    /// Nonnegative padding to the left of the input.
    pub pad_width_start: i64,
    /// Nonnegative padding to the right of the input.
    pub pad_width_end: i64,
    /// Positive vertical spacing within the pooling window.
    pub dilation_height: i64,
    /// Positive horizontal spacing within the pooling window.
    pub dilation_width: i64,
}

/// Validates pooling options and infers spatial output dimensions.
///
/// Average pooling requires unit dilation.
/// Each side's padding is at most half the corresponding undilated kernel size.
/// `diagnostic_prefix` identifies operation-specific errors.
///
/// # Errors
///
/// Returns an error if rank is not three or four, average-pooling dilation is not unit, batch or padding is negative, channels, spatial extents, kernel, stride, or dilation are nonpositive, padding is too large, output dimensions are invalid, or shape arithmetic overflows.
pub fn pool2d_problem(
    input: &Shape,
    kernel_size: Spatial2dI64,
    stride: Spatial2dI64,
    padding: Padding2d,
    dilation: Spatial2dI64,
    ceil_mode: bool,
    reduction: PoolReduction,
    diagnostic_prefix: &'static str,
) -> Result<Pool2dProblem> {
    if reduction == PoolReduction::Average && dilation != Spatial2d::square(1) {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} avg pool dilation"
        )));
    }

    let rank_error = || Error::invalid_argument(format!("{diagnostic_prefix} input rank"));
    let input_dimensions = input.dimensions();
    let rank = input_dimensions.len();
    let (batch, channels, input_height, input_width) = match *input_dimensions {
        [batch, channels, height, width] => (batch, channels, height, width),
        [channels, height, width] => (1, channels, height, width),
        _ => return Err(rank_error()),
    };
    let output_shape =
        pool2d_output_shape(input, kernel_size, stride, padding, dilation, ceil_mode)?;
    let [.., output_height, output_width] = *output_shape.dimensions() else {
        return Err(rank_error());
    };

    Ok(Pool2dProblem {
        output_shape,
        rank,
        batch,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height: kernel_size.height,
        kernel_width: kernel_size.width,
        stride_height: stride.height,
        stride_width: stride.width,
        pad_height_start: padding.height.before,
        pad_height_end: padding.height.after,
        pad_width_start: padding.width.before,
        pad_width_end: padding.width.after,
        dilation_height: dilation.height,
        dilation_width: dilation.width,
    })
}

/// Computes a checked 2-D pooling output shape for CHW or NCHW tensors.
///
/// # Errors
///
/// Returns an error if input rank is neither three nor four, batch, channel, or spatial geometry is invalid, or output shape arithmetic overflows.
pub fn pool2d_output_shape(
    input: &Shape,
    kernel_size: Spatial2dI64,
    stride: Spatial2dI64,
    padding: Padding2d,
    dilation: Spatial2dI64,
    ceil_mode: bool,
) -> Result<Shape> {
    let input_dimensions = input.dimensions();
    let (batch, channels, input_height, input_width) = match *input_dimensions {
        [batch, channels, height, width] => (Some(batch), channels, height, width),
        [channels, height, width] => (None, channels, height, width),
        _ => {
            return Err(Error::InvalidTensorRank {
                operation: "pool2d".into(),
                name: "input".into(),
                expected_rank: 4,
                actual_rank: input_dimensions.len(),
                layout: "CHW or NCHW".into(),
                shape: input_dimensions.to_vec(),
            });
        }
    };
    validate_pool2d_shape(
        batch.unwrap_or(1),
        channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
    )?;

    let output_height = pool2d_output_dimension(
        input_height,
        kernel_size.height,
        padding.height.before,
        padding.height.after,
        stride.height,
        dilation.height,
        ceil_mode,
    )?;
    let output_width = pool2d_output_dimension(
        input_width,
        kernel_size.width,
        padding.width.before,
        padding.width.after,
        stride.width,
        dilation.width,
        ceil_mode,
    )?;
    let dimensions = batch.map_or_else(
        || vec![channels, output_height, output_width],
        |batch| vec![batch, channels, output_height, output_width],
    );
    Shape::create(dimensions)
}

/// Computes one checked 2-D pooling output dimension.
///
/// # Errors
///
/// Returns an error if input, kernel, stride, or dilation is not positive, padding is negative, or checked extent or ceiling-window arithmetic overflows.
pub fn pool2d_output_dimension(
    input: i64,
    kernel: i64,
    padding_start: i64,
    padding_end: i64,
    stride: i64,
    dilation: i64,
    ceil_mode: bool,
) -> Result<i64> {
    if input <= 0
        || kernel <= 0
        || stride <= 0
        || dilation <= 0
        || padding_start < 0
        || padding_end < 0
    {
        return Err(Error::invalid_argument("pool2d output dimension"));
    }

    let effective_kernel = checked_add_i64(
        checked_mul_i64(
            dilation,
            checked_sub_i64(kernel, 1, "pool2d kernel")?,
            "pool2d effective kernel",
        )?,
        1,
        "pool2d effective kernel",
    )?;
    let padded = checked_add_i64(
        checked_add_i64(input, padding_start, "pool2d padded input")?,
        padding_end,
        "pool2d padded input",
    )?;
    if padded < effective_kernel {
        return Ok(0);
    }

    let numerator = checked_sub_i64(padded, effective_kernel, "pool2d output numerator")?;
    let mut output = if ceil_mode {
        checked_add_i64(
            numerator,
            checked_sub_i64(stride, 1, "pool2d stride")?,
            "pool2d ceil numerator",
        )? / stride
            + 1
    } else {
        numerator / stride + 1
    };
    if ceil_mode
        && output > 0
        && checked_mul_i64(
            checked_sub_i64(output, 1, "pool2d output")?,
            stride,
            "pool2d ceil window start",
        )? >= checked_add_i64(input, padding_start, "pool2d padded input start")?
    {
        output -= 1;
    }
    Ok(output)
}

pub fn validate_pool2d_shape(
    batch: i64,
    channels: i64,
    input_height: i64,
    input_width: i64,
    kernel_size: Spatial2dI64,
    stride: Spatial2dI64,
    padding: Padding2d,
    dilation: Spatial2dI64,
) -> Result<()> {
    let Spatial2d {
        height: kernel_height,
        width: kernel_width,
    } = kernel_size;
    let Spatial2d {
        height: stride_height,
        width: stride_width,
    } = stride;
    let (pad_height_start, pad_height_end) = (padding.height.before, padding.height.after);
    let (pad_width_start, pad_width_end) = (padding.width.before, padding.width.after);
    let Spatial2d {
        height: dilation_height,
        width: dilation_width,
    } = dilation;
    validate_non_negative("pool2d batch", batch)?;
    for (name, value) in [
        ("pool2d channels", channels),
        ("pool2d input height", input_height),
        ("pool2d input width", input_width),
        ("pool2d kernel height", kernel_height),
        ("pool2d kernel width", kernel_width),
        ("pool2d stride height", stride_height),
        ("pool2d stride width", stride_width),
        ("pool2d dilation height", dilation_height),
        ("pool2d dilation width", dilation_width),
    ] {
        validate_positive(name, value)?;
    }

    for (name, value) in [
        ("pool2d padding height start", pad_height_start),
        ("pool2d padding height end", pad_height_end),
        ("pool2d padding width start", pad_width_start),
        ("pool2d padding width end", pad_width_end),
    ] {
        validate_non_negative(name, value)?;
    }

    if pad_height_start > kernel_height / 2
        || pad_height_end > kernel_height / 2
        || pad_width_start > kernel_width / 2
        || pad_width_end > kernel_width / 2
    {
        return Err(Error::invalid_argument("pool2d padding"));
    }
    Ok(())
}
