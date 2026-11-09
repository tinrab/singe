use crate::{
    backend::spec::{validate_non_negative, validate_positive},
    error::{Error, Result},
    shape::Shape,
    utility::{checked_add_i64, checked_mul_i64, checked_sub_i64},
};

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
/// Spatial tuples use height then width; padding is top, bottom, left, and right.
/// Average pooling requires unit dilation.
/// Each side's padding is at most half the corresponding undilated kernel size.
/// `diagnostic_prefix` identifies operation-specific errors.
///
/// # Errors
///
/// Returns an error if rank is not three or four, average-pooling dilation is not unit, batch or padding is negative, channels, spatial extents, kernel, stride, or dilation are nonpositive, padding is too large, output dimensions are invalid, or shape arithmetic overflows.
pub fn pool2d_problem(
    input: &Shape,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
    is_avg_pool: bool,
    diagnostic_prefix: &'static str,
) -> Result<Pool2dProblem> {
    if is_avg_pool && dilation != (1, 1) {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} avg pool dilation"
        )));
    }

    let input_dimensions = input.dimensions();
    let rank = input_dimensions.len();
    if rank != 3 && rank != 4 {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} input rank"
        )));
    }

    let (batch, channels, input_height, input_width) = if rank == 4 {
        (
            input_dimensions[0],
            input_dimensions[1],
            input_dimensions[2],
            input_dimensions[3],
        )
    } else {
        (
            1,
            input_dimensions[0],
            input_dimensions[1],
            input_dimensions[2],
        )
    };
    let output_shape =
        pool2d_output_shape(input, kernel_size, stride, padding, dilation, ceil_mode)?;
    let output_dimensions = output_shape.dimensions();
    let output_height = output_dimensions[rank - 2];
    let output_width = output_dimensions[rank - 1];
    let (kernel_height, kernel_width) = kernel_size;
    let (stride_height, stride_width) = stride;
    let (pad_height_start, pad_height_end, pad_width_start, pad_width_end) = padding;
    let (dilation_height, dilation_width) = dilation;

    Ok(Pool2dProblem {
        output_shape,
        rank,
        batch,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        pad_height_start,
        pad_height_end,
        pad_width_start,
        pad_width_end,
        dilation_height,
        dilation_width,
    })
}

/// Computes a checked 2-D pooling output shape for CHW or NCHW tensors.
///
/// # Errors
///
/// Returns an error if input rank is neither three nor four, batch, channel, or spatial geometry is invalid, or output shape arithmetic overflows.
pub fn pool2d_output_shape(
    input: &Shape,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    ceil_mode: bool,
) -> Result<Shape> {
    let input_dimensions = input.dimensions();
    let rank = input_dimensions.len();
    if rank != 3 && rank != 4 {
        return Err(Error::InvalidTensorRank {
            operation: "pool2d".into(),
            name: "input".into(),
            expected_rank: 4,
            actual_rank: rank,
            layout: "CHW or NCHW".into(),
            shape: input_dimensions.to_vec(),
        });
    }

    let (batch, channels, input_height, input_width) = if rank == 4 {
        (
            Some(input_dimensions[0]),
            input_dimensions[1],
            input_dimensions[2],
            input_dimensions[3],
        )
    } else {
        (
            None,
            input_dimensions[0],
            input_dimensions[1],
            input_dimensions[2],
        )
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
        kernel_size.0,
        padding.0,
        padding.1,
        stride.0,
        dilation.0,
        ceil_mode,
    )?;
    let output_width = pool2d_output_dimension(
        input_width,
        kernel_size.1,
        padding.2,
        padding.3,
        stride.1,
        dilation.1,
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
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
) -> Result<()> {
    let (kernel_height, kernel_width) = kernel_size;
    let (stride_height, stride_width) = stride;
    let (pad_height_start, pad_height_end, pad_width_start, pad_width_end) = padding;
    let (dilation_height, dilation_width) = dilation;
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
