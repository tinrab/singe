use crate::{
    backend::{
        registry::Conv3dAttributes,
        spec::{
            neural::{ensure_non_negative_dimensions, ensure_positive_dimensions},
            validate_non_negative, validate_positive, validate_rank,
        },
    },
    error::{Error, Result},
    shape::Shape,
    utility::{checked_add_i64, checked_mul_i64, checked_sub_i64, to_usize},
};

/// Validated rank-four grouped convolution dimensions and spatial parameters used during shape inference.
pub struct Conv2dParameters {
    pub(crate) batch_size: i64,
    pub(crate) input_channels: i64,
    pub(crate) output_channels: i64,
    pub(crate) input_height: i64,
    pub(crate) input_width: i64,
    pub(crate) weight_channels: i64,
    pub(crate) kernel_height: i64,
    pub(crate) kernel_width: i64,
    pub(crate) stride_height: i64,
    pub(crate) stride_width: i64,
    pub(crate) pad_height_start: i64,
    pub(crate) pad_height_end: i64,
    pub(crate) pad_width_start: i64,
    pub(crate) pad_width_end: i64,
    pub(crate) dilation_height: i64,
    pub(crate) dilation_width: i64,
    pub(crate) groups: i64,
}

impl Conv2dParameters {
    pub(crate) fn create(
        input_dimensions: &[i64],
        weight_dimensions: &[i64],
        stride: (i64, i64),
        padding: (i64, i64, i64, i64),
        dilation: (i64, i64),
        groups: i64,
        diagnostic_prefix: &'static str,
    ) -> Result<Self> {
        validate_conv2d_rank(diagnostic_prefix, "input", input_dimensions)?;
        validate_conv2d_rank(diagnostic_prefix, "weight", weight_dimensions)?;
        if groups <= 0 {
            return Err(Error::invalid_argument(format!(
                "{diagnostic_prefix} groups"
            )));
        }

        let parameters = Self {
            batch_size: input_dimensions[0],
            input_channels: input_dimensions[1],
            input_height: input_dimensions[2],
            input_width: input_dimensions[3],
            output_channels: weight_dimensions[0],
            weight_channels: weight_dimensions[1],
            kernel_height: weight_dimensions[2],
            kernel_width: weight_dimensions[3],
            stride_height: stride.0,
            stride_width: stride.1,
            pad_height_start: padding.0,
            pad_height_end: padding.1,
            pad_width_start: padding.2,
            pad_width_end: padding.3,
            dilation_height: dilation.0,
            dilation_width: dilation.1,
            groups,
        };
        parameters.validate(diagnostic_prefix)?;
        Ok(parameters)
    }

    fn validate(&self, diagnostic_prefix: &'static str) -> Result<()> {
        ensure_non_negative_dimensions(diagnostic_prefix, &[("batch size", self.batch_size)])?;
        ensure_positive_dimensions(
            diagnostic_prefix,
            &[
                ("input channels", self.input_channels),
                ("input height", self.input_height),
                ("input width", self.input_width),
                ("output channels", self.output_channels),
                ("weight channels", self.weight_channels),
                ("kernel height", self.kernel_height),
                ("kernel width", self.kernel_width),
                ("stride height", self.stride_height),
                ("stride width", self.stride_width),
                ("dilation height", self.dilation_height),
                ("dilation width", self.dilation_width),
            ],
        )?;
        ensure_non_negative_dimensions(
            diagnostic_prefix,
            &[
                ("pad height start", self.pad_height_start),
                ("pad height end", self.pad_height_end),
                ("pad width start", self.pad_width_start),
                ("pad width end", self.pad_width_end),
            ],
        )?;
        if self.input_channels % self.groups != 0 || self.output_channels % self.groups != 0 {
            return Err(Error::shape_mismatch(
                format!("{diagnostic_prefix} grouped channels"),
                vec![self.groups],
                vec![self.input_channels, self.output_channels],
            ));
        }

        if self.weight_channels != self.input_channels / self.groups {
            return Err(Error::length_mismatch(
                format!("{diagnostic_prefix} weight channels"),
                to_usize(self.input_channels / self.groups, "backend spec dimension")?,
                to_usize(self.weight_channels, "backend spec dimension")?,
            ));
        }
        Ok(())
    }
}

/// Validated output shape and dimensions for grouped two-dimensional convolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionProblem {
    /// Output dimensions in batch, channel, height, and width order.
    pub output_shape: Shape,
    /// Number of images in the batch; zero is allowed.
    pub batch_size: i64,
    /// Total number of input channels across all groups.
    pub input_channels: i64,
    /// Total number of output channels across all groups.
    pub output_channels: i64,
    /// Input height in elements.
    pub input_height: i64,
    /// Input width in elements.
    pub input_width: i64,
    /// Number of filter rows before dilation.
    pub kernel_height: i64,
    /// Number of filter columns before dilation.
    pub kernel_width: i64,
    /// Dilated filter height, `dilation_height * (kernel_height - 1) + 1`.
    pub effective_kernel_height: i64,
    /// Dilated filter width, `dilation_width * (kernel_width - 1) + 1`.
    pub effective_kernel_width: i64,
    /// Inferred output height after padding, dilation, and stride.
    pub output_height: i64,
    /// Inferred output width after padding, dilation, and stride.
    pub output_width: i64,
    /// Positive vertical step between filter positions.
    pub stride_height: i64,
    /// Positive horizontal step between filter positions.
    pub stride_width: i64,
    /// Nonnegative padding above the image.
    pub pad_height_start: i64,
    /// Nonnegative padding below the image.
    pub pad_height_end: i64,
    /// Nonnegative padding to the left of the image.
    pub pad_width_start: i64,
    /// Nonnegative padding to the right of the image.
    pub pad_width_end: i64,
    /// Positive spacing between adjacent filter rows in input coordinates.
    pub dilation_height: i64,
    /// Positive spacing between adjacent filter columns in input coordinates.
    pub dilation_width: i64,
    /// Positive number of channel groups, dividing both input and output channel counts.
    pub groups: i64,
}

/// Validates grouped two-dimensional convolution parameters and infers its output shape.
///
/// Input shape is `[batch, input_channels, height, width]`; weights are `[output_channels, input_channels / groups, kernel_height, kernel_width]`.
/// Spatial tuples use height then width.
/// Padding is top, bottom, left, and right.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if either rank is not four, channel grouping is incompatible, batch or padding is negative, any other extent, stride, dilation, or group count is nonpositive, the effective filter exceeds the padded input, checked spatial arithmetic overflows, or constructing the output shape fails.
pub fn conv2d_problem(
    input_shape: &Shape,
    weight_shape: &Shape,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
    diagnostic_prefix: &'static str,
) -> Result<ConvolutionProblem> {
    let input_dimensions = input_shape.dimensions();
    let weight_dimensions = weight_shape.dimensions();
    let parameters = Conv2dParameters::create(
        input_dimensions,
        weight_dimensions,
        stride,
        padding,
        dilation,
        groups,
        diagnostic_prefix,
    )?;
    let spatial = conv2d_spatial_extent(
        parameters.input_height,
        parameters.input_width,
        (parameters.kernel_height, parameters.kernel_width),
        (parameters.stride_height, parameters.stride_width),
        (
            parameters.pad_height_start,
            parameters.pad_height_end,
            parameters.pad_width_start,
            parameters.pad_width_end,
        ),
        (parameters.dilation_height, parameters.dilation_width),
        diagnostic_prefix,
    )?;
    let output_shape = Shape::create([
        parameters.batch_size,
        parameters.output_channels,
        spatial.output_height,
        spatial.output_width,
    ])?;
    Ok(ConvolutionProblem {
        output_shape,
        batch_size: parameters.batch_size,
        input_channels: parameters.input_channels,
        output_channels: parameters.output_channels,
        input_height: parameters.input_height,
        input_width: parameters.input_width,
        kernel_height: parameters.kernel_height,
        kernel_width: parameters.kernel_width,
        effective_kernel_height: spatial.effective_kernel_height,
        effective_kernel_width: spatial.effective_kernel_width,
        output_height: spatial.output_height,
        output_width: spatial.output_width,
        stride_height: parameters.stride_height,
        stride_width: parameters.stride_width,
        pad_height_start: parameters.pad_height_start,
        pad_height_end: parameters.pad_height_end,
        pad_width_start: parameters.pad_width_start,
        pad_width_end: parameters.pad_width_end,
        dilation_height: parameters.dilation_height,
        dilation_width: parameters.dilation_width,
        groups: parameters.groups,
    })
}

/// Effective filter dimensions and output spatial dimensions computed from convolution parameters.
pub struct Conv2dSpatialExtent {
    pub(crate) effective_kernel_height: i64,
    pub(crate) effective_kernel_width: i64,
    pub(crate) output_height: i64,
    pub(crate) output_width: i64,
}

/// Computes effective filter sizes and floor-rounded output spatial dimensions.
///
/// For each axis, the effective kernel is `dilation * (kernel - 1) + 1` and the output is `(input + pad_before + pad_after - effective_kernel) / stride + 1`.
/// Padding is top, bottom, left, and right.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// This helper assumes validated spatial parameters, including positive strides.
/// [`conv2d_problem`] validates complete convolution parameters before calling it.
///
/// # Errors
///
/// Returns an error if checked spatial arithmetic overflows `i64` or an effective filter is larger than the padded input.
///
/// # Panics
///
/// Panics if a zero stride reaches division.
/// Adding one to an output extent can also panic on integer overflow when overflow checks are enabled.
pub fn conv2d_spatial_extent(
    input_height: i64,
    input_width: i64,
    kernel_size: (i64, i64),
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    diagnostic_prefix: &'static str,
) -> Result<Conv2dSpatialExtent> {
    let (kernel_height, kernel_width) = kernel_size;
    let (stride_height, stride_width) = stride;
    let (pad_height_start, pad_height_end, pad_width_start, pad_width_end) = padding;
    let (dilation_height, dilation_width) = dilation;
    let effective_kernel_height = effective_kernel_dimension(
        kernel_height,
        dilation_height,
        &format!("{diagnostic_prefix} effective kernel height"),
        &format!("{diagnostic_prefix} kernel height minus one"),
    )?;
    let effective_kernel_width = effective_kernel_dimension(
        kernel_width,
        dilation_width,
        &format!("{diagnostic_prefix} effective kernel width"),
        &format!("{diagnostic_prefix} kernel width minus one"),
    )?;
    let height_numerator = output_dimension_numerator(
        input_height,
        pad_height_start,
        pad_height_end,
        effective_kernel_height,
        &format!("{diagnostic_prefix} output height numerator"),
        &format!("{diagnostic_prefix} padded input height"),
    )?;
    let width_numerator = output_dimension_numerator(
        input_width,
        pad_width_start,
        pad_width_end,
        effective_kernel_width,
        &format!("{diagnostic_prefix} output width numerator"),
        &format!("{diagnostic_prefix} padded input width"),
    )?;
    if height_numerator < 0 || width_numerator < 0 {
        return Err(Error::invalid_argument(format!(
            "{diagnostic_prefix} output spatial extent"
        )));
    }

    Ok(Conv2dSpatialExtent {
        effective_kernel_height,
        effective_kernel_width,
        output_height: height_numerator / stride_height + 1,
        output_width: width_numerator / stride_width + 1,
    })
}

fn effective_kernel_dimension(
    kernel: i64,
    dilation: i64,
    name: &str,
    minus_one_name: &str,
) -> Result<i64> {
    checked_add_i64(
        checked_mul_i64(dilation, checked_sub_i64(kernel, 1, minus_one_name)?, name)?,
        1,
        name,
    )
}

fn output_dimension_numerator(
    input: i64,
    padding_start: i64,
    padding_end: i64,
    effective_kernel: i64,
    numerator_name: &str,
    padded_name: &str,
) -> Result<i64> {
    checked_sub_i64(
        checked_add_i64(
            checked_add_i64(input, padding_start, padded_name)?,
            padding_end,
            padded_name,
        )?,
        effective_kernel,
        numerator_name,
    )
}

/// Checks an optional rank-one bias with one element per output channel.
///
/// `None` is accepted.
/// `diagnostic_prefix` identifies the operation in errors.
///
/// # Errors
///
/// Returns an error if the bias is not rank one, its element count differs from the output channel count, or converting a reported count fails.
pub fn validate_conv2d_bias(
    problem: &ConvolutionProblem,
    bias_shape: Option<&Shape>,
    diagnostic_prefix: &'static str,
) -> Result<()> {
    let Some(bias_shape) = bias_shape else {
        return Ok(());
    };
    if bias_shape.dimensions().len() != 1 {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} bias rank"),
            1,
            bias_shape.dimensions().len(),
        ));
    }

    let bias_count = i64::try_from(bias_shape.element_count())
        .map_err(|_| Error::out_of_range(format!("{diagnostic_prefix} bias elements")))?;
    if bias_count != problem.output_channels {
        return Err(Error::length_mismatch(
            format!("{diagnostic_prefix} bias channels"),
            to_usize(problem.output_channels, "backend spec dimension")?,
            to_usize(bias_count, "backend spec dimension")?,
        ));
    }
    Ok(())
}

/// Checks that `dimensions` describes a rank-four convolution tensor.
///
/// The expected axis notation is NCHW.
/// `diagnostic_prefix` and `name` identify the operation and tensor in errors.
/// This checks rank only.
///
/// # Errors
///
/// Returns an error if the slice has any length other than four.
pub fn validate_conv2d_rank(
    diagnostic_prefix: &'static str,
    name: &str,
    dimensions: &[i64],
) -> Result<()> {
    validate_rank(diagnostic_prefix, name, dimensions, 4, "NCHW")
}

#[cfg(any(test, feature = "onnx"))]
/// Computes a checked 2-D convolution output shape for NCHW/OIHW tensors.
///
/// # Errors
///
/// Returns an error if input or filter rank is not four, group or channel counts are invalid or incompatible, spatial geometry is invalid, or output shape arithmetic overflows.
pub fn conv2d_output_shape(
    input: &Shape,
    weight: &Shape,
    stride: (i64, i64),
    padding: (i64, i64, i64, i64),
    dilation: (i64, i64),
    groups: i64,
) -> Result<Shape> {
    ConvolutionWindow::conv2d(stride, padding, dilation, groups).output_shape(input, weight)
}

// A convolution's parameters over any number of spatial axes: each axis's stride, padding before and after, and dilation, and the group count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionWindow {
    pub stride: Vec<i64>,
    pub padding: Vec<(i64, i64)>,
    pub dilation: Vec<i64>,
    pub groups: i64,
}

impl ConvolutionWindow {
    // Padding is ordered as height start, height end, width start, and width end.
    #[cfg(any(test, feature = "onnx", feature = "cuda"))]
    #[must_use]
    pub fn conv2d(
        stride: (i64, i64),
        padding: (i64, i64, i64, i64),
        dilation: (i64, i64),
        groups: i64,
    ) -> Self {
        Self {
            stride: vec![stride.0, stride.1],
            padding: vec![(padding.0, padding.1), (padding.2, padding.3)],
            dilation: vec![dilation.0, dilation.1],
            groups,
        }
    }

    #[must_use]
    pub fn conv3d(attributes: &Conv3dAttributes) -> Self {
        let Conv3dAttributes {
            stride,
            padding,
            dilation,
            groups,
            ..
        } = *attributes;
        Self {
            stride: vec![stride.0, stride.1, stride.2],
            padding: vec![
                (padding.0, padding.0),
                (padding.1, padding.1),
                (padding.2, padding.2),
            ],
            dilation: vec![dilation.0, dilation.1, dilation.2],
            groups,
        }
    }

    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn spatial_rank(&self) -> usize {
        self.stride.len()
    }

    #[cfg(feature = "cuda")]
    // Every parameter in one list: the strides, the paddings, the dilations, and the group count.
    #[must_use]
    pub fn parameters(&self) -> Vec<i64> {
        self.stride
            .iter()
            .copied()
            .chain(self.padding.iter().flat_map(|&pair| <[i64; 2]>::from(pair)))
            .chain(self.dilation.iter().copied())
            .chain([self.groups])
            .collect()
    }

    // The output shape of the convolution of `[N, C, spatial...]` with `[O, C / groups, kernel...]` weights.
    pub fn output_shape(&self, input: &Shape, weight: &Shape) -> Result<Shape> {
        let Self {
            stride,
            padding,
            dilation,
            groups,
        } = self;
        let groups = *groups;
        let input_dimensions = input.dimensions();
        let weight_dimensions = weight.dimensions();
        let rank = stride.len() + 2;
        validate_rank(
            "convolution input",
            "input",
            input_dimensions,
            rank,
            "NC...",
        )?;
        validate_rank(
            "convolution weight",
            "weight",
            weight_dimensions,
            rank,
            "OI...",
        )?;
        if padding.len() != stride.len() || dilation.len() != stride.len() {
            return Err(Error::invalid_argument("convolution spatial parameters"));
        }

        validate_positive("convolution groups", groups)?;

        let batch = input_dimensions[0];
        let input_channels = input_dimensions[1];
        let output_channels = weight_dimensions[0];
        let weight_channels = weight_dimensions[1];
        validate_non_negative("convolution batch", batch)?;
        validate_positive("convolution input channels", input_channels)?;
        validate_positive("convolution output channels", output_channels)?;
        validate_positive("convolution weight channels", weight_channels)?;
        if input_channels % groups != 0 {
            return Err(Error::invalid_argument("convolution input channels groups"));
        }

        if output_channels % groups != 0 {
            return Err(Error::invalid_argument(
                "convolution output channels groups",
            ));
        }

        let expected_weight_channels = input_channels / groups;
        if weight_channels != expected_weight_channels {
            return Err(Error::length_mismatch(
                "convolution weight channels",
                to_usize(expected_weight_channels, "backend spec dimension")?,
                to_usize(weight_channels, "backend spec dimension")?,
            ));
        }

        let mut output = vec![batch, output_channels];
        for (axis, ((&stride, &(before, after)), &dilation)) in
            stride.iter().zip(padding).zip(dilation).enumerate()
        {
            output.push(conv2d_output_dimension(
                input_dimensions[axis + 2],
                weight_dimensions[axis + 2],
                before,
                after,
                stride,
                dilation,
            )?);
        }
        Shape::create(output)
    }
}

/// Computes one checked 2-D convolution output dimension.
///
/// # Errors
///
/// Returns an error if input, kernel, stride, or dilation is not positive, padding is negative, the effective kernel exceeds the padded input, or checked extent arithmetic overflows.
pub fn conv2d_output_dimension(
    input: i64,
    kernel: i64,
    padding_start: i64,
    padding_end: i64,
    stride: i64,
    dilation: i64,
) -> Result<i64> {
    if input <= 0
        || kernel <= 0
        || stride <= 0
        || dilation <= 0
        || padding_start < 0
        || padding_end < 0
    {
        return Err(Error::invalid_argument("conv2d output dimension"));
    }

    let effective_kernel = checked_add_i64(
        checked_mul_i64(
            dilation,
            checked_sub_i64(kernel, 1, "conv2d kernel")?,
            "conv2d effective kernel",
        )?,
        1,
        "conv2d effective kernel",
    )?;
    let padded = checked_add_i64(
        checked_add_i64(input, padding_start, "conv2d padded input")?,
        padding_end,
        "conv2d padded input",
    )?;
    let numerator = checked_sub_i64(padded, effective_kernel, "conv2d output numerator")?;
    if numerator < 0 {
        return Err(Error::invalid_argument("conv2d output dimension"));
    }
    Ok(numerator / stride + 1)
}
