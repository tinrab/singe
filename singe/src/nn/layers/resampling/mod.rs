// Pooling and resampling over spatial axes.

use crate::{
    backend::registry::{Operation, OperationAttributes, Pool2dAttributes},
    dispatch::call_one,
    error::{Error, Result},
    nn::{Layer, ModelState},
    tensor::{InterpolateMode, Tensor},
};

/// Value aggregation used by a two-dimensional pooling layer.
#[derive(Debug, Clone, Copy)]
pub enum Pool2dKind {
    /// Selects the largest value in each pooling window.
    Max,
    /// Averages values in each pooling window.
    Average {
        /// Whether padded positions contribute to the averaging divisor.
        count_include_pad: bool,
    },
}

/// Two-dimensional pooling window and spatial options, validated when the layer is applied.
#[derive(Debug, Clone, Copy)]
pub struct Pool2dConfig {
    /// Window height and width.
    pub kernel: (i64, i64),
    /// Vertical and horizontal step between windows.
    pub stride: (i64, i64),
    /// Padding above, below, left, and right of the input.
    pub padding: (i64, i64, i64, i64),
    /// Vertical and horizontal spacing within the window; average pooling requires unit dilation.
    pub dilation: (i64, i64),
    /// Whether spatial output inference uses ceiling division before excluding windows starting beyond the input.
    pub ceil_mode: bool,
}

impl Pool2dConfig {
    /// Creates options with stride equal to `kernel`, zero padding, unit dilation, and ceiling mode disabled.
    ///
    /// This stores options without validation.
    #[must_use]
    pub fn new(kernel: (i64, i64)) -> Self {
        Self {
            kernel,
            stride: kernel,
            padding: (0, 0, 0, 0),
            dilation: (1, 1),
            ceil_mode: false,
        }
    }
}

/// A stateless maximum or average pooling layer for channel-height-width or batch-channel-height-width inputs.
#[derive(Debug, Clone, Copy)]
pub struct Pool2d {
    kind: Pool2dKind,
    config: Pool2dConfig,
}

impl Pool2d {
    /// Stores the pooling kind and spatial options without validating them.
    #[must_use]
    pub fn new(kind: Pool2dKind, config: Pool2dConfig) -> Self {
        Self { kind, config }
    }
}

impl Layer for Pool2d {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let config = self.config;
        let (operation, count_include_pad) = match self.kind {
            Pool2dKind::Max => (Operation::MaxPool2d, None),
            Pool2dKind::Average { count_include_pad } => {
                (Operation::AvgPool2d, Some(count_include_pad))
            }
        };
        call_one(
            operation,
            OperationAttributes::Pool2d(Pool2dAttributes {
                kernel_size: config.kernel,
                stride: config.stride,
                padding: config.padding,
                dilation: config.dilation,
                ceil_mode: config.ceil_mode,
                count_include_pad,
            }),
            &[input],
        )
    }
}

// Scales the last two axes, height and width, by a whole factor, as `torch.nn.Upsample(scale_factor=…)` does.
#[derive(Debug, Clone, Copy)]
pub struct Upsample {
    scale: i64,
    mode: InterpolateMode,
}

impl Upsample {
    // Repeats each value `scale` times along both axes.
    #[must_use]
    pub const fn nearest(scale: i64) -> Self {
        Self {
            scale,
            mode: InterpolateMode::Nearest,
        }
    }

    #[must_use]
    pub const fn bilinear(scale: i64) -> Self {
        Self {
            scale,
            mode: InterpolateMode::Bilinear,
        }
    }
}

impl Layer for Upsample {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let dimensions = input.shape().dimensions();
        let [.., height, width] = dimensions else {
            return Err(Error::invalid_argument(
                "upsampling requires height and width axes",
            ));
        };
        if self.scale <= 0 {
            return Err(Error::invalid_argument("upsampling scale"));
        }
        input.interpolate((height * self.scale, width * self.scale), self.mode)
    }
}
