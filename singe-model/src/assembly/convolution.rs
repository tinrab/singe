//! Padding that keeps a convolution's length or size, as `PyTorch`'s `padding="same"` and `kernel_size // 2` give it, and a convolution followed by a group norm.

use singe::{
    Result as SingeResult, Tensor,
    nn::{Activation, Conv1dConfig, Conv2d, Conv2dConfig, GroupNorm, Layer, ModelState},
    shape::spatial::{Padding, Padding2d, Spatial2d},
};

use crate::{assembly::PYTORCH_NORM_EPS, schema::Schema};

#[must_use]
pub const fn same_1d(kernel: i64, dilation: i64) -> Conv1dConfig {
    let padding = dilation * (kernel - 1) / 2;
    Conv1dConfig {
        stride: 1,
        padding: Padding::symmetric(padding),
        dilation,
        groups: 1,
    }
}

#[must_use]
pub const fn same_2d(kernel: i64, stride: i64) -> Conv2dConfig {
    let padding = kernel / 2;
    Conv2dConfig {
        stride: Spatial2d::square(stride),
        padding: Padding2d::symmetric(padding, padding),
        dilation: Spatial2d::square(1),
        groups: 1,
    }
}

/// A convolution padded by half its kernel, then a group norm over `groups` groups and any activation, as the DETR family's level projections and `Mask2Former`'s pixel decoder have them.
#[derive(Debug, Clone)]
pub struct ConvGroupNorm {
    convolution: Conv2d,
    norm: GroupNorm,
    activation: Option<Activation>,
}

impl ConvGroupNorm {
    /// Registers the convolution as `convolution` and the norm as `norm`.
    pub fn create(
        schema: &mut Schema<'_>,
        [input, out, kernel, stride, groups]: [i64; 5],
        bias: bool,
    ) -> SingeResult<Self> {
        Ok(Self {
            convolution: schema.conv2d(
                "convolution",
                input,
                out,
                kernel,
                same_2d(kernel, stride),
                bias,
            )?,
            norm: schema.group_norm("norm", out, groups, PYTORCH_NORM_EPS)?,
            activation: None,
        })
    }

    #[must_use]
    pub const fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }
}

impl Layer for ConvGroupNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> SingeResult<Tensor> {
        let normed = input
            .apply(&self.convolution, state)?
            .apply(&self.norm, state)?;
        match &self.activation {
            Some(activation) => normed.apply(activation, state),
            None => Ok(normed),
        }
    }
}
