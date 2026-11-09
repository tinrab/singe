use crate::{
    backend::registry::{Conv3dAttributes, Operation, OperationAttributes},
    dispatch::call_one,
    error::{Error, Result},
    tensor::Tensor,
};

// Transposed convolutions compose 2D convolution, padding, and slices.
impl Tensor {
    // Convolves `[N, C, D, H, W]` with `[O, C / groups, KD, KH, KW]` weights, as PyTorch's `conv3d` does.
    pub fn convolution3d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (i64, i64, i64),
        padding: (i64, i64, i64),
        dilation: (i64, i64, i64),
        groups: i64,
    ) -> Result<Self> {
        let attributes = OperationAttributes::Conv3d(Conv3dAttributes {
            has_bias: bias.is_some(),
            stride,
            padding,
            dilation,
            groups,
        });
        match bias {
            Some(bias) => call_one(Operation::Conv3d, attributes, &[self, weight, bias]),
            None => call_one(Operation::Conv3d, attributes, &[self, weight]),
        }
    }

    // The transposed convolution of `[N, C, H, W]` with `[C, O / groups, KH, KW]` weights, as PyTorch's `conv_transpose2d` defines it: a convolution, with flipped weights whose input and output channels swap within each group, of the input with `stride - 1` zeros between its points and `dilation · (kernel - 1) - padding` zeros around them, plus `output_padding` more after.
    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments are PyTorch's conv_transpose2d's"
    )]
    pub fn convolution_transpose2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: (i64, i64),
        padding: (i64, i64),
        output_padding: (i64, i64),
        dilation: (i64, i64),
        groups: i64,
    ) -> Result<Self> {
        let &[inputs, per_group, kernel_height, kernel_width] = weight.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "transposed convolution needs [C, O / groups, KH, KW] weights",
            ));
        };
        if groups <= 0 || inputs % groups != 0 || self.shape().rank() != 4 {
            return Err(Error::invalid_argument(
                "invalid transposed convolution geometry",
            ));
        }

        let flipped = weight
            .reshape([
                groups,
                inputs / groups,
                per_group,
                kernel_height,
                kernel_width,
            ])?
            .transpose(1, 2)?
            .reshape([
                groups * per_group,
                inputs / groups,
                kernel_height,
                kernel_width,
            ])?
            .flip(&[2, 3])?;
        let mut spread = self
            .with_zeros_between(2, stride.0)?
            .with_zeros_between(3, stride.1)?;
        // Padding beyond the kernel's reach crops the spread input instead.
        let mut margins = Vec::new();
        for (axis, kernel, pad, extra, dilation) in [
            (2, kernel_height, padding.0, output_padding.0, dilation.0),
            (3, kernel_width, padding.1, output_padding.1, dilation.1),
        ] {
            let before = dilation * (kernel - 1) - pad;
            let after = before + extra;
            let extent = spread.shape().dimensions()[axis];
            let crop_before = (-before).max(0);
            let crop_after = (-after).max(0);
            spread = spread.narrow(
                i64::try_from(axis).map_err(|_| Error::DimensionOutOfBounds)?,
                crop_before,
                extent - crop_before - crop_after,
            )?;
            margins.push((before.max(0), after.max(0)));
        }
        spread.convolution2d(
            &flipped,
            bias,
            (1, 1),
            (margins[0].0, margins[0].1, margins[1].0, margins[1].1),
            dilation,
            groups,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the arguments are PyTorch's conv_transpose1d's"
    )]
    pub fn convolution_transpose1d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: i64,
        padding: i64,
        output_padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Result<Self> {
        self.unsqueeze(2)?
            .convolution_transpose2d(
                &weight.unsqueeze(2)?,
                bias,
                (1, stride),
                (0, padding),
                (0, output_padding),
                (1, dilation),
                groups,
            )?
            .squeeze(2)
    }

    // This tensor with `stride - 1` zeros between consecutive points along `dim`.
    fn with_zeros_between(&self, dim: usize, stride: i64) -> Result<Self> {
        if stride <= 0 {
            return Err(Error::invalid_argument(
                "convolution stride must be positive",
            ));
        }

        let dimensions = self.shape().dimensions();
        let extent = dimensions[dim];
        if stride == 1 || extent == 0 {
            return Ok(self.clone());
        }

        let axis = i64::try_from(dim).map_err(|_| Error::DimensionOutOfBounds)?;
        let mut padding = vec![(0, 0); dimensions.len() + 1];
        padding[dim + 1] = (0, stride - 1);
        let mut merged = dimensions.to_vec();
        merged[dim] = extent * stride;
        self.unsqueeze(axis + 1)?
            .pad(&padding, 0.0)?
            .reshape(merged)?
            .narrow(axis, 0, (extent - 1) * stride + 1)
    }
}
