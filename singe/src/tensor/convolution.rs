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
        let &[batch, _, height, width] = self.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "transposed convolution needs [N, C, H, W] input",
            ));
        };
        let &[inputs, per_group, kernel_height, kernel_width] = weight.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "transposed convolution needs [C, O / groups, KH, KW] weights",
            ));
        };
        if groups <= 0 || inputs % groups != 0 {
            return Err(Error::invalid_argument(
                "invalid transposed convolution geometry",
            ));
        }

        // A stride of the kernel's size, with no padding, spreads each input point over its own block of the output: one matmul of the points by the weights, laid out block by block, does it without the zeros.
        if stride == (kernel_height, kernel_width)
            && padding == (0, 0)
            && output_padding == (0, 0)
            && dilation == (1, 1)
            && groups == 1
        {
            return self.convolution_transpose2d_by_blocks(
                weight,
                bias,
                [batch, height, width],
                [inputs, per_group, kernel_height, kernel_width],
            );
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
                i64::try_from(axis)
                    .map_err(|_| Error::dimension_out_of_bounds("transposed convolution axis"))?,
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

    // The transposed convolution whose stride is its kernel's size: `[N, C, H, W]` points by `[C, O, KH, KW]` weights into `[N, O, H · KH, W · KW]`.
    fn convolution_transpose2d_by_blocks(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        [batch, height, width]: [i64; 3],
        [inputs, outputs, kernel_height, kernel_width]: [i64; 4],
    ) -> Result<Self> {
        let blocks = self
            .permute(&[0, 2, 3, 1])?
            .contiguous()?
            .reshape([batch * height * width, inputs])?
            .matmul(&weight.reshape([inputs, outputs * kernel_height * kernel_width])?)?
            .reshape([batch, height, width, outputs, kernel_height, kernel_width])?
            .permute(&[0, 3, 1, 4, 2, 5])?
            .contiguous()?
            .reshape([batch, outputs, height * kernel_height, width * kernel_width])?;
        match bias {
            Some(bias) => blocks.add(&bias.reshape([1, outputs, 1, 1])?),
            None => Ok(blocks),
        }
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

        let axis =
            i64::try_from(dim).map_err(|_| Error::dimension_out_of_bounds("zero insertion dim"))?;
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
