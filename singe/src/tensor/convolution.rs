use crate::{
    backend::registry::{Conv3dAttributes, Operation, OperationAttributes},
    dispatch::call_one,
    error::{Error, Result},
    shape::spatial::{Padding, Padding2d, Spatial2d, Spatial2dI64, Spatial3dI64},
    tensor::Tensor,
    utility::to_i64,
};

// Transposed convolutions compose 2D convolution, padding, and slices.
impl Tensor {
    // Convolves `[N, C, D, H, W]` with `[O, C / groups, KD, KH, KW]` weights, as PyTorch's `conv3d` does.
    pub fn convolution3d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: impl Into<Spatial3dI64>,
        padding: impl Into<Spatial3dI64>,
        dilation: impl Into<Spatial3dI64>,
        groups: i64,
    ) -> Result<Self> {
        let attributes = OperationAttributes::Conv3d(Conv3dAttributes {
            has_bias: bias.is_some(),
            stride: stride.into(),
            padding: padding.into(),
            dilation: dilation.into(),
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
        stride: impl Into<Spatial2dI64>,
        padding: impl Into<Spatial2dI64>,
        output_padding: impl Into<Spatial2dI64>,
        dilation: impl Into<Spatial2dI64>,
        groups: i64,
    ) -> Result<Self> {
        let (stride, padding, output_padding, dilation) = (
            stride.into(),
            padding.into(),
            output_padding.into(),
            dilation.into(),
        );
        if self.shape().rank() != 4 {
            return Err(Error::invalid_argument(
                "transposed convolution needs [N, C, H, W] input",
            ));
        }

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
        if stride == Spatial2d::new(kernel_height, kernel_width)
            && padding == Spatial2d::square(0)
            && output_padding == Spatial2d::square(0)
            && dilation == Spatial2d::square(1)
            && groups == 1
        {
            return self.convolution_transpose2d_by_blocks(
                weight,
                bias,
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
            .with_zeros_between(2, stride.height)?
            .with_zeros_between(3, stride.width)?;
        // Padding beyond the kernel's reach crops the spread input instead.
        let mut margins = Padding2d::default();
        for (axis, kernel, pad, extra, dilation, margin) in [
            (
                2,
                kernel_height,
                padding.height,
                output_padding.height,
                dilation.height,
                &mut margins.height,
            ),
            (
                3,
                kernel_width,
                padding.width,
                output_padding.width,
                dilation.width,
                &mut margins.width,
            ),
        ] {
            let before = dilation * (kernel - 1) - pad;
            let after = before + extra;
            let crop_before = (-before).max(0);
            let crop_after = (-after).max(0);
            if crop_before > 0 || crop_after > 0 {
                let length = spread.dimension(axis)? - (crop_before + crop_after);
                spread = spread.narrow_with(
                    to_i64(axis, "transposed convolution axis")?,
                    crop_before,
                    length,
                )?;
            }

            *margin = Padding::new(before.max(0), after.max(0));
        }
        spread.convolution2d(&flipped, bias, (1, 1), margins, dilation, groups)
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
        [inputs, outputs, kernel_height, kernel_width]: [i64; 4],
    ) -> Result<Self> {
        let (batch, height, width) = (self.dimension(0)?, self.dimension(2)?, self.dimension(3)?);
        let blocks = self
            .permute(&[0, 2, 3, 1])?
            .contiguous()?
            .matmul(&weight.reshape([inputs, outputs * kernel_height * kernel_width])?)?
            .reshape_with(&[
                batch.clone(),
                height.clone(),
                width.clone(),
                outputs.into(),
                kernel_height.into(),
                kernel_width.into(),
            ])?
            .permute(&[0, 3, 1, 4, 2, 5])?
            .contiguous()?
            .reshape_with(&[
                batch,
                outputs.into(),
                height * kernel_height,
                width * kernel_width,
            ])?;
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

        let rank = self.shape().dimensions().len();
        if stride == 1 || self.shape().dimensions()[dim] == 0 {
            return Ok(self.clone());
        }

        let axis = to_i64(dim, "zero insertion dim")?;
        let mut padding = vec![Padding::ZERO; rank + 1];
        padding[dim + 1] = Padding::new(0, stride - 1);
        let mut merged = (0..rank)
            .map(|axis| self.dimension(axis))
            .collect::<Result<Vec<_>>>()?;
        let extent = merged[dim].clone();
        merged[dim] = extent.clone() * stride;
        self.unsqueeze(axis + 1)?
            .pad(&padding, 0.0)?
            .reshape_with(&merged)?
            .narrow_with(axis, 0_i64, (extent - 1) * stride + 1)
    }
}
