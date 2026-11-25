use crate::{
    backend::registry::Conv3dAttributes,
    error::{Error, Result},
    shape::spatial::{Padding, Padding2d},
    tensor::Tensor,
};

impl Tensor {
    // Convolves `[N, C, D, H, W]` with `[O, C / groups, KD, KH, KW]` weights as a sum over the kernel's depth taps, each a 2D convolution of every `stride.depth`-th depth plane it reads, with the planes as batch entries.
    pub fn convolution3d_by_planes(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        attributes: &Conv3dAttributes,
    ) -> Result<Self> {
        let Conv3dAttributes {
            stride,
            padding,
            dilation,
            groups,
            ..
        } = *attributes;
        let (&[batch, channels, depth, height, width], &[outputs, _, taps, ..]) =
            (self.shape().dimensions(), weight.shape().dimensions())
        else {
            return Err(Error::invalid_argument(
                "3d convolution needs [N, C, D, H, W] inputs and [O, C / groups, KD, KH, KW] weights",
            ));
        };
        if weight.shape().rank() != 5 || stride.depth <= 0 || dilation.depth <= 0 {
            return Err(Error::invalid_argument("invalid 3d convolution geometry"));
        }

        let padded_depth = depth + 2 * padding.depth;
        let span = dilation.depth * (taps - 1) + 1;
        if padded_depth < span {
            return Err(Error::invalid_argument(
                "3d convolution kernel is deeper than its padded input",
            ));
        }

        let planes = (padded_depth - span) / stride.depth + 1;
        let input = if padding.depth > 0 {
            self.pad(
                &[
                    Padding::ZERO,
                    Padding::ZERO,
                    Padding::symmetric(padding.depth),
                ],
                0.0,
            )?
        } else {
            self.clone()
        };
        let mut sum: Option<Self> = None;
        for tap in 0..taps {
            let rows = input
                .slice(2, tap * dilation.depth, planes, stride.depth)?
                .movedim(2, 1)?
                .reshape([batch * planes, channels, height, width])?;
            // Each tap's kernel is its own contiguous tensor, as cuDNN reads weights.
            let kernel = weight.select(2, tap)?.contiguous()?;
            let convolved = rows.convolution2d(
                &kernel,
                None,
                stride.planar(),
                Padding2d::symmetric(padding.height, padding.width),
                dilation.planar(),
                groups,
            )?;
            sum = Some(match sum {
                Some(sum) => sum.add(&convolved)?,
                None => convolved,
            });
        }

        let sum = sum.ok_or_else(|| Error::invalid_argument("3d convolution kernel is empty"))?;
        let &[_, _, rows, columns] = sum.shape().dimensions() else {
            return Err(Error::invalid_argument("invalid 3d convolution geometry"));
        };
        let result = sum
            .reshape([batch, planes, outputs, rows, columns])?
            .movedim(1, 2)?;
        match bias {
            Some(bias) => result.add(&bias.reshape([outputs, 1, 1, 1])?),
            None => Ok(result),
        }
    }
}
