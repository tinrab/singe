use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::capture,
    shape::Shape,
    tensor::Tensor,
};

// How `interpolate` computes a value between input points, as PyTorch's `mode` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolateMode {
    // The input point at `floor(i · in / out)`.
    Nearest,
    // The linear blend of the two nearest input points, with corners not aligned, separately along each axis.
    Bilinear,
}

// Pooling and resampling composed from 2D pooling, slices, means, index selections, and element-wise blends.
impl Tensor {
    // Pools windows along the last axis, as 2D pooling with one-row windows does.
    pub fn max_pool1d(
        &self,
        kernel_size: i64,
        stride: i64,
        padding: (i64, i64),
        dilation: i64,
        ceil_mode: bool,
    ) -> Result<Self> {
        self.unsqueeze(-2)?
            .max_pool2d(
                (1, kernel_size),
                (1, stride),
                (0, 0, padding.0, padding.1),
                (1, dilation),
                ceil_mode,
            )?
            .squeeze(-2)
    }

    pub fn avg_pool1d(
        &self,
        kernel_size: i64,
        stride: i64,
        padding: (i64, i64),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Self> {
        self.unsqueeze(-2)?
            .avg_pool2d(
                (1, kernel_size),
                (1, stride),
                (0, 0, padding.0, padding.1),
                (1, 1),
                ceil_mode,
                count_include_pad,
            )?
            .squeeze(-2)
    }

    // Averages the last axis into `size` windows, the `i`th from `floor(i · in / size)` up to `ceil((i + 1) · in / size)`, as PyTorch's adaptive pooling divides it.
    pub fn adaptive_avg_pool1d(&self, size: i64) -> Result<Self> {
        self.adaptive_mean(-1, size)
    }

    // Averages the last two axes into `size` windows; a window's mean is the mean along one axis of the means along the other, since every window of a row has the same width.
    pub fn adaptive_avg_pool2d(&self, size: (i64, i64)) -> Result<Self> {
        self.adaptive_mean(-2, size.0)?.adaptive_mean(-1, size.1)
    }

    // Resamples the last two axes to `size`.
    pub fn interpolate(&self, size: (i64, i64), mode: InterpolateMode) -> Result<Self> {
        self.resampled(-2, size.0, mode)?
            .resampled(-1, size.1, mode)
    }

    // Takes the maximum of the last axis's windows, divided as `adaptive_avg_pool1d` divides them.
    pub fn adaptive_max_pool1d(&self, size: i64) -> Result<Self> {
        self.adaptive(-1, size, Self::max_reduce)
    }

    pub fn adaptive_max_pool2d(&self, size: (i64, i64)) -> Result<Self> {
        self.adaptive(-2, size.0, Self::max_reduce)?
            .adaptive(-1, size.1, Self::max_reduce)
    }

    // Pools `[N, C, D, H, W]` windows, separably: a box's maximum, and the count of its points inside the padded input, are the products of each axis's, so the depth pools first and the height and width after.
    pub fn max_pool3d(
        &self,
        kernel_size: (i64, i64, i64),
        stride: (i64, i64, i64),
        padding: (i64, i64, i64),
        dilation: (i64, i64, i64),
        ceil_mode: bool,
    ) -> Result<Self> {
        self.pooled_depth(|depth| {
            depth.max_pool1d(
                kernel_size.0,
                stride.0,
                (padding.0, padding.0),
                dilation.0,
                ceil_mode,
            )
        })?
        .pooled_planes(|planes| {
            planes.max_pool2d(
                (kernel_size.1, kernel_size.2),
                (stride.1, stride.2),
                (padding.1, padding.1, padding.2, padding.2),
                (dilation.1, dilation.2),
                ceil_mode,
            )
        })
    }

    pub fn avg_pool3d(
        &self,
        kernel_size: (i64, i64, i64),
        stride: (i64, i64, i64),
        padding: (i64, i64, i64),
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Self> {
        self.pooled_depth(|depth| {
            depth.avg_pool1d(
                kernel_size.0,
                stride.0,
                (padding.0, padding.0),
                ceil_mode,
                count_include_pad,
            )
        })?
        .pooled_planes(|planes| {
            planes.avg_pool2d(
                (kernel_size.1, kernel_size.2),
                (stride.1, stride.2),
                (padding.1, padding.1, padding.2, padding.2),
                (1, 1),
                ceil_mode,
                count_include_pad,
            )
        })
    }

    // Applies `pool` along the depth of `[N, C, D, H, W]`, as rows of one channel.
    fn pooled_depth(&self, pool: impl Fn(&Self) -> Result<Self>) -> Result<Self> {
        let &[batch, channels, _, height, width] = self.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "3d pooling needs [N, C, D, H, W] inputs",
            ));
        };
        let rows = self.movedim(2, 4)?.reshape([
            batch * channels * height * width,
            1,
            self.extent(-3)?,
        ])?;
        let pooled = pool(&rows)?;
        let depth = pooled.extent(-1)?;
        pooled
            .reshape([batch, channels, height, width, depth])?
            .movedim(4, 2)
    }

    // Applies 2D `pool` to each depth plane of `[N, C, D, H, W]`.
    fn pooled_planes(&self, pool: impl Fn(&Self) -> Result<Self>) -> Result<Self> {
        let &[batch, channels, depth, height, width] = self.shape().dimensions() else {
            return Err(Error::invalid_argument(
                "3d pooling needs [N, C, D, H, W] inputs",
            ));
        };
        let pooled = pool(&self.reshape([batch, channels * depth, height, width])?)?;
        let (height, width) = (pooled.extent(-2)?, pooled.extent(-1)?);
        pooled.reshape([batch, channels, depth, height, width])
    }

    fn adaptive_mean(&self, dim: i64, size: i64) -> Result<Self> {
        self.adaptive(dim, size, Self::mean)
    }

    // Reduces `dim` into `size` adaptive windows with `reduce`.
    fn adaptive(
        &self,
        dim: i64,
        size: i64,
        reduce: fn(&Self, &[i64], bool) -> Result<Self>,
    ) -> Result<Self> {
        let extent = self.extent(dim)?;
        if size <= 0 || extent <= 0 {
            return Err(Error::invalid_argument(
                "adaptive pooling needs positive input and output extents",
            ));
        }

        if extent % size == 0 {
            let window = extent / size;
            let mut grouped = self.shape().dimensions().to_vec();
            let axis =
                grouped.len() - usize::try_from(-dim).map_err(|_| Error::DimensionOutOfBounds)?;
            grouped.splice(axis..=axis, [size, window]);
            let reduced = i64::try_from(axis + 1).map_err(|_| Error::DimensionOutOfBounds)?;
            return reduce(&self.reshape(grouped)?, &[reduced], false);
        }

        let windows = (0..size)
            .map(|index| {
                let start = index * extent / size;
                let end = ((index + 1) * extent + size - 1) / size;
                reduce(&self.narrow(dim, start, end - start)?, &[dim], true)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::concat(&windows, dim)
    }

    fn resampled(&self, dim: i64, size: i64, mode: InterpolateMode) -> Result<Self> {
        let extent = self.extent(dim)?;
        if size <= 0 || extent <= 0 {
            return Err(Error::invalid_argument(
                "interpolation needs positive input and output extents",
            ));
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "spatial extents are far below 2^53"
        )]
        let scale = extent as f64 / size as f64;
        let indices = |positions: Vec<i64>| {
            capture::literal(self.device(), |device| {
                Self::from_vec(device, Shape::create([size])?, positions)
            })
        };
        match mode {
            InterpolateMode::Nearest => {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    reason = "positions are small nonnegative values"
                )]
                let positions = (0..size)
                    .map(|index| ((index as f64 * scale).floor() as i64).min(extent - 1))
                    .collect();
                self.index_select(dim, &indices(positions)?)
            }
            InterpolateMode::Bilinear => {
                let (mut low, mut high, mut weights) = (Vec::new(), Vec::new(), Vec::new());
                for index in 0..size {
                    // PyTorch rounds the product before subtracting, so a fused multiply-add could pick another source point.
                    #[expect(
                        clippy::cast_precision_loss,
                        clippy::suboptimal_flops,
                        reason = "positions are small values, rounded as PyTorch rounds them"
                    )]
                    let source = ((index as f64 + 0.5) * scale - 0.5).max(0.0);
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "positions are small nonnegative values"
                    )]
                    let first = (source.floor() as i64).min(extent - 1);
                    low.push(first);
                    high.push((first + 1).min(extent - 1));
                    #[expect(clippy::cast_precision_loss, reason = "positions are small values")]
                    weights.push(source - first as f64);
                }

                // The blend weights vary along `dim` and broadcast over the other axes.
                let mut shape = vec![1; self.shape().dimensions().len()];
                let axis =
                    shape.len() - usize::try_from(-dim).map_err(|_| Error::DimensionOutOfBounds)?;
                shape[axis] = size;
                // The literal holds `F64` or `F32` values, cast outside it to a half-precision input's type, since graphs capture literals as they are created.
                let weight = if self.data_type() == DataType::F64 {
                    capture::literal(self.device(), |device| {
                        Self::from_vec(device, Shape::create([size])?, weights)
                    })?
                } else {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "blend weights lie in [0, 1]"
                    )]
                    let weights = weights
                        .iter()
                        .map(|&weight| weight as f32)
                        .collect::<Vec<_>>();
                    capture::literal(self.device(), |device| {
                        Self::from_vec(device, Shape::create([size])?, weights)
                    })?
                    .cast(self.data_type())?
                }
                .reshape(shape)?;
                let (low, high) = (
                    self.index_select(dim, &indices(low)?)?,
                    self.index_select(dim, &indices(high)?)?,
                );
                low.add(&high.sub(&low)?.mul(&weight)?)
            }
        }
    }

    // The extent of a negative `dim` counted from the last axis.
    fn extent(&self, dim: i64) -> Result<i64> {
        let dimensions = self.shape().dimensions();
        let axis = i64::try_from(dimensions.len()).map_err(|_| Error::DimensionOutOfBounds)? + dim;
        usize::try_from(axis)
            .ok()
            .and_then(|axis| dimensions.get(axis))
            .copied()
            .ok_or(Error::DimensionOutOfBounds)
    }
}
