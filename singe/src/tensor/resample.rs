use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    shape::Shape,
    tensor::{Tensor, TensorDataType},
};

// How `interpolate` computes a value between input points, as PyTorch's `mode` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolateMode {
    // The input point at `floor(i · in / out)`.
    Nearest,
    // The linear blend of the two nearest input points, separately along each axis.
    Bilinear,
    // The cubic convolution of the four nearest input points, separately along each axis, with `A = -0.75`, or `-0.5` when antialiased, as PyTorch computes it.
    Bicubic,
}

// How `interpolate` resamples, as PyTorch's `mode`, `align_corners`, and `antialias` choose it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Interpolation {
    pub mode: InterpolateMode,
    // Maps the corner points of the input and output onto each other, rather than the corners of their pixels.
    pub align_corners: bool,
    // Widens the filter by the downscaling factor, so a smaller output averages every input point it covers, as PIL and torchvision resize.
    pub antialias: bool,
}

impl Interpolation {
    #[must_use]
    pub const fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            align_corners: false,
            antialias: false,
        }
    }

    #[must_use]
    pub const fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }

    #[must_use]
    pub const fn with_antialias(mut self, antialias: bool) -> Self {
        self.antialias = antialias;
        self
    }
}

impl From<InterpolateMode> for Interpolation {
    fn from(mode: InterpolateMode) -> Self {
        Self::new(mode)
    }
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
    pub fn interpolate(
        &self,
        size: (i64, i64),
        interpolation: impl Into<Interpolation>,
    ) -> Result<Self> {
        let interpolation = interpolation.into();
        self.resampled(-2, size.0, interpolation)?
            .resampled(-1, size.1, interpolation)
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
            let axis = grouped.len()
                - usize::try_from(-dim)
                    .map_err(|_| Error::dimension_out_of_bounds("adaptive pooling dim"))?;
            grouped.splice(axis..=axis, [size, window]);
            let reduced = i64::try_from(axis + 1)
                .map_err(|_| Error::dimension_out_of_bounds("adaptive pooling axis"))?;
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

    fn resampled(&self, dim: i64, size: i64, interpolation: Interpolation) -> Result<Self> {
        let extent = self.extent(dim)?;
        if size <= 0 || extent <= 0 {
            return Err(Error::invalid_argument(
                "interpolation needs positive input and output extents",
            ));
        }

        let scale = extent as f64 / size as f64;
        let indices = |positions: Vec<i64>| {
            capture::literal(self.device(), |device| {
                Self::from_vec(device, Shape::create([size])?, positions)
            })
        };
        let plain = !interpolation.align_corners && !interpolation.antialias;
        match interpolation.mode {
            InterpolateMode::Nearest if !plain => Err(Error::invalid_argument(
                "nearest interpolation neither aligns corners nor antialiases",
            )),
            InterpolateMode::Nearest => {
                let positions = (0..size)
                    .map(|index| ((index as f64 * scale).floor() as i64).min(extent - 1))
                    .collect();
                self.index_select(dim, &indices(positions)?)
            }
            InterpolateMode::Bilinear if plain => {
                let (mut low, mut high, mut weights) = (Vec::new(), Vec::new(), Vec::new());
                for index in 0..size {
                    // PyTorch rounds the product before subtracting, so a fused multiply-add could pick another source point.
                    let source = ((index as f64 + 0.5) * scale - 0.5).max(0.0);
                    let first = (source.floor() as i64).min(extent - 1);
                    low.push(first);
                    high.push((first + 1).min(extent - 1));
                    weights.push(source - first as f64);
                }

                // The literal holds `F64` or `F32` values, cast outside it to a half-precision input's type, since graphs capture literals as they are created.
                let weight = if self.data_type() == DataType::F64 {
                    capture::literal(self.device(), |device| {
                        Self::from_vec(device, Shape::create([size])?, weights)
                    })?
                } else {
                    let weights = weights
                        .iter()
                        .map(|&weight| weight as f32)
                        .collect::<Vec<_>>();
                    capture::literal(self.device(), |device| {
                        Self::from_vec(device, Shape::create([size])?, weights)
                    })?
                    .cast(self.data_type())?
                };
                // One kernel gathers each output point's two source points and blends them by its weight.
                self.resample_linear(dim, &indices(low)?, &indices(high)?, &weight)
            }
            InterpolateMode::Bilinear | InterpolateMode::Bicubic => {
                self.resampled_by_matrix(dim, size, interpolation)
            }
        }
    }

    pub(crate) fn resample_linear(
        &self,
        dim: i64,
        low: &Self,
        high: &Self,
        weight: &Self,
    ) -> Result<Self> {
        call_one(
            Operation::ResampleLinear,
            OperationAttributes::Dimension(dim),
            &[self, low, high, weight],
        )
    }

    // The extent of a negative `dim` counted from the last axis.
    fn extent(&self, dim: i64) -> Result<i64> {
        let dimensions = self.shape().dimensions();
        let axis = i64::try_from(dimensions.len())
            .map_err(|_| Error::dimension_out_of_bounds("resample rank"))?
            + dim;
        usize::try_from(axis)
            .ok()
            .and_then(|axis| dimensions.get(axis))
            .copied()
            .ok_or_else(|| Error::dimension_out_of_bounds(format!("resample dim {dim}")))
    }
}

impl Tensor {
    // Resamples `dim` by a matrix whose column `i` holds output point `i`'s weights over the input points.
    fn resampled_by_matrix(
        &self,
        dim: i64,
        size: i64,
        interpolation: Interpolation,
    ) -> Result<Self> {
        let extent = self.extent(dim)?;
        let weights = resampling_weights(extent, size, interpolation)?;
        let shape = Shape::create([extent, size])?;
        // The literal holds `F64` or `F32` values, cast outside it to a half-precision input's type, as `resampled` builds its blend weights.
        let matrix = if self.data_type() == DataType::F64 {
            literal_matrix(self, shape, weights)?
        } else {
            let weights = weights
                .iter()
                .map(|&weight| weight as f32)
                .collect::<Vec<_>>();
            literal_matrix(self, shape, weights)?.cast(self.data_type())?
        };
        self.movedim(dim, -1)?.matmul(&matrix)?.movedim(-1, dim)
    }
}

// `values` as a captured literal of `shape` on `like`'s device.
fn literal_matrix<T: TensorDataType>(
    like: &Tensor,
    shape: Shape,
    values: Vec<T>,
) -> Result<Tensor> {
    capture::literal(like.device(), |device| {
        Tensor::from_vec(device, shape, values)
    })
}

// The weights of `size` output points over `extent` input points, `[extent, size]` in row-major order, as PyTorch's `interpolate` computes them for bilinear and bicubic modes.
fn resampling_weights(extent: i64, size: i64, interpolation: Interpolation) -> Result<Vec<f64>> {
    let (input, output) = (extent as f64, size as f64);
    let scale = match (interpolation.align_corners, size > 1) {
        (true, true) => (input - 1.0) / (output - 1.0),
        (true, false) => 0.0,
        (false, _) => input / output,
    };
    let cubic = interpolation.mode == InterpolateMode::Bicubic;
    let count = usize::try_from(extent * size)
        .map_err(|_| Error::dimension_out_of_bounds("resampling weight count"))?;
    let mut weights = vec![0.0; count];
    let mut add = |point: i64, source: i64, weight: f64| {
        if let Ok(at) = usize::try_from(source * size + point) {
            weights[at] += weight;
        }
    };
    for point in 0..size {
        let position = point as f64;
        if interpolation.antialias {
            // PIL's filters, widened by the scale when downscaling: a triangle, or Keys' cubic with `A = -0.5`.
            let taps = if cubic { 4.0 } else { 2.0 };
            let (support, inverse) = if scale >= 1.0 {
                (taps * 0.5 * scale, 1.0 / scale)
            } else {
                (taps * 0.5, 1.0)
            };
            let center = scale * (position + 0.5);
            let first = ((center - support + 0.5) as i64).max(0);
            let widest = (support.ceil() as i64) * 2 + 1;
            let taken = (((center + support + 0.5) as i64).min(extent) - first).clamp(0, widest);
            let filter = |offset: f64| {
                if cubic {
                    keys(offset, -0.5)
                } else {
                    (1.0 - offset.abs()).max(0.0)
                }
            };
            let taps = (0..taken)
                .map(|tap| filter(((tap + first) as f64 - center + 0.5) * inverse))
                .collect::<Vec<_>>();
            let total = taps.iter().sum::<f64>();
            for (source, weight) in (first..).zip(taps) {
                add(
                    point,
                    source,
                    if total == 0.0 { weight } else { weight / total },
                );
            }

            continue;
        }

        let source = if interpolation.align_corners {
            scale * position
        } else {
            (position + 0.5) * scale - 0.5
        };
        if cubic {
            // Unclamped, so the four points around a position before the first keep their weights, and the border repeats.
            let floor = source.floor();
            let fraction = source - floor;
            let coefficients = [
                keys(fraction + 1.0, -0.75),
                keys(fraction, -0.75),
                keys(1.0 - fraction, -0.75),
                keys(2.0 - fraction, -0.75),
            ];
            for (offset, coefficient) in (-1..).zip(coefficients) {
                add(
                    point,
                    (floor as i64 + offset).clamp(0, extent - 1),
                    coefficient,
                );
            }
        } else {
            let source = source.max(0.0);
            let low = (source.floor() as i64).min(extent - 1);
            let weight = (source - low as f64).clamp(0.0, 1.0);
            add(point, low, 1.0 - weight);
            add(point, (low + 1).min(extent - 1), weight);
        }
    }
    Ok(weights)
}

// Keys' cubic convolution kernel with parameter `a`.
fn keys(offset: f64, a: f64) -> f64 {
    let x = offset.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}
