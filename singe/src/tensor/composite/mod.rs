//! Tensor operations composed from registry operations, as `PyTorch` composes its composite operations, so they need no kernels of their own and capture as the operations they are made of.
//!
//! Operations whose result shape depends on values, such as `nonzero` and `masked_select`, read the values they need back to the host, waiting for the work that produces them, as `PyTorch`'s eager calls do.
//! Graph capture rejects them, since a captured graph has no values to read, and so do meta tensors.

use std::iter;

use crate::{
    backend::{
        registry::{IndexPutAttributes, Operation, OperationAttributes, StridedAttributes},
        spec,
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    layout::Layout,
    scalar::Scalar,
    shape::{Shape, normalize_dim_usize},
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

impl Tensor {
    /// Returns the view of the slice at `index` along `dim`, which removes `dim`, as `torch.select` does.
    ///
    /// Negative `dim` and `index` values count from the end.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` or `index` is out of range.
    pub fn select(&self, dim: i64, index: i64) -> Result<Self> {
        let rank = self.shape().rank();
        let dim = if dim < 0 { dim + rank } else { dim };
        let extent = usize::try_from(dim)
            .ok()
            .and_then(|dim| self.shape().dimensions().get(dim).copied())
            .unwrap_or(0);
        let index = if index < 0 { index + extent } else { index };
        self.narrow(dim, index, 1)?.squeeze(dim)
    }

    /// Returns the tensor tiled `repeats[axis]` times along each axis, as `torch.Tensor.repeat` does; extra leading repeats add leading axes.
    ///
    /// # Errors
    ///
    /// Returns an error if `repeats` has fewer entries than the tensor has axes, or a repeat is negative.
    pub fn repeat(&self, repeats: &[i64]) -> Result<Self> {
        let leading = usize::try_from(self.shape().rank())
            .ok()
            .and_then(|rank| repeats.len().checked_sub(rank))
            .ok_or_else(|| Error::invalid_argument("repeat count"))?;
        let dimensions = iter::repeat_n(1, leading)
            .chain(self.shape().dimensions().iter().copied())
            .collect::<Vec<_>>();
        // Each axis becomes a repeat axis of size one before it, which expands to the repeat count.
        let interleaved = dimensions
            .iter()
            .flat_map(|&dimension| [1, dimension])
            .collect::<Vec<_>>();
        let expanded = repeats
            .iter()
            .zip(&dimensions)
            .flat_map(|(&repeat, &dimension)| [repeat, dimension])
            .collect::<Vec<_>>();
        let tiled = repeats
            .iter()
            .zip(&dimensions)
            .map(|(&repeat, &dimension)| repeat * dimension)
            .collect::<Vec<_>>();
        self.reshape(interleaved)?
            .expand(&Shape::create(expanded)?)?
            .reshape(tiled)
    }

    /// Returns `beta × self + alpha × (mat1 @ mat2)`, as `torch.addmm` does; a zero `beta` ignores `self`, even its NaNs.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrices do not multiply or the product does not broadcast with `self`.
    pub fn addmm(
        &self,
        mat1: &Self,
        mat2: &Self,
        beta: impl Into<Scalar>,
        alpha: impl Into<Scalar>,
    ) -> Result<Self> {
        let (beta, alpha) = (beta.into(), alpha.into());
        // Half-precision operands compute in `F32` and round once, as `PyTorch`'s do, rather than after the product and again after the sum.
        if matches!(self.data_type(), DataType::F16 | DataType::Bf16) {
            let data_type = self.data_type();
            return self
                .cast(DataType::F32)?
                .addmm(
                    &mat1.cast(DataType::F32)?,
                    &mat2.cast(DataType::F32)?,
                    beta,
                    alpha,
                )?
                .cast(data_type);
        }

        let product = mat1.matmul(mat2)?;
        let product = if alpha == Scalar::Signed(1) || alpha == Scalar::Float(1.0) {
            product
        } else {
            product.mul_scalar(alpha)?
        };
        if beta == Scalar::Signed(0) || beta == Scalar::Float(0.0) {
            return Ok(product);
        }

        let input = if beta == Scalar::Signed(1) || beta == Scalar::Float(1.0) {
            self.clone()
        } else {
            self.mul_scalar(beta)?
        };
        input.add(&product)
    }

    /// Applies 1-D convolution to an NCL input with OIK weights, as the 2-D convolution of one row.
    ///
    /// `padding` is the zeros before and after the length axis.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes, parameters, data types, devices, or backend capabilities are incompatible.
    pub fn convolution1d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: i64,
        padding: (i64, i64),
        dilation: i64,
        groups: i64,
    ) -> Result<Self> {
        self.unsqueeze(2)?
            .convolution2d(
                &weight.unsqueeze(2)?,
                bias,
                (1, stride),
                (0, 0, padding.0, padding.1),
                (1, dilation),
                groups,
            )?
            .squeeze(2)
    }

    /// Returns the `I64` coordinates of the nonzero elements, one row per element in row-major order, as `torch.nonzero` does.
    ///
    /// The number of rows depends on the values: the coordinates are written into rows bounded by the element count, and the call reads the count back to the host; a captured graph returns the result narrowed to its count after each run.
    ///
    /// # Errors
    ///
    /// Returns an error if the count cannot be read, as for meta tensors.
    pub fn nonzero(&self) -> Result<Self> {
        let truth = self.cast(DataType::Bool)?;
        let (rows, count) = selection_rows(&truth)?;
        let dimensions = self.shape().dimensions().to_vec();
        let rank = self.shape().rank();
        let elements = self.shape().element_count();
        let extent = to_i64(elements, "nonzero count")?;
        let coordinates = capture::literal(self.device(), |device| {
            let mut coordinates = Vec::with_capacity(elements * dimensions.len());
            for linear in 0..extent {
                let start = coordinates.len();
                let mut remaining = linear;
                for &axis in dimensions.iter().rev() {
                    coordinates.push(remaining % axis);
                    remaining /= axis;
                }

                coordinates[start..].reverse();
            }
            Self::from_vec(device, Shape::create([extent, rank])?, coordinates)
        })?;
        let bound = capture::literal(self.device(), |device| {
            Scalar::Signed(0).full(device, DataType::I64, &Shape::create([extent + 1, rank])?)
        })?;
        // A scalar's coordinates have no columns, so there is nothing to write.
        let bound = if rank == 0 {
            bound
        } else {
            bound.index_copy(0, &rows, &coordinates)?
        }
        .narrow(0, 0, extent)?;
        capture::counted(&bound, &count)
    }

    /// Returns the elements where the Boolean `mask` is true, in row-major order, as a vector, as `torch.masked_select` does; the tensor and the mask broadcast together.
    ///
    /// The number of elements depends on the mask, which the call reads as [`Self::nonzero`] does.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` is not Boolean, the shapes do not broadcast, or the count cannot be read.
    pub fn masked_select(&self, mask: &Self) -> Result<Self> {
        if mask.data_type() != DataType::Bool {
            return Err(Error::data_type_mismatch(
                "masked select mask",
                DataType::Bool,
                mask.data_type(),
            ));
        }

        let shape = self.shape().broadcast_with(mask.shape())?;
        let extent = to_i64(shape.element_count(), "masked select count")?;
        let values = self.expand(&shape)?.reshape([extent])?;
        let (rows, count) = selection_rows(&mask.expand(&shape)?)?;
        let bound = capture::literal(self.device(), |device| {
            Scalar::Signed(0).full(device, self.data_type(), &Shape::create([extent + 1])?)
        })?
        .index_copy(0, &rows, &values)?
        .narrow(0, 0, extent)?;
        capture::counted(&bound, &count)
    }

    pub fn flip(&self, dims: &[i64]) -> Result<Self> {
        // PyTorch tensors have no negative strides, so a flip copies, as `torch.flip` does.
        let mut flipped = self.copy()?;
        for &dim in dims {
            let axis = self.axis(dim)?;
            let extent = self.shape().dimensions()[axis];
            if extent > 1 {
                let reversed = capture::literal(self.device(), |device| {
                    Self::arange(device, extent - 1, -1, -1)
                })?;
                flipped = flipped.index_select(dim, &reversed)?;
            }
        }
        Ok(flipped)
    }

    pub fn roll(&self, shifts: &[i64], dims: &[i64]) -> Result<Self> {
        if dims.is_empty() {
            let [shift] = shifts else {
                return Err(Error::invalid_argument(
                    "roll of a flattened tensor takes one shift",
                ));
            };
            return self
                .reshape([-1])?
                .roll(&[*shift], &[0])?
                .reshape(self.shape().dimensions().to_vec());
        }

        if shifts.len() != dims.len() {
            return Err(Error::invalid_argument(
                "roll takes one shift per dimension",
            ));
        }

        let mut rolled = self.copy()?;
        for (&shift, &dim) in shifts.iter().zip(dims) {
            let extent = self.shape().dimensions()[self.axis(dim)?];
            if extent == 0 {
                continue;
            }

            let shift = shift.rem_euclid(extent);
            if shift != 0 {
                rolled = Self::concat(
                    &[
                        rolled.narrow(dim, extent - shift, shift)?,
                        rolled.narrow(dim, 0, extent - shift)?,
                    ],
                    dim,
                )?;
            }
        }
        Ok(rolled)
    }

    // Pads `dim` with `before` and `after` elements reflected about its first and last, which are not repeated, as `torch.nn.functional.pad(mode="reflect")` does.
    pub fn pad_reflect(&self, dim: i64, before: i64, after: i64) -> Result<Self> {
        let axis = self.axis(dim)?;
        let extent = self.shape().dimensions().get(axis).copied().unwrap_or(1);
        if before < 0 || after < 0 || before >= extent || after >= extent {
            return Err(Error::invalid_argument(
                "reflect padding must be shorter than the axis",
            ));
        }

        let indices = (-before..extent + after)
            .map(|index| {
                if index < 0 {
                    -index
                } else if index >= extent {
                    2 * (extent - 1) - index
                } else {
                    index
                }
            })
            .collect::<Vec<_>>();
        let length = extent + before + after;
        let indices = capture::literal(self.device(), |device| {
            Self::from_vec(device, Shape::create([length])?, indices)
        })?;
        self.index_select(dim, &indices)
    }

    pub fn unfold(&self, dim: i64, size: i64, step: i64) -> Result<Self> {
        let axis = self.axis(dim)?;
        let extent = self.shape().dimensions().get(axis).copied().unwrap_or(1);
        if step <= 0 || size < 0 || size > extent {
            return Err(Error::invalid_argument("unfold window"));
        }

        let mut dimensions = self.shape().dimensions().to_vec();
        let mut strides = Layout::from_shape(self.shape()).strides().to_vec();
        let stride = strides.get(axis).copied().unwrap_or(1);
        if let (Some(dimension), Some(axis_stride)) =
            (dimensions.get_mut(axis), strides.get_mut(axis))
        {
            *dimension = (extent - size) / step + 1;
            *axis_stride = stride * step;
        }

        dimensions.push(size);
        strides.push(stride);
        self.strided_view(dimensions, strides, 0)
    }

    pub fn diagonal(&self, offset: i64, dim1: i64, dim2: i64) -> Result<Self> {
        let (first, second) = (self.axis(dim1)?, self.axis(dim2)?);
        if first == second {
            return Err(Error::invalid_argument("diagonal input rank"));
        }

        let dimensions = self.shape().dimensions();
        let contiguous = Layout::from_shape(self.shape());
        let strides = contiguous.strides();
        let length = if offset >= 0 {
            dimensions[first].min(dimensions[second] - offset)
        } else {
            (dimensions[first] + offset).min(dimensions[second])
        }
        .max(0);
        // The diagonal starts `offset` columns right of, or above, the first element, unless it is empty.
        let start = if length == 0 {
            0
        } else if offset >= 0 {
            offset * strides[second]
        } else {
            -offset * strides[first]
        };
        let (mut kept_dimensions, mut kept_strides): (Vec<_>, Vec<_>) = dimensions
            .iter()
            .zip(strides)
            .enumerate()
            .filter(|&(axis, _)| axis != first && axis != second)
            .map(|(_, (&dimension, &stride))| (dimension, stride))
            .unzip();
        kept_dimensions.push(length);
        kept_strides.push(strides[first] + strides[second]);
        let start = to_usize(start, "diagonal offset")?;
        self.strided_view(kept_dimensions, kept_strides, start)
    }

    pub fn unbind(&self, dim: i64) -> Result<Vec<Self>> {
        let extent = self.shape().dimensions()[self.axis(dim)?];
        (0..extent).map(|index| self.select(dim, index)).collect()
    }

    pub fn split(&self, size: i64, dim: i64) -> Result<Vec<Self>> {
        if size <= 0 {
            return Err(Error::invalid_argument("split size"));
        }

        let extent = self.shape().dimensions()[self.axis(dim)?];
        if extent == 0 {
            return Ok(vec![self.clone()]);
        }
        (0..extent)
            .step_by(to_usize(size, "split size")?)
            .map(|start| self.narrow(dim, start, size.min(extent - start)))
            .collect()
    }

    pub fn chunk(&self, chunks: i64, dim: i64) -> Result<Vec<Self>> {
        if chunks <= 0 {
            return Err(Error::invalid_argument("chunk count"));
        }

        let extent = self.shape().dimensions()[self.axis(dim)?];
        self.split(((extent + chunks - 1) / chunks).max(1), dim)
    }

    fn axis(&self, dim: i64) -> Result<usize> {
        normalize_dim_usize(dim, self.shape().rank())
    }

    // Strided views address the tensor's row-major elements, read from a flat view, as `as_strided` addresses a rank-one operand; a strided tensor is copied first.
    fn strided_view(&self, dimensions: Vec<i64>, strides: Vec<i64>, offset: usize) -> Result<Self> {
        let flat = self.contiguous()?.reshape([-1])?;
        call_one(
            Operation::AsStrided,
            OperationAttributes::Strided(StridedAttributes {
                dimensions,
                strides,
                offset,
            }),
            &[&flat],
        )
    }

    pub fn any(&self, dims: &[i64], keepdim: bool) -> Result<Self> {
        self.truth_reduction(dims, keepdim, true)
    }

    pub fn all(&self, dims: &[i64], keepdim: bool) -> Result<Self> {
        self.truth_reduction(dims, keepdim, false)
    }

    // A Boolean maximum is a disjunction and a minimum a conjunction, so `any` and `all` are those reductions of the truth values; over no elements they are their identities, false and true.
    fn truth_reduction(&self, dims: &[i64], keepdim: bool, any: bool) -> Result<Self> {
        let output = spec::reduction_problem(self.shape(), dims, keepdim)?.output_shape;
        let truth = self.cast(DataType::Bool)?;
        let result = if self.shape().element_count() == 0 {
            capture::literal(self.device(), |device| {
                Scalar::Bool(!any).full(device, DataType::Bool, &output)
            })?
        } else if any {
            truth.max_reduce(dims, keepdim)?
        } else {
            truth.min_reduce(dims, keepdim)?
        };
        // PyTorch keeps `uint8` for `uint8` operands, as its older masks were bytes.
        if self.data_type() == DataType::U8 {
            result.cast(DataType::U8)
        } else {
            Ok(result)
        }
    }
}

// Returns each element's row in a selection of the true elements of `truth`, flattened, and the number of rows: a true element's row is the count of true elements before it, and every false one shares the row past the last.
fn selection_rows(truth: &Tensor) -> Result<(Tensor, Tensor)> {
    let extent = to_i64(truth.shape().element_count(), "selection count")?;
    let flat = truth.reshape([extent])?;
    let ones = flat.cast(DataType::I64)?;
    let count = ones.sum(&[], false)?;
    let rows = ones.cumsum(0)?.sub_scalar(Scalar::Signed(1))?;
    let past = capture::literal(truth.device(), |device| Tensor::scalar(device, extent))?;
    Ok((Tensor::where_select(&flat, &rows, &past)?, count))
}

// Shape and creation operations composed from views, comparisons, and selections.
impl Tensor {
    // `tensors`, which share a shape, joined along a new dimension `dim`.
    pub fn stack(tensors: &[Self], dim: i64) -> Result<Self> {
        let unsqueezed = tensors
            .iter()
            .map(|tensor| tensor.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::concat(&unsqueezed, dim)
    }

    // The elements on and above diagonal `diagonal` of the last two dimensions, and zeros below.
    pub fn triu(&self, diagonal: i64) -> Result<Self> {
        self.triangle(diagonal, true)
    }

    // The elements on and below diagonal `diagonal` of the last two dimensions, and zeros above.
    pub fn tril(&self, diagonal: i64) -> Result<Self> {
        self.triangle(diagonal, false)
    }

    // An `rows` by `columns` matrix of `data_type` with ones on its main diagonal.
    pub fn eye(device: Device, rows: i64, columns: i64, data_type: DataType) -> Result<Self> {
        let offsets = Self::diagonal_offsets(device, rows, columns)?;
        let zero = capture::literal(device, |device| Self::scalar(device, 0_i64))?;
        offsets.equal(&zero)?.cast(data_type)
    }

    // `I64` indices as rows of `classes` zeros with a one at each index.
    pub fn one_hot(&self, classes: i64) -> Result<Self> {
        let classes =
            capture::literal(self.device(), |device| Self::arange(device, 0, classes, 1))?;
        self.unsqueeze(-1)?.equal(&classes)?.cast(DataType::I64)
    }

    // Each element repeated `repeats` times along `dim`, one copy after another.
    pub fn repeat_interleave(&self, repeats: i64, dim: i64) -> Result<Self> {
        let axis = self.axis(dim)?;
        let mut expanded = self.shape().dimensions().to_vec();
        expanded.insert(axis + 1, repeats);
        let mut merged = self.shape().dimensions().to_vec();
        merged[axis] *= repeats;
        self.unsqueeze(i64::try_from(axis + 1).map_err(|_| Error::DimensionOutOfBounds)?)?
            .expand(&Shape::create(expanded)?)?
            .reshape(merged)
    }

    // A view with dimension `source` moved to position `destination`.
    pub fn movedim(&self, source: i64, destination: i64) -> Result<Self> {
        let (source, destination) = (self.axis(source)?, self.axis(destination)?);
        let mut order = (0..self.shape().rank()).collect::<Vec<_>>();
        let moved = order.remove(source);
        order.insert(destination, moved);
        self.permute(&order)
    }

    // The outer product of two vectors.
    pub fn outer(&self, other: &Self) -> Result<Self> {
        self.unsqueeze(1)?.mul(&other.unsqueeze(0)?)
    }

    // A copy with `value` where `mask` is true.
    pub fn masked_fill(&self, mask: &Self, value: impl Into<Scalar>) -> Result<Self> {
        let data_type = self.data_type();
        let fill = capture::literal(self.device(), |device| {
            Self::zeroed_on(device, data_type, Shape::scalar())
        })?
        .add_scalar(value)?
        .cast(data_type)?;
        Self::where_select(mask, &fill, self)
    }

    fn triangle(&self, diagonal: i64, upper: bool) -> Result<Self> {
        let dimensions = self.shape().dimensions();
        let [.., rows, columns] = dimensions else {
            return Err(Error::invalid_argument(
                "triangular parts need at least two dimensions",
            ));
        };
        let offsets = Self::diagonal_offsets(self.device(), *rows, *columns)?;
        let diagonal = capture::literal(self.device(), |device| Self::scalar(device, diagonal))?;
        let keep = if upper {
            offsets.greater_equal(&diagonal)?
        } else {
            offsets.less_equal(&diagonal)?
        };
        let data_type = self.data_type();
        let zero = capture::literal(self.device(), |device| {
            Self::zeroed_on(device, data_type, Shape::scalar())
        })?;
        Self::where_select(&keep, self, &zero)
    }

    // A `rows` by `columns` `I64` matrix whose element `(i, j)` is `j − i`, the diagonal it lies on.
    fn diagonal_offsets(device: Device, rows: i64, columns: i64) -> Result<Self> {
        let row = capture::literal(device, |device| Self::arange(device, 0, rows, 1))?;
        let column = capture::literal(device, |device| Self::arange(device, 0, columns, 1))?;
        column.sub(&row.unsqueeze(1)?)
    }
}

// Scatters and counts over `index_put`, which writes slices at the positions of several index tensors, one per axis.
impl Tensor {
    // A copy of this tensor with `source[i][j]...` written at the position that replaces coordinate `dim` with `index[i][j]...`, as `torch.scatter` does; `index` is `I64`, no larger than `source` along any axis, and a repeated position takes an unspecified one of its values.
    pub fn scatter(&self, dim: i64, index: &Self, source: &Self) -> Result<Self> {
        self.scatter_with(dim, index, source, false)
    }

    // Like `scatter`, with every value written at a position added to it, as `torch.scatter_add` does.
    pub fn scatter_add(&self, dim: i64, index: &Self, source: &Self) -> Result<Self> {
        self.scatter_with(dim, index, source, true)
    }

    fn scatter_with(
        &self,
        dim: i64,
        index: &Self,
        source: &Self,
        accumulate: bool,
    ) -> Result<Self> {
        let axis = self.axis(dim)?;
        let rank = self.shape().dimensions().len();
        let extents = index.shape().dimensions().to_vec();
        if extents.len() != rank || source.shape().dimensions().len() != rank {
            return Err(Error::invalid_argument(
                "scatter needs an index and a source of the tensor's rank",
            ));
        }

        // The source's leading part the index covers, and every axis's coordinates at each of the index's positions.
        let mut values = source.clone();
        let mut coordinates = Vec::with_capacity(rank);
        for (position, &extent) in extents.iter().enumerate() {
            let dimension = i64::try_from(position).map_err(|_| Error::DimensionOutOfBounds)?;
            values = values.narrow(dimension, 0, extent)?;
            if position == axis {
                coordinates.push(index.clone());
                continue;
            }

            let mut shape = vec![1; rank];
            shape[position] = extent;
            let range = capture::literal(self.device(), |device| {
                Self::arange(device, 0_i64, extent, 1)
            })?;
            coordinates.push(range.reshape(shape)?.expand(index.shape())?);
        }

        let mut inputs = vec![self, &values];
        inputs.extend(coordinates.iter());
        call_one(
            Operation::IndexPut,
            OperationAttributes::IndexPut(IndexPutAttributes { dim: 0, accumulate }),
            &inputs,
        )
    }

    // How often each of `bins` values occurs in this one-dimensional `I64` tensor of values from 0 to `bins - 1`, or the sum of `weights` at each, as `torch.bincount` counts them with `minlength` and no larger value.
    pub fn bincount(&self, weights: Option<&Self>, bins: i64) -> Result<Self> {
        let length = self.shape().dimensions().first().copied().unwrap_or(0);
        if self.shape().rank() != 1 {
            return Err(Error::invalid_argument(
                "bincount needs a one-dimensional tensor",
            ));
        }

        let (data_type, counted) = match weights {
            Some(weights) => (weights.data_type(), weights.clone()),
            None => (
                DataType::I64,
                capture::literal(self.device(), |device| {
                    Scalar::Signed(1).full(device, DataType::I64, &Shape::create([length])?)
                })?,
            ),
        };
        let shape = Shape::create([bins])?;
        capture::literal(self.device(), |device| {
            Scalar::Signed(0).full(device, data_type, &shape)
        })?
        .scatter_add(0, self, &counted)
    }
}
