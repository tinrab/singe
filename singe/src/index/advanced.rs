use std::iter::repeat_n;

use crate::{
    backend::{
        registry::{IndexPutAttributes, Operation, OperationAttributes, SliceAttributes},
        spec::multi_index_problem,
    },
    data_type::DataType,
    device::Device,
    dispatch::{call, call_one},
    error::{Error, Result},
    graph::capture,
    index::TensorIndex,
    shape::Shape,
    tensor::Tensor,
    utility::to_i64,
};

/// A tensor index or mask positioned at one dimension of the basic-indexed view.
struct Advanced {
    axis: i64,
    index: Tensor,
}

impl Tensor {
    /// Selects elements with integers, slices, ellipsis, new axes, integer tensors, and Boolean masks.
    ///
    /// Integers, slices, ellipsis, and new axes produce a view of this tensor.
    /// Integer tensor indices broadcast together and lower to `gather` for one tensor or `index` for several, so the result is a copy and graph capture records it.
    /// A mask reads its values to the host to find the selected positions, so it runs eagerly and graph capture rejects it.
    /// Separated tensor indices move their result dimensions to the front, as in `NumPy`.
    ///
    /// # Errors
    ///
    /// Returns an error if the index addresses more dimensions than the tensor has, uses more than one ellipsis, an integer or tensor index is out of range, a step is not positive, or a mask's shape differs from the dimensions it addresses.
    pub fn index(&self, indices: &[TensorIndex]) -> Result<Self> {
        let (view, advanced) = self.basic_index(indices)?;
        match advanced.as_slice() {
            [] => Ok(view),
            [Advanced { axis, index }] if index.data_type() != DataType::Bool => {
                view.take(*axis, &index.to_device(view.device())?)
            }
            _ => {
                let arranged = Arranged::create(view, &advanced)?;
                let mut operands = vec![&arranged.view];
                operands.extend(&arranged.indices);
                call_one(
                    Operation::Index,
                    OperationAttributes::Dimension(arranged.axis),
                    &operands,
                )
            }
        }
    }

    /// Returns a copy of this tensor with `values` written into, or with `accumulate` added to, the elements `indices` selects, as [`Tensor::index`] reads them.
    ///
    /// `values` broadcasts to the selected shape and converts to this tensor's data type.
    /// When positions repeat without `accumulate`, which value is written is unspecified on CUDA; with `accumulate`, every value adds.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is invalid as for [`Tensor::index`], `values` does not broadcast to the selected shape, or the backend does not support accumulation for the data type.
    pub fn index_put(
        &self,
        indices: &[TensorIndex],
        values: &Self,
        accumulate: bool,
    ) -> Result<Self> {
        let output = self.copy()?;
        output.write_index(indices, values, accumulate)?;
        Ok(output)
    }

    /// Writes `values` into the elements `indices` selects, as [`Tensor::index`] reads them.
    ///
    /// `values` broadcasts to the selected shape and converts to this tensor's data type.
    /// Basic indices write through a view; tensor indices go through `index_put` and write its result back.
    /// When positions repeat, which value is written is unspecified on CUDA.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is invalid as for [`Tensor::index`], `values` does not broadcast to the selected shape, or the write fails.
    pub fn index_put_(&self, indices: &[TensorIndex], values: &Self) -> Result<()> {
        self.write_index(indices, values, false)
    }

    fn write_index(&self, indices: &[TensorIndex], values: &Self, accumulate: bool) -> Result<()> {
        let (view, advanced) = self.basic_index(indices)?;
        if !advanced.is_empty() {
            return Arranged::create(view, &advanced)?.write(values, accumulate);
        }

        if !accumulate {
            return view.assign(values);
        }

        let values = converted(values, view.device(), view.data_type())?;
        view.assign(&view.add(&values)?)
    }

    /// Applies integers, slices, ellipsis, and new axes as views, returning tensor indices with their view dimensions.
    fn basic_index(&self, indices: &[TensorIndex]) -> Result<(Self, Vec<Advanced>)> {
        let rank = self.shape().rank();
        let consumed = indices
            .iter()
            .map(TensorIndex::consumed_dimensions)
            .sum::<i64>();
        let ellipses = indices
            .iter()
            .filter(|index| matches!(index, TensorIndex::Ellipsis))
            .count();
        if ellipses > 1 {
            return Err(Error::invalid_argument(
                "tensor index has more than one ellipsis",
            ));
        }

        if consumed > rank {
            return Err(Error::DimensionOutOfBounds);
        }

        let fill = usize::try_from(rank - consumed).map_err(|_| Error::DimensionOutOfBounds)?;
        let mut expanded = Vec::with_capacity(indices.len() + fill);
        for index in indices {
            if matches!(index, TensorIndex::Ellipsis) {
                expanded.extend(repeat_n(TensorIndex::full(), fill));
            } else {
                expanded.push(index.clone());
            }
        }

        if ellipses == 0 {
            expanded.extend(repeat_n(TensorIndex::full(), fill));
        }

        let mut view = self.clone();
        let mut axis = 0_i64;
        let mut advanced = Vec::new();
        for index in expanded {
            match index {
                TensorIndex::Single(position) => {
                    let size = dimension(&view, axis)?;
                    let position = if position < 0 {
                        position + size
                    } else {
                        position
                    };
                    if !(0..size).contains(&position) {
                        return Err(Error::IndexOutOfBounds);
                    }

                    view = slice(&view, axis, position, 1, 1)?.squeeze(axis)?;
                }
                TensorIndex::Slice { start, end, step } => {
                    if step < 1 {
                        return Err(Error::invalid_argument(
                            "tensor index step must be positive",
                        ));
                    }

                    let size = dimension(&view, axis)?;
                    let start = clamp_bound(start.unwrap_or(0), size);
                    let end = clamp_bound(end.unwrap_or(size), size);
                    let length = if end > start {
                        (end - start - 1) / step + 1
                    } else {
                        0
                    };
                    if (start, length, step) != (0, size, 1) {
                        view = slice(&view, axis, start, length, step)?;
                    }

                    axis += 1;
                }
                TensorIndex::NewAxis => {
                    view = view.unsqueeze(axis)?;
                    axis += 1;
                }
                TensorIndex::Indices(index) => {
                    advanced.push(Advanced { axis, index });
                    axis += 1;
                }
                TensorIndex::Mask(mask) => {
                    let count = mask.shape().rank();
                    let start = usize::try_from(axis).map_err(|_| Error::DimensionOutOfBounds)?;
                    let end =
                        usize::try_from(axis + count).map_err(|_| Error::DimensionOutOfBounds)?;
                    if view.shape().dimensions()[start..end] != *mask.shape().dimensions() {
                        return Err(Error::shape_mismatch(
                            "tensor index mask",
                            view.shape().dimensions()[start..end].to_vec(),
                            mask.shape().dimensions(),
                        ));
                    }

                    for (offset, positions) in mask_positions(&mask)?.into_iter().enumerate() {
                        let offset =
                            i64::try_from(offset).map_err(|_| Error::DimensionOutOfBounds)?;
                        advanced.push(Advanced {
                            axis: axis + offset,
                            index: positions,
                        });
                    }

                    axis += count;
                }
                TensorIndex::Ellipsis => {}
            }
        }
        Ok((view, advanced))
    }
}

/// Moves `tensor` to `device` and converts it to `data_type`, without a cast when the types already match.
fn converted(tensor: &Tensor, device: Device, data_type: DataType) -> Result<Tensor> {
    // A same-device transfer would record a copy a program cannot move to another device.
    let tensor = if tensor.device() == device {
        tensor.clone()
    } else {
        tensor.to_device(device)?
    };
    if tensor.data_type() == data_type {
        Ok(tensor)
    } else {
        tensor.cast(data_type)
    }
}

fn dimension(tensor: &Tensor, axis: i64) -> Result<i64> {
    usize::try_from(axis)
        .ok()
        .and_then(|axis| tensor.shape().dimensions().get(axis).copied())
        .ok_or(Error::DimensionOutOfBounds)
}

/// Normalizes a negative bound from the end and clamps it to `[0, size]`.
fn clamp_bound(bound: i64, size: i64) -> i64 {
    if bound < 0 {
        size.saturating_add(bound).max(0)
    } else {
        bound.min(size)
    }
}

fn slice(tensor: &Tensor, dim: i64, start: i64, length: i64, step: i64) -> Result<Tensor> {
    call_one(
        Operation::Slice,
        OperationAttributes::Slice(SliceAttributes {
            dim,
            start,
            length,
            step,
        }),
        &[tensor],
    )
}

/// Returns, for each mask dimension, an `I64` vector of the positions where the mask is true.
fn mask_positions(mask: &Tensor) -> Result<Vec<Tensor>> {
    if mask.data_type() != DataType::Bool {
        return Err(Error::unsupported_data_type(
            "tensor index mask must be bool",
        ));
    }

    let dimensions = mask.shape().dimensions().to_vec();
    let values = mask.to_vec::<bool>()?;
    let mut positions = vec![Vec::new(); dimensions.len()];
    for (linear, _) in values.iter().enumerate().filter(|(_, value)| **value) {
        let mut remainder = to_i64(linear, "mask position")?;
        for (axis, &size) in dimensions.iter().enumerate().rev() {
            positions[axis].push(remainder % size);
            remainder /= size;
        }
    }
    positions
        .into_iter()
        .map(|positions| {
            let length = to_i64(positions.len(), "mask position count")?;
            Tensor::from_vec(mask.device(), Shape::create([length])?, positions)
        })
        .collect()
}

/// A view with its tensor-indexed dimensions moved together, and its index tensors converted to `I64` on its device.
struct Arranged {
    /// The view, with the indexed dimensions consecutive from `axis`.
    view: Tensor,
    /// The first indexed dimension of `view`.
    axis: i64,
    /// One `I64` index tensor per indexed dimension.
    indices: Vec<Tensor>,
}

impl Arranged {
    /// Keeps adjacent indexed dimensions in place and moves separated ones to the front, as `NumPy` does.
    fn create(view: Tensor, advanced: &[Advanced]) -> Result<Self> {
        let indexed = advanced.iter().map(|entry| entry.axis).collect::<Vec<_>>();
        let adjacent = indexed.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let (view, axis) = if adjacent {
            (view, indexed[0])
        } else {
            let mut order = indexed.clone();
            order.extend((0..view.shape().rank()).filter(|dim| !indexed.contains(dim)));
            (view.permute(&order)?, 0)
        };
        let indices = advanced
            .iter()
            .map(|entry| {
                if !entry.index.data_type().is_int() {
                    return Err(Error::unsupported_data_type(
                        "tensor indices must be integers",
                    ));
                }
                converted(&entry.index, view.device(), DataType::I64)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            view,
            axis,
            indices,
        })
    }

    /// Writes `values` into, or with `accumulate` adds them to, the selected elements of the view.
    fn write(&self, values: &Tensor, accumulate: bool) -> Result<()> {
        let shapes = self.indices.iter().map(Tensor::shape).collect::<Vec<_>>();
        let selected = multi_index_problem(self.view.shape(), &shapes, self.axis)?.output_shape;
        let values =
            converted(values, self.view.device(), self.view.data_type())?.expand(&selected)?;
        let mut operands = vec![&self.view, &values];
        operands.extend(&self.indices);
        let attributes = OperationAttributes::IndexPut(IndexPutAttributes {
            dim: self.axis,
            accumulate,
        });
        // Eagerly, the write lands where the elements are; a captured graph writes a scatter's result back, which compiling turns into a write in place.
        if !capture::is_active() {
            return call(Operation::IndexWrite, attributes, &operands).map(drop);
        }

        let updated = call_one(Operation::IndexPut, attributes, &operands)?;
        self.view.assign(&updated)
    }
}
