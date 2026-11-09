use crate::{
    backend::registry::Operation,
    device::Device,
    error::{Error, Result},
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
        operation::{
            view_flatten_output_shape, view_output_layout, view_output_shape,
            view_start_offset_delta,
        },
    },
    layout::Layout,
    shape::{Shape, normalize_dim_usize, transform::transpose_order},
    tensor::Tensor,
};

impl NodeRef<'_> {
    pub(crate) fn view_output_layout(&self, input: &Layout) -> Result<Layout> {
        self.operation().map_or_else(
            || Ok(input.clone()),
            |operation| view_output_layout(operation, &operation_attributes(self), input),
        )
    }

    /// Folds a constant tensor through this view-like operation.
    pub(crate) fn folded_view_constant(&self, input: &Tensor) -> Result<Option<Tensor>> {
        match self.operation() {
            Some(Operation::Transpose) => {
                let Some((_, dim0, dim1)) = self.transpose() else {
                    return Ok(None);
                };
                Ok(Some(input.transpose(dim0, dim1)?))
            }
            Some(Operation::Permute) => {
                let Some((_, order)) = self.permute() else {
                    return Ok(None);
                };
                Ok(Some(input.permute(&order)?))
            }
            Some(Operation::Reshape) => {
                let Some((_, shape)) = self.reshape() else {
                    return Ok(None);
                };
                Ok(Some(input.reshape(shape.dimensions().to_vec())?))
            }
            Some(Operation::Squeeze) => {
                let Some((_, dim)) = self.single_axis_shape_view() else {
                    return Ok(None);
                };
                Ok(Some(input.squeeze(dim)?))
            }
            Some(Operation::Unsqueeze) => {
                let Some((_, dim)) = self.single_axis_shape_view() else {
                    return Ok(None);
                };
                Ok(Some(input.unsqueeze(dim)?))
            }
            Some(Operation::Slice) => {
                let Some((_, dim, start, length)) = self.slice() else {
                    return Ok(None);
                };
                Ok(Some(input.narrow(dim, start, length)?))
            }
            Some(Operation::Contiguous) => Ok(Some(input.contiguous()?)),
            Some(Operation::Flatten) => {
                let Some((_, start_dim, end_dim)) = self.flatten() else {
                    return Ok(None);
                };
                let shape = view_flatten_output_shape(input.shape(), start_dim, end_dim)?;
                Ok(Some(input.reshape(shape.dimensions().to_vec())?))
            }
            Some(Operation::Expand) => {
                let Some((_, shape)) = self.expand() else {
                    return Ok(None);
                };
                Ok(Some(input.expand(&shape)?))
            }
            _ => Ok(None),
        }
    }
}

impl NodeRef<'_> {
    /// Infers this view-like node's output shape from an input shape.
    pub(crate) fn view_output_shape(&self, input_shape: &Shape) -> Result<Shape> {
        self.operation().map_or_else(
            || Ok(input_shape.clone()),
            |operation| view_output_shape(operation, &operation_attributes(self), input_shape),
        )
    }

    /// Infers this view-like node's output layout offset from an input layout.
    pub(crate) fn view_output_offset(&self, input_layout: &Layout) -> Result<usize> {
        let Some((_, dim, start)) = self.slice_window() else {
            return Ok(input_layout.storage_offset());
        };
        let axis = normalize_dim_usize(dim, input_layout.shape().rank())?;
        view_start_offset_delta(
            start,
            input_layout.strides()[axis],
            input_layout.storage_offset(),
        )
    }

    /// Infers this view-like node's output device from an input device.
    pub(crate) fn view_output_device(&self, input_device: Device) -> Device {
        self.copy_to().map_or(input_device, |(_, device)| device)
    }
}

impl NodeRef<'_> {
    /// Returns the source input for materializing copy/view operations.
    pub(crate) fn materialization_input(&self) -> Option<ValueId> {
        matches!(
            self.operation(),
            Some(Operation::Contiguous | Operation::CopyTo)
        )
        .then(|| self.inputs().first().copied())
        .flatten()
    }

    /// Returns whether this node materializes a contiguous output.
    pub(crate) fn is_contiguous_operation(&self) -> bool {
        self.operation() == Some(Operation::Contiguous)
    }
}

impl NodeRef<'_> {
    /// Returns the source input and axes for transpose view operations.
    pub(crate) fn transpose(&self) -> Option<(ValueId, i64, i64)> {
        if self.operation() != Some(Operation::Transpose) {
            return None;
        }

        let transpose = operation_attributes(self).transpose().copied()?;
        Some((
            self.inputs().first().copied()?,
            transpose.dim0,
            transpose.dim1,
        ))
    }

    /// Returns the source input and order for permute view operations.
    pub(crate) fn permute(&self) -> Option<(ValueId, Vec<i64>)> {
        if self.operation() != Some(Operation::Permute) {
            return None;
        }
        Some((
            self.inputs().first().copied()?,
            operation_attributes(self).permutation_order()?.to_vec(),
        ))
    }

    /// Returns the explicit permutation order represented by a permutation-like node.
    pub(crate) fn permutation_order(&self, input_rank: i64) -> Result<Option<Vec<i64>>> {
        Ok(match self.operation() {
            Some(Operation::Transpose) => {
                let transpose = operation_attributes(self)
                    .transpose()
                    .copied()
                    .ok_or_else(|| Error::missing_node("permutation order"))?;
                Some(transpose_order(input_rank, transpose.dim0, transpose.dim1)?)
            }
            Some(Operation::Permute) => Some(
                operation_attributes(self)
                    .permutation_order()
                    .ok_or_else(|| Error::missing_node("permutation order"))?
                    .to_vec(),
            ),
            _ => None,
        })
    }
}

impl NodeRef<'_> {
    /// Returns the source input for shape-only view operations.
    pub(crate) fn shape_only_view_input(&self) -> Option<ValueId> {
        match self.operation()? {
            Operation::Reshape | Operation::Flatten | Operation::Squeeze | Operation::Unsqueeze => {
                self.inputs().first().copied()
            }
            _ => None,
        }
    }

    /// Returns the source input and target shape for reshape view operations.
    pub(crate) fn reshape(&self) -> Option<(ValueId, Shape)> {
        if self.operation() != Some(Operation::Reshape) {
            return None;
        }
        Some((
            self.inputs().first().copied()?,
            operation_attributes(self).shape()?.clone(),
        ))
    }

    /// Returns the source input and axes for flatten view operations.
    pub(crate) fn flatten(&self) -> Option<(ValueId, i64, i64)> {
        if self.operation() != Some(Operation::Flatten) {
            return None;
        }

        let flatten = operation_attributes(self).flatten().copied()?;
        Some((
            self.inputs().first().copied()?,
            flatten.start_dim,
            flatten.end_dim,
        ))
    }

    /// Returns the source input and axis for squeeze or unsqueeze view operations.
    pub(crate) fn single_axis_shape_view(&self) -> Option<(ValueId, i64)> {
        match self.operation()? {
            Operation::Squeeze | Operation::Unsqueeze => Some((
                self.inputs().first().copied()?,
                operation_attributes(self).dimension()?,
            )),
            _ => None,
        }
    }

    /// Returns the source input for expand view operations.
    pub(crate) fn expand_input(&self) -> Option<ValueId> {
        if self.operation() != Some(Operation::Expand) {
            return None;
        }
        self.inputs().first().copied()
    }

    /// Returns the source input and target shape for expand view operations.
    pub(crate) fn expand(&self) -> Option<(ValueId, Shape)> {
        if self.operation() != Some(Operation::Expand) {
            return None;
        }
        Some((
            self.inputs().first().copied()?,
            operation_attributes(self).shape()?.clone(),
        ))
    }
}

impl NodeRef<'_> {
    /// Returns the source input, axis, and start offset for slice view operations.
    pub(crate) fn slice_window(&self) -> Option<(ValueId, i64, i64)> {
        let (input, dim, start, _) = self.slice()?;
        Some((input, dim, start))
    }

    /// Returns the source input, axis, start, and length for slice view operations.
    pub(crate) fn slice(&self) -> Option<(ValueId, i64, i64, i64)> {
        if self.operation() != Some(Operation::Slice) {
            return None;
        }

        let slice = operation_attributes(self).slice().copied()?;
        if slice.step != 1 {
            return None;
        }
        Some((
            self.inputs().first().copied()?,
            slice.dim,
            slice.start,
            slice.length,
        ))
    }
}
