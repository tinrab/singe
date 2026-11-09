use std::sync::{Arc, RwLock};

use crate::{
    backend::{
        registry::{
            IndexPutAttributes, Operation, OperationAttributes, SliceAttributes,
            TransposeAttributes,
        },
        spec,
    },
    dispatch::call_one,
    error::{Error, Result},
    graph::{DimensionExpr, capture},
    layout::Layout,
    shape::{Shape, normalize_dim_usize},
    tensor::{Tensor, TensorId, TensorInner},
    utility::to_i64,
};

impl Tensor {
    /// Returns a narrowed view along `dim` starting at `start` with `length` elements.
    ///
    /// This records a slice operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` or the requested range is out of bounds, or if offset arithmetic overflows.
    pub fn narrow(&self, dim: i64, start: i64, length: i64) -> Result<Self> {
        validate_narrow_range(self, dim, start, length)?;
        call_one(
            Operation::Slice,
            OperationAttributes::Slice(SliceAttributes {
                dim,
                start,
                length,
                step: 1,
            }),
            &[self],
        )
    }

    // A view of `length` elements along `dim`, `step` apart from `start`.
    pub fn slice(&self, dim: i64, start: i64, length: i64, step: i64) -> Result<Self> {
        if step <= 0 {
            return Err(Error::invalid_argument("slice step must be positive"));
        }

        validate_narrow_range(
            self,
            dim,
            start,
            (length - 1).max(0) * step + i64::from(length > 0),
        )?;
        call_one(
            Operation::Slice,
            OperationAttributes::Slice(SliceAttributes {
                dim,
                start,
                length,
                step,
            }),
            &[self],
        )
    }

    /// Returns a copy of this tensor with each slice of `source` along `dim` written at the position `indices` names.
    ///
    /// Negative indices count from the end of `dim`; when an index repeats, which slice is written is unspecified on CUDA.
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` is not `I64`, an index is out of range, or `source` has the wrong shape or data type.
    pub fn index_copy(&self, dim: i64, indices: &Self, source: &Self) -> Result<Self> {
        self.index_write(dim, indices, source, false)
    }

    /// Returns a copy of this tensor with each slice of `source` along `dim` added at the position `indices` names.
    ///
    /// Repeated indices add every slice they name.
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` is not `I64`, an index is out of range, `source` has the wrong shape or data type, or the backend does not support accumulation for the data type.
    pub fn index_add(&self, dim: i64, indices: &Self, source: &Self) -> Result<Self> {
        self.index_write(dim, indices, source, true)
    }

    /// Selects, for every position of `indices`, the element along `dim` that it names, as `torch.gather` does.
    ///
    /// `indices` has this tensor's rank, its extents select positions along `dim`, and the result has its shape.
    /// Negative indices count from the end of `dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if the ranks differ, `dim` is out of range, an index is out of range, or another extent of `indices` exceeds this tensor's.
    pub fn gather(&self, dim: i64, indices: &Self) -> Result<Self> {
        let rank = self.shape().rank();
        if indices.shape().rank() != rank {
            return Err(Error::shape_mismatch(
                "take along dimension indices",
                self.shape().dimensions(),
                indices.shape().dimensions(),
            ));
        }

        let axis = normalize_dim_usize(dim, rank)?;
        let ones = vec![1_i64; usize::try_from(rank).map_err(|_| Error::DimensionOutOfBounds)?];
        let mut positions = Vec::with_capacity(ones.len());
        for (current, &extent) in indices.shape().dimensions().iter().enumerate() {
            if current == axis {
                positions.push(indices.clone());
                continue;
            }

            let mut dimensions = ones.clone();
            dimensions[current] = extent;
            let range = capture::literal(self.device(), |device| {
                Self::arange(device, 0_i64, extent, 1)
            })?;
            positions.push(range.reshape(dimensions)?);
        }

        let mut operands = vec![self];
        operands.extend(&positions);
        call_one(
            Operation::Index,
            OperationAttributes::Dimension(0),
            &operands,
        )
    }

    fn index_write(
        &self,
        dim: i64,
        indices: &Self,
        source: &Self,
        accumulate: bool,
    ) -> Result<Self> {
        call_one(
            Operation::IndexPut,
            OperationAttributes::IndexPut(IndexPutAttributes { dim, accumulate }),
            &[self, source, indices],
        )
    }
}

pub fn validate_narrow_range(tensor: &Tensor, dim: i64, start: i64, length: i64) -> Result<usize> {
    let rank = tensor.shape().rank();
    let dim_usize = normalize_dim_usize(dim, rank)
        .map_err(|_| Error::invalid_dimension("tensor narrow dimension", dim, rank))?;
    let normalized_dim = to_i64(dim_usize, "tensor narrow dimension")?;
    if length < 0 {
        return Err(Error::invalid_argument("tensor narrow length"));
    }

    let end = start
        .checked_add(length)
        .ok_or_else(|| Error::invalid_argument("tensor narrow end"))?;
    if start < 0 || end > tensor.shape().dimensions()[dim_usize] {
        return Err(Error::out_of_range("tensor narrow range"));
    }

    debug_assert_eq!(
        usize::try_from(normalized_dim).expect("normalized dimension is a valid host axis"),
        dim_usize
    );
    Ok(dim_usize)
}

impl Tensor {
    /// Flattens the inclusive axis range while preserving logical element order.
    ///
    /// Returns a view when strides permit it, otherwise copies.
    /// Scalars accept axes `0` and `-1` and produce a one-element vector.
    /// Capture records the resulting reshape and any required materialization.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid axis range or failed materialization.
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Result<Self> {
        let problem = spec::flatten_problem(self.shape(), start_dim, end_dim, "flatten")?;
        self.reshape(problem.output_shape.dimensions())
    }

    /// Creates a view of this tensor with a different shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape is not stride-compatible with this tensor, its explicit strides disagree with the inferred view, or capture fails.
    pub fn view(&self, shape: impl Into<Shape>) -> Result<Self> {
        let inferred = self
            .layout()
            .reshape_view(&shape.into())?
            .ok_or(Error::InvalidStride)?;
        call_one(
            Operation::Reshape,
            OperationAttributes::Shape(inferred.shape().clone()),
            &[self],
        )
    }

    /// Reshapes this tensor to `dimensions` while preserving element count.
    ///
    /// A single `-1` dimension may be used for inference.
    /// This records a graph operation when called inside graph capture.
    /// Copies logical values into contiguous storage when a view cannot preserve their order.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape is invalid or the element count changes.
    pub fn reshape(&self, dimensions: impl Into<Vec<i64>>) -> Result<Self> {
        let new_shape = self.shape().reshape(dimensions)?;
        if let Some(view) = self.layout().reshape_view(&new_shape)? {
            return call_one(
                Operation::Reshape,
                OperationAttributes::Shape(view.shape().clone()),
                &[self],
            );
        }
        self.contiguous()?.reshape(new_shape.dimensions().to_vec())
    }

    // Returns one dimension as an expression: a constant eagerly, and under capture the axis of this tensor's value, which follows the run-time shape of a program with declared run-time axes.
    pub fn dimension(&self, axis: usize) -> Result<DimensionExpr> {
        if let Some(expression) = capture::dimension(self, axis)? {
            return Ok(expression);
        }
        self.shape()
            .dimensions()
            .get(axis)
            .copied()
            .map(DimensionExpr::constant)
            .ok_or(Error::DimensionOutOfBounds)
    }

    // Reshapes to `dimensions`, which may depend on [`Self::dimension`], so a captured program's reshape follows its run-time shapes.
    // Literal zero means an empty dimension, and `-1` infers one.
    pub fn reshape_with(&self, dimensions: &[DimensionExpr]) -> Result<Self> {
        let shape = capture::evaluate_dimensions(dimensions)?;
        let output = self.reshape(shape)?;
        capture::record_reshape(&output, dimensions)?;
        Ok(output)
    }

    /// Reorders dimensions and strides according to `order`.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if `order` has the wrong length, contains duplicate axes, or references an out-of-range axis.
    pub fn permute(&self, order: &[i64]) -> Result<Self> {
        self.layout().permute(order)?;
        call_one(
            Operation::Permute,
            OperationAttributes::Permute(order.to_vec()),
            &[self],
        )
    }

    /// Swaps two dimensions of this tensor.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is out of range.
    pub fn transpose(&self, dim1: i64, dim2: i64) -> Result<Self> {
        self.layout().transpose(dim1, dim2)?;
        call_one(
            Operation::Transpose,
            OperationAttributes::Transpose(TransposeAttributes {
                dim0: dim1,
                dim1: dim2,
            }),
            &[self],
        )
    }
}

impl Tensor {
    pub(crate) fn alias_unrecorded(&self) -> Self {
        self.alias_with_layout_unrecorded(self.layout().clone())
    }

    pub(crate) fn alias_with_layout_unrecorded(&self, layout: Layout) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::generate(),
                storage: Arc::clone(&self.inner.storage),
                data_type: self.inner.data_type,
                layout,
                autograd: RwLock::default(),
            }),
        }
    }

    pub(crate) fn view_unrecorded(&self, layout: Layout) -> Result<Self> {
        if layout.element_count() != self.element_count() {
            return Err(Error::length_mismatch(
                "tensor view element count",
                self.element_count(),
                layout.element_count(),
            ));
        }

        Self::validate_storage_span(&self.inner.storage, self.inner.data_type, &layout)?;
        Ok(self.alias_with_layout_unrecorded(layout))
    }

    /// Views this tensor's storage through `layout`, which may address any subset of it.
    pub(crate) fn strided_view_unrecorded(&self, layout: Layout) -> Result<Self> {
        Self::validate_storage_span(&self.inner.storage, self.inner.data_type, &layout)?;
        Ok(self.alias_with_layout_unrecorded(layout))
    }
}
