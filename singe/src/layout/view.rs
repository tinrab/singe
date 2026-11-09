use crate::{
    error::{Error, Result},
    layout::Layout,
    shape::transform::{normalize_dim_usize, permutation, transpose_order},
    utility::to_usize,
};

impl Layout {
    /// Selects `length` positions of `axis`, starting at `start` and spaced by `step`.
    ///
    /// # Errors
    ///
    /// Returns an error if `axis` is out of range, `step` is not positive, the selection leaves the axis, or offset arithmetic overflows.
    pub fn slice(&self, axis: i64, start: i64, length: i64, step: i64) -> Result<Self> {
        let axis = normalize_dim_usize(axis, self.rank())?;
        let size = self.dimensions()[axis];
        if step < 1 {
            return Err(Error::out_of_range("slice step"));
        }

        if start < 0 || length < 0 {
            return Err(Error::IndexOutOfBounds);
        }

        let end = length
            .checked_sub(1)
            .and_then(|count| count.checked_mul(step))
            .and_then(|reach| reach.checked_add(start))
            .and_then(|last| last.checked_add(1))
            .ok_or_else(|| Error::out_of_range("slice end"))?;
        if start > size || (length > 0 && end > size) {
            return Err(Error::IndexOutOfBounds);
        }

        let stride = self.strides[axis];
        let delta = to_usize(
            start
                .checked_mul(stride)
                .ok_or_else(|| Error::out_of_range("slice offset"))?,
            "slice offset",
        )?;
        let mut dimensions = self.dimensions().to_vec();
        dimensions[axis] = length;
        let mut strides = self.strides.clone();
        strides[axis] = stride
            .checked_mul(step)
            .ok_or_else(|| Error::out_of_range("slice stride"))?;
        Ok(Self::with_strides(dimensions, strides)?.with_offset(
            self.offset
                .checked_add(delta)
                .ok_or_else(|| Error::out_of_range("slice offset"))?,
        ))
    }

    /// Swaps two logical axes while preserving strides and allocation offset.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid axis.
    /// Scalars accept `0` and `-1`.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Self> {
        self.permute(&transpose_order(self.rank(), dim0, dim1)?)
    }

    /// Removes a singleton dimension, leaving other dimensions unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if `dimension` is out of range.
    /// Scalars accept `0` and `-1`.
    pub fn squeeze(&self, dimension: i64) -> Result<Self> {
        if self.rank() == 0 && matches!(dimension, 0 | -1) {
            return Ok(self.clone());
        }

        let dim = normalize_dim_usize(dimension, self.rank())?;
        if self.shape.dimensions()[dim] != 1 {
            return Ok(self.clone());
        }

        let mut dims = self.shape.dimensions().to_vec();
        let mut strides = self.strides.clone();
        dims.remove(dim);
        strides.remove(dim);
        Ok(Self::with_strides(dims, strides)?.with_offset(self.offset))
    }

    /// Inserts a singleton dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if `dimension` is out of range for the expanded rank.
    pub fn unsqueeze(&self, dimension: i64) -> Result<Self> {
        let rank = self.rank();
        let rank_plus_one = rank
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("unsqueeze rank"))?;
        let dim = normalize_dim_usize(dimension, rank_plus_one)?;
        let mut dims = self.shape.dimensions().to_vec();
        let mut strides = self.strides.clone();
        let stride = if dim == dims.len() {
            1
        } else {
            strides[dim]
                .checked_mul(dims[dim])
                .ok_or_else(|| Error::out_of_range("unsqueeze stride"))?
        };
        dims.insert(dim, 1);
        strides.insert(dim, stride);
        Ok(Self::with_strides(dims, strides)?.with_offset(self.offset))
    }

    /// Permutes dimensions and strides according to `order`.
    ///
    /// # Errors
    ///
    /// Returns an error if `order` has the wrong length, contains out-of-range dimensions, or repeats a dimension.
    pub fn permute(&self, order: &[i64]) -> Result<Self> {
        let rank = self.rank();
        let order = permutation(order, rank)?;
        let dims: Vec<i64> = order.iter().map(|&i| self.shape.dimensions()[i]).collect();
        let strides: Vec<i64> = order.iter().map(|&i| self.strides[i]).collect();
        Ok(Self::with_strides(dims, strides)?.with_offset(self.offset))
    }

    /// Swaps the two axes of a rank-2 shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape rank is not `2`.
    pub fn transpose_2d(&self) -> Result<Self> {
        if self.rank() != 2 {
            return Err(Error::InvalidTensorRank {
                operation: "transpose_2d".into(),
                name: "input".into(),
                expected_rank: 2,
                actual_rank: self.shape.dimensions().len(),
                layout: "any".into(),
                shape: self.shape.dimensions().to_vec(),
            });
        }
        self.permute(&[1, 0])
    }
}
