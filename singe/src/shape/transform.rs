use std::iter::repeat_n;

use crate::{
    error::{Error, Result},
    shape::Shape,
    utility::{to_i64, to_usize},
};

pub fn normalize_dim(dim: i64, rank: i64) -> Result<i64> {
    if rank < 0 {
        return Err(Error::out_of_range("rank"));
    }

    let normalized = if dim < 0 {
        rank.checked_add(dim).ok_or(Error::DimensionOutOfBounds)?
    } else {
        dim
    };
    if normalized < 0 || normalized >= rank {
        Err(Error::DimensionOutOfBounds)
    } else {
        Ok(normalized)
    }
}

pub fn normalize_dim_usize(dim: i64, rank: i64) -> Result<usize> {
    usize::try_from(normalize_dim(dim, rank)?).map_err(|_| Error::DimensionOutOfBounds)
}

// The axes of a valid `order` for a tensor of `rank`, each once.
pub fn permutation(order: &[i64], rank: i64) -> Result<Vec<usize>> {
    let rank_usize = to_usize(rank, "permute rank")?;
    if order.len() != rank_usize {
        return Err(Error::length_mismatch(
            "permute order",
            rank_usize,
            order.len(),
        ));
    }

    let order = order
        .iter()
        .map(|&axis| normalize_dim_usize(axis, rank))
        .collect::<Result<Vec<_>>>()?;
    let mut seen = vec![false; rank_usize];
    for &axis in &order {
        if seen[axis] {
            return Err(Error::invalid_argument("permute order"));
        }

        seen[axis] = true;
    }
    Ok(order)
}

pub fn transpose_order(rank: i64, dim0: i64, dim1: i64) -> Result<Vec<i64>> {
    let axis0 = normalize_dim_usize(dim0, rank.max(1))?;
    let axis1 = normalize_dim_usize(dim1, rank.max(1))?;
    let mut order = (0..rank).collect::<Vec<_>>();
    if rank != 0 {
        order.swap(axis0, axis1);
    }
    Ok(order)
}

impl Shape {
    /// Left-pads dimensions with `1` until `target_len`.
    ///
    /// This is useful for right-aligned broadcasting.
    #[must_use]
    pub fn pad_left(dimensions: &[i64], target_len: usize) -> Vec<i64> {
        let mut out = vec![1_i64; target_len];
        let offset = target_len - dimensions.len();
        out[offset..].copy_from_slice(dimensions);
        out
    }

    /// Broadcasts two shapes using NumPy-style right-aligned broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error if any non-singleton dimensions are incompatible.
    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        Self::create(Self::broadcast_dimensions(
            &self.dimensions,
            &other.dimensions,
            "broadcast",
        )?)
    }

    // The extents of two right-aligned dimension lists after broadcasting, where `name` identifies the operation in the error.
    pub(crate) fn broadcast_dimensions(lhs: &[i64], rhs: &[i64], name: &str) -> Result<Vec<i64>> {
        let rank = lhs.len().max(rhs.len());
        let lhs_axes = repeat_n(1, rank - lhs.len()).chain(lhs.iter().copied());
        let rhs_axes = repeat_n(1, rank - rhs.len()).chain(rhs.iter().copied());
        lhs_axes
            .zip(rhs_axes)
            .map(|(lhs_extent, rhs_extent)| {
                if lhs_extent == rhs_extent || rhs_extent == 1 {
                    Ok(lhs_extent)
                } else if lhs_extent == 1 {
                    Ok(rhs_extent)
                } else {
                    Err(Error::broadcast_mismatch(
                        name.to_owned(),
                        Self::pad_left(lhs, rank),
                        Self::pad_left(rhs, rank),
                    ))
                }
            })
            .collect()
    }
}

impl Shape {
    /// Reshapes while preserving element count.
    ///
    /// A single `-1` dimension may be used for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid, inference is ambiguous, or the new shape does not preserve the element count.
    pub fn reshape(&self, new_dimensions: impl Into<Vec<i64>>) -> Result<Self> {
        let new_dims = new_dimensions.into();
        let mut dims = new_dims.clone();
        let element_count = to_i64(self.element_count, "element count")?;

        let mut infer_idx = None;
        for (idx, &dimension) in dims.iter().enumerate() {
            if dimension == -1 {
                if infer_idx.is_some() {
                    return Err(Error::InvalidArgument {
                        name: "reshape inferred dimension".into(),
                    });
                }

                infer_idx = Some(idx);
            } else if dimension < 0 {
                return Err(Error::out_of_range("reshape dimension"));
            }
        }

        if let Some(idx) = infer_idx {
            let mut known = 1i64;
            for (i, &dimension) in dims.iter().enumerate() {
                if i == idx {
                    continue;
                }

                known = known
                    .checked_mul(dimension)
                    .ok_or_else(|| Error::out_of_range("reshape known element count"))?;
            }

            if known == 0 || element_count % known != 0 {
                return Err(Error::shape_mismatch(
                    "reshape element count",
                    vec![element_count],
                    new_dims,
                ));
            }

            dims[idx] = element_count / known;
        } else {
            let mut new_count = 1i64;
            for &dimension in &dims {
                new_count = new_count
                    .checked_mul(dimension)
                    .ok_or_else(|| Error::out_of_range("reshape element count"))?;
            }
            if new_count != element_count {
                return Err(Error::shape_mismatch(
                    "reshape element count",
                    vec![element_count],
                    vec![new_count],
                ));
            }
        }

        if dims.is_empty() {
            return Ok(Self::scalar());
        }
        Self::create(dims)
    }
}

impl Shape {
    /// Swaps two logical axes.
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
        if self.dimensions[dim] != 1 {
            return Ok(self.clone());
        }

        let mut dims = self.dimensions.clone();
        dims.remove(dim);
        Self::create(dims)
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
        let mut dims = self.dimensions.clone();
        dims.insert(dim, 1);
        Self::create(dims)
    }

    /// Permutes logical dimensions according to `order`.
    ///
    /// # Errors
    ///
    /// Returns an error if `order` has the wrong length, contains out-of-range dimensions, or repeats a dimension.
    pub fn permute(&self, order: &[i64]) -> Result<Self> {
        let rank = self.rank();
        let order = permutation(order, rank)?;
        let dims: Vec<i64> = order.iter().map(|&i| self.dimensions[i]).collect();
        Self::create(dims)
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
                actual_rank: self.dimensions.len(),
                layout: "any".into(),
                shape: self.dimensions.clone(),
            });
        }
        self.permute(&[1, 0])
    }
}
