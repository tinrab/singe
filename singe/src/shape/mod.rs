mod metrics;
pub(crate) mod transform;

use std::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
};

pub(crate) use self::transform::normalize_dim_usize;
use crate::{
    error::{Error, Result},
    utility::{checked_element_count, to_i64},
};

/// Logical tensor dimensions, independent of strides, offset, dtype, and device.
///
/// Rank-zero `[]` contains one scalar, `[1]` is a one-element vector, and `[0]` is an empty vector.
/// Physical metadata belongs to `Layout`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape {
    dimensions: Vec<i64>,
    rank: i64,
    element_count: usize,
}

impl From<&Self> for Shape {
    fn from(shape: &Self) -> Self {
        shape.clone()
    }
}

impl Shape {
    /// Returns dimensions in logical axis order.
    #[must_use]
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    /// Returns the number of logical axes.
    #[must_use]
    pub fn rank(&self) -> i64 {
        self.rank
    }

    /// Returns the checked number of logical elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.element_count
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dimensions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }

            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl Shape {
    /// Creates a logical shape.
    /// Rank-zero `[]` contains one scalar element.
    ///
    /// Dimensions, element counts, and representable contiguous extents are checked.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is negative, contiguous stride arithmetic overflows `i64`, the rank does not fit `i64`, or the element count does not fit the host size.
    pub fn create(dimensions: impl IntoIterator<Item = impl Borrow<i64>>) -> Result<Self> {
        let dimensions: Vec<_> = dimensions.into_iter().map(|x| *x.borrow()).collect();
        if dimensions.iter().any(|&dimension| dimension < 0) {
            return Err(Error::out_of_range("shape dimension"));
        }

        let mut extent = 1_i64;
        for &dimension in dimensions.iter().rev() {
            extent = extent
                .checked_mul(dimension)
                .ok_or_else(|| Error::out_of_range("contiguous layout stride"))?;
        }

        let rank = to_i64(dimensions.len(), "rank")?;
        let element_count = checked_element_count(&dimensions, "shape element count")?;
        Ok(Self {
            dimensions,
            rank,
            element_count,
        })
    }

    /// Returns the rank-zero scalar shape, containing one element.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            dimensions: vec![],
            rank: 0,
            element_count: 1,
        }
    }
}

#[cfg(test)]
mod tests;
