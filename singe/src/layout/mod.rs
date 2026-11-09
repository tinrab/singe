mod metrics;
mod reshape;
mod view;

use std::{
    borrow::Borrow,
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use crate::{
    error::{Error, Result},
    shape::Shape,
    tensor::Tensor,
};

/// A logical shape with element strides and an allocation-relative element offset.
///
/// Nonnegative strides, including zero-stride immutable broadcast views, are supported.
/// A nonzero offset does not affect contiguity.
// Layouts compare, hash, and order by the elements they address, so an axis of extent one does not count its stride, and a layout of no elements addresses nothing at any offset.
#[derive(Debug, Clone)]
pub struct Layout {
    shape: Shape,
    strides: Vec<i64>,
    offset: usize,
}

impl Layout {
    // The strides that decide which elements this layout addresses, with zero for each axis whose stride does not.
    #[must_use]
    pub fn addressing_strides(&self) -> Vec<i64> {
        let empty = self.element_count() == 0;
        self.shape
            .dimensions()
            .iter()
            .zip(&self.strides)
            .map(|(&dimension, &stride)| if empty || dimension == 1 { 0 } else { stride })
            .collect()
    }

    // The offset that decides which elements this layout addresses, zero for one that addresses none.
    #[must_use]
    pub fn addressing_offset(&self) -> usize {
        if self.element_count() == 0 {
            0
        } else {
            self.offset
        }
    }
}

impl PartialEq for Layout {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.addressing_offset() == other.addressing_offset()
            && self.addressing_strides() == other.addressing_strides()
    }
}

impl Eq for Layout {}

impl Hash for Layout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.addressing_strides().hash(state);
        self.addressing_offset().hash(state);
    }
}

impl PartialOrd for Layout {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Layout {
    fn cmp(&self, other: &Self) -> Ordering {
        self.shape
            .cmp(&other.shape)
            .then_with(|| self.addressing_strides().cmp(&other.addressing_strides()))
            .then_with(|| self.addressing_offset().cmp(&other.addressing_offset()))
    }
}

impl Layout {
    /// Broadcasts immutable storage by assigning zero strides to expanded axes.
    ///
    /// # Errors
    ///
    /// Returns an error if the target rank is smaller, an existing nonunit dimension differs from its target dimension, or the resulting shape or stride metadata is invalid.
    pub fn expand(&self, target: &Shape) -> Result<Self> {
        if self.dimensions().len() > target.dimensions().len() {
            return Err(Error::InvalidStride);
        }

        let leading = target.dimensions().len() - self.dimensions().len();
        let mut strides = vec![0; target.dimensions().len()];
        for (axis, (&dimension, &stride)) in self.dimensions().iter().zip(&self.strides).enumerate()
        {
            let output_dimension = target.dimensions()[leading + axis];
            if dimension == output_dimension {
                strides[leading + axis] = stride;
            } else if dimension != 1 {
                return Err(Error::broadcast_mismatch(
                    "layout expand",
                    self.dimensions().to_vec(),
                    target.dimensions().to_vec(),
                ));
            }
        }
        Ok(Self::with_strides(target.dimensions(), strides)?.with_offset(self.offset))
    }

    /// Creates contiguous layout metadata at an allocation-relative element offset.
    #[must_use]
    pub fn new(shape: Shape, offset: usize) -> Self {
        // Shape construction checks every contiguous stride multiplication.
        let mut stride = 1_i64;
        let mut strides = vec![0; shape.dimensions().len()];
        for (axis, &dimension) in shape.dimensions().iter().enumerate().rev() {
            strides[axis] = stride;
            stride *= dimension;
        }
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Creates contiguous zero-offset layout metadata for a validated shape.
    #[must_use]
    pub fn from_shape(shape: &Shape) -> Self {
        Self::new(shape.clone(), 0)
    }

    /// Creates a contiguous layout from dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is negative or shape, element-count, or contiguous-stride arithmetic cannot be represented.
    pub fn contiguous(dimensions: impl IntoIterator<Item = impl Borrow<i64>>) -> Result<Self> {
        Ok(Self::from_shape(&Shape::create(dimensions)?))
    }

    /// Creates zero-offset layout metadata with explicit nonnegative element strides.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions cannot form a valid shape, the number of strides differs from the rank, or any stride is negative.
    pub fn with_strides(
        dimensions: impl IntoIterator<Item = impl Borrow<i64>>,
        strides: impl IntoIterator<Item = impl Borrow<i64>>,
    ) -> Result<Self> {
        let shape = Shape::create(dimensions)?;
        let strides: Vec<_> = strides.into_iter().map(|x| *x.borrow()).collect();
        if strides.len() != shape.dimensions().len() {
            return Err(Error::length_mismatch(
                "layout strides",
                shape.dimensions().len(),
                strides.len(),
            ));
        }

        if strides.iter().any(|&stride| stride < 0) {
            return Err(Error::InvalidStride);
        }
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    /// Returns this layout with a different allocation-relative element offset.
    #[must_use]
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Copies the physical metadata of a tensor descriptor.
    #[must_use]
    pub fn from_tensor(tensor: &Tensor) -> Self {
        tensor.layout().clone()
    }

    /// Returns the logical shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns logical dimensions.
    #[must_use]
    pub fn dimensions(&self) -> &[i64] {
        self.shape.dimensions()
    }

    /// Returns the number of axes.
    #[must_use]
    pub fn rank(&self) -> i64 {
        self.shape.rank()
    }

    /// Returns the number of logical elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.element_count()
    }

    /// Returns element strides in logical axis order.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Returns the allocation-relative element offset.
    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.offset
    }

    /// Tests row-major traversal, ignoring singleton strides and the storage offset.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.element_count() == 0 {
            return true;
        }

        let mut expected = 1_i64;
        for (&dimension, &stride) in self.dimensions().iter().zip(&self.strides).rev() {
            if dimension != 1 {
                if stride != expected {
                    return false;
                }

                let Some(next) = expected.checked_mul(dimension) else {
                    return false;
                };
                expected = next;
            }
        }
        true
    }
}

impl From<Shape> for Layout {
    fn from(shape: Shape) -> Self {
        Self::from_shape(&shape)
    }
}

impl From<&Shape> for Layout {
    fn from(shape: &Shape) -> Self {
        Self::from_shape(shape)
    }
}

impl From<&Self> for Layout {
    fn from(layout: &Self) -> Self {
        layout.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use super::*;

    fn hashed(layout: &Layout) -> u64 {
        let mut hasher = DefaultHasher::new();
        layout.hash(&mut hasher);
        hasher.finish()
    }

    // Two layouts that address the same elements are one layout, whatever an axis of extent one or an empty layout holds as its stride.
    #[test]
    fn test_layouts_compare_by_the_elements_they_address() {
        let contiguous = Layout::contiguous([1, 64]).unwrap();
        let strided = Layout::with_strides([1, 64], vec![1, 1]).unwrap();
        assert_eq!(contiguous, strided);
        assert_eq!(hashed(&contiguous), hashed(&strided));
        assert_eq!(contiguous.cmp(&strided), Ordering::Equal);
        assert_eq!(strided.strides(), [1, 1]);
        assert_eq!(strided.addressing_strides(), [0, 1]);
        // A stride over more than one element addresses other elements.
        let transposed = Layout::with_strides([2, 64], vec![1, 2]).unwrap();
        assert_ne!(Layout::contiguous([2, 64]).unwrap(), transposed);
        // An empty layout addresses nothing, at any offset.
        let empty = Layout::with_strides([0, 4], vec![4, 1]).unwrap();
        let other = Layout::with_strides([0, 4], vec![1, 7])
            .unwrap()
            .with_offset(3);
        assert_eq!(empty, other);
        // An offset moves what a layout of elements addresses.
        assert_ne!(contiguous.clone().with_offset(1), contiguous);
    }
}
