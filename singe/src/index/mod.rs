mod advanced;

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::{data_type::DataType, tensor::Tensor};

/// One entry of a tensor index, as accepted by [`Tensor::index`] and [`Tensor::index_put_`].
///
/// Integers, slices, ellipsis, and new axes lower to views; tensor indices and masks select elements by value.
#[derive(Debug, Clone)]
pub enum TensorIndex {
    /// One position along a dimension, removing that dimension; negative positions count from the end.
    Single(i64),
    /// Positions from `start` up to `end`, spaced by a positive `step`.
    ///
    /// Missing bounds select from the start or through the end, negative bounds count from the end, and out-of-range bounds are clamped.
    Slice {
        /// First position, or the start of the dimension.
        start: Option<i64>,
        /// Position after the last, or the end of the dimension.
        end: Option<i64>,
        /// Distance between selected positions.
        step: i64,
    },
    /// Every dimension not addressed by other entries.
    Ellipsis,
    /// A new dimension of size one.
    NewAxis,
    /// Integer positions along one dimension, which the index's shape replaces.
    Indices(Tensor),
    /// Boolean selection over as many dimensions as the mask has, which one dimension of selected elements replaces.
    Mask(Tensor),
}

impl TensorIndex {
    /// Selects every position of one dimension.
    #[must_use]
    pub const fn full() -> Self {
        Self::Slice {
            start: None,
            end: None,
            step: 1,
        }
    }

    /// Returns this slice with its positions spaced by `step`; other entries are returned unchanged.
    #[must_use]
    pub fn step_by(self, step: i64) -> Self {
        match self {
            Self::Slice { start, end, .. } => Self::Slice { start, end, step },
            other => other,
        }
    }

    /// Returns the number of tensor dimensions this entry addresses.
    pub(crate) fn consumed_dimensions(&self) -> i64 {
        match self {
            Self::Single(_) | Self::Slice { .. } | Self::Indices(_) => 1,
            Self::Mask(mask) => mask.shape().rank(),
            Self::Ellipsis | Self::NewAxis => 0,
        }
    }
}

impl From<i64> for TensorIndex {
    fn from(index: i64) -> Self {
        Self::Single(index)
    }
}

impl From<i32> for TensorIndex {
    fn from(index: i32) -> Self {
        Self::Single(index.into())
    }
}

impl From<Range<i64>> for TensorIndex {
    fn from(range: Range<i64>) -> Self {
        Self::Slice {
            start: Some(range.start),
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<Range<i32>> for TensorIndex {
    fn from(range: Range<i32>) -> Self {
        Self::from(i64::from(range.start)..i64::from(range.end))
    }
}

impl From<RangeInclusive<i64>> for TensorIndex {
    fn from(range: RangeInclusive<i64>) -> Self {
        let (start, end) = range.into_inner();
        Self::Slice {
            start: Some(start),
            end: if end == -1 { None } else { end.checked_add(1) },
            step: 1,
        }
    }
}

impl From<RangeFrom<i64>> for TensorIndex {
    fn from(range: RangeFrom<i64>) -> Self {
        Self::Slice {
            start: Some(range.start),
            end: None,
            step: 1,
        }
    }
}

impl From<RangeFrom<i32>> for TensorIndex {
    fn from(range: RangeFrom<i32>) -> Self {
        Self::from(i64::from(range.start)..)
    }
}

impl From<RangeTo<i64>> for TensorIndex {
    fn from(range: RangeTo<i64>) -> Self {
        Self::Slice {
            start: None,
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<RangeTo<i32>> for TensorIndex {
    fn from(range: RangeTo<i32>) -> Self {
        Self::from(..i64::from(range.end))
    }
}

impl From<RangeToInclusive<i64>> for TensorIndex {
    fn from(range: RangeToInclusive<i64>) -> Self {
        Self::Slice {
            start: None,
            end: if range.end == -1 {
                None
            } else {
                range.end.checked_add(1)
            },
            step: 1,
        }
    }
}

impl From<RangeFull> for TensorIndex {
    fn from(_: RangeFull) -> Self {
        Self::full()
    }
}

impl From<Tensor> for TensorIndex {
    fn from(tensor: Tensor) -> Self {
        if tensor.data_type() == DataType::Bool {
            Self::Mask(tensor)
        } else {
            Self::Indices(tensor)
        }
    }
}

impl From<&Tensor> for TensorIndex {
    fn from(tensor: &Tensor) -> Self {
        Self::from(tensor.clone())
    }
}

/// Builds an array of [`TensorIndex`] entries with Python-like syntax.
///
/// `a..b; s` is a slice with step `s`, `...` is an ellipsis, `None` is a new axis, and any other expression converts through `Into<TensorIndex>`.
///
/// ```
/// # use singe::{Device, Shape, Tensor, idx};
/// let x = Tensor::from_vec(Device::Cpu, Shape::create([2, 6])?, (0..12).map(|v| v as f32).collect())?;
/// let y = x.index(&idx![-1, 1..; 2, None])?;
/// assert_eq!(y.shape().dimensions(), [3, 1]);
/// assert_eq!(y.to_vec::<f32>()?, [7., 9., 11.]);
/// # Ok::<(), singe::Error>(())
/// ```
#[macro_export]
macro_rules! idx {
    (@collect [$($out:expr),*]) => {
        [$($out),*]
    };
    (@collect [$($out:expr),*] ... $(, $($rest:tt)*)?) => {
        $crate::idx!(@collect [$($out,)* $crate::index::TensorIndex::Ellipsis] $($($rest)*)?)
    };
    (@collect [$($out:expr),*] None $(, $($rest:tt)*)?) => {
        $crate::idx!(@collect [$($out,)* $crate::index::TensorIndex::NewAxis] $($($rest)*)?)
    };
    (@collect [$($out:expr),*] $index:expr ; $step:expr $(, $($rest:tt)*)?) => {
        $crate::idx!(@collect [$($out,)* $crate::index::TensorIndex::from($index).step_by($step)] $($($rest)*)?)
    };
    (@collect [$($out:expr),*] $index:expr $(, $($rest:tt)*)?) => {
        $crate::idx!(@collect [$($out,)* $crate::index::TensorIndex::from($index)] $($($rest)*)?)
    };
    ($($items:tt)*) => {
        $crate::idx!(@collect [] $($items)*)
    };
}

#[cfg(test)]
mod tests;
