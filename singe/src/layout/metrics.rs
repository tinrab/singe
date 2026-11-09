use crate::{
    error::{Error, Result},
    layout::Layout,
    utility::to_usize,
};

impl Layout {
    /// Maps a logical row-major index through a layout validated at a tensor boundary.
    /// The caller iterates only over `0..element_count()`.
    pub(crate) fn linear_offset(&self, mut linear: usize) -> usize {
        let mut offset = self.offset;
        for (&dimension, &stride) in self.dimensions().iter().zip(&self.strides).rev() {
            let dimension =
                usize::try_from(dimension).expect("validated dimension fits the host size");
            offset += (linear % dimension)
                * usize::try_from(stride).expect("validated stride fits the host size");
            linear /= dimension;
        }
        offset
    }

    /// Returns the required allocation span, including the element offset.
    ///
    /// This is one greater than the maximum allocation-relative address reached by the layout.
    /// Empty tensors span zero elements regardless of offset.
    ///
    /// # Errors
    ///
    /// Returns an error if stride extent calculation overflows.
    pub fn storage_span_element_count(&self) -> Result<usize> {
        if self.shape.element_count() == 0 {
            return Ok(0);
        }

        let mut max_offset = 0_i64;
        for (&dimension, &stride) in self.shape.dimensions().iter().zip(&self.strides) {
            if dimension == 0 {
                return Ok(0);
            }

            let last_index = dimension
                .checked_sub(1)
                .ok_or_else(|| Error::out_of_range("shape storage span dimension"))?;
            let dimension_offset = last_index
                .checked_mul(stride)
                .ok_or_else(|| Error::out_of_range("shape storage span offset"))?;
            max_offset = max_offset
                .checked_add(dimension_offset)
                .ok_or_else(|| Error::out_of_range("shape storage span offset"))?;
        }

        let span = max_offset
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("shape storage span"))?;
        let span = to_usize(span, "layout storage span")?;
        self.offset
            .checked_add(span)
            .ok_or_else(|| Error::out_of_range("layout storage span"))
    }

    /// Returns the allocation-relative byte span of this layout, including its offset.
    ///
    /// # Errors
    ///
    /// Returns an error if stride extent or byte-size calculation overflows.
    pub fn storage_span_byte_size(&self, data_type_size: usize) -> Result<usize> {
        self.storage_span_element_count()?
            .checked_mul(data_type_size)
            .ok_or_else(|| Error::out_of_range("shape storage byte size"))
    }
}

impl Layout {
    /// Returns whether two logical positions of this layout can address the same element.
    ///
    /// The check is conservative: a layout it cannot prove free of overlap counts as overlapping.
    #[must_use]
    pub fn has_internal_overlap(&self) -> bool {
        let mut axes = self
            .dimensions()
            .iter()
            .zip(&self.strides)
            .filter(|&(&dimension, _)| dimension > 1)
            .map(|(&dimension, &stride)| (dimension, stride))
            .collect::<Vec<_>>();
        axes.sort_by_key(|&(_, stride)| stride);
        let mut extent = 0_i64;
        for (dimension, stride) in axes {
            if stride <= extent {
                return true;
            }

            let Some(next) = (dimension - 1)
                .checked_mul(stride)
                .and_then(|reach| reach.checked_add(extent))
            else {
                return true;
            };
            extent = next;
        }
        false
    }

    /// Returns whether this layout and `other` can address a common element of one allocation.
    ///
    /// The check compares the element ranges both layouts span, so it may report overlap for interleaved layouts that share no element.
    #[must_use]
    pub fn may_overlap(&self, other: &Self) -> bool {
        let (Ok(end), Ok(other_end)) = (
            self.storage_span_element_count(),
            other.storage_span_element_count(),
        ) else {
            return true;
        };
        end > self.offset
            && other_end > other.offset
            && self.offset < other_end
            && other.offset < end
    }
}
