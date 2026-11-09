use crate::{
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
};

impl Layout {
    /// Infers strides for a reshape that preserves logical iteration order.
    ///
    /// Adjacent axes form a chunk only when their strides describe one linear traversal.
    /// This follows the chunk-based approach in `ATen`'s computeStride.
    pub(crate) fn reshape_view(&self, target: &Shape) -> Result<Option<Self>> {
        if self.shape.element_count() != target.element_count() {
            return Err(Error::length_mismatch(
                "tensor view element count",
                self.shape.element_count(),
                target.element_count(),
            ));
        }

        if self.shape.element_count() == 0 {
            return Ok(Some(if self.shape.dimensions() == target.dimensions() {
                self.clone()
            } else {
                Self::from_shape(target).with_offset(self.offset)
            }));
        }

        if self.rank() == 0 {
            return Ok(Some(Self::from_shape(target).with_offset(self.offset)));
        }

        let mut strides = vec![0; target.dimensions().len()];
        let mut remaining = target.dimensions().len();
        let mut base_stride = self.strides[self.strides.len() - 1];
        let mut source_count = 1_i64;
        let mut target_count = 1_i64;
        for axis in (0..self.shape.dimensions().len()).rev() {
            source_count = source_count
                .checked_mul(self.shape.dimensions()[axis])
                .ok_or_else(|| Error::out_of_range("reshape view chunk"))?;
            let adjacent = source_count.checked_mul(base_stride);
            let chunk_end = axis == 0
                || (self.shape.dimensions()[axis - 1] != 1
                    && Some(self.strides[axis - 1]) != adjacent);
            if !chunk_end {
                continue;
            }

            while remaining > 0
                && (target_count < source_count || target.dimensions()[remaining - 1] == 1)
            {
                remaining -= 1;
                strides[remaining] = target_count
                    .checked_mul(base_stride)
                    .ok_or_else(|| Error::out_of_range("reshape view stride"))?;
                target_count = target_count
                    .checked_mul(target.dimensions()[remaining])
                    .ok_or_else(|| Error::out_of_range("reshape view elements"))?;
            }

            if target_count != source_count {
                return Ok(None);
            }
            if axis > 0 {
                base_stride = self.strides[axis - 1];
                source_count = 1;
                target_count = 1;
            }
        }

        if remaining != 0 {
            return Ok(None);
        }
        Ok(Some(
            Self::with_strides(target.dimensions(), strides)?.with_offset(self.offset),
        ))
    }
}
