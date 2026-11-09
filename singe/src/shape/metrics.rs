use crate::{
    error::{Error, Result},
    shape::Shape,
};

impl Shape {
    /// Returns the checked number of bytes required for this shape and element size.
    ///
    /// # Errors
    ///
    /// Returns an error if `element_count * data_type_size` overflows `usize`.
    pub fn byte_size(&self, data_type_size: usize) -> Result<usize> {
        self.element_count
            .checked_mul(data_type_size)
            .ok_or_else(|| Error::out_of_range("shape byte size"))
    }
}
