use crate::{error::Result, tensor::Tensor};

impl Tensor {
    /// Alias for [`Tensor::truncate`].
    ///
    /// # Errors
    ///
    /// Returns an error if truncation is unsupported by the backend.
    pub fn trunc(&self) -> Result<Self> {
        self.truncate()
    }
}
