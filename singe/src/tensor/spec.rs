use crate::{data_type::DataType, device::Device, error::Result, shape::Shape};

/// Metadata required to allocate or validate a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Device that owns the tensor storage.
    pub device: Device,
    /// Element data type stored by the tensor.
    pub data_type: DataType,
    /// Logical tensor shape.
    pub shape: Shape,
}

impl TensorSpec {
    /// Creates tensor metadata after validating its byte size.
    ///
    /// # Errors
    ///
    /// Returns an error if `shape.element_count() * data_type.size_bytes()` overflows.
    pub fn create(device: Device, data_type: DataType, shape: Shape) -> Result<Self> {
        shape.byte_size(data_type.size_bytes())?;
        Ok(Self {
            device,
            data_type,
            shape,
        })
    }

    /// Returns the number of logical tensor elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.element_count()
    }

    /// Returns the number of bytes required by this tensor specification.
    ///
    /// # Errors
    ///
    /// Returns an error if byte-size calculation overflows.
    pub fn byte_size(&self) -> Result<usize> {
        self.shape.byte_size(self.data_type.size_bytes())
    }
}
