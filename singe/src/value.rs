use crate::{data_type::DataType, shape::Shape};

/// Logical metadata for a value, independent of storage, strides, and placement.
///
/// This contract describes concrete dimensions.
/// `GraphInputSpec` pairs a representative spec with explicit symbolic guards; `DimensionExpr` records metadata-derived dimensions for specialization.
/// Neither requires fabricated allocation layouts for unresolved sizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueSpec {
    data_type: DataType,
    shape: Shape,
}

impl ValueSpec {
    /// Creates a logical specification from an already validated shape.
    #[must_use]
    pub fn new(data_type: DataType, shape: Shape) -> Self {
        Self { data_type, shape }
    }

    /// Returns the logical element type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the logical dimensions.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
