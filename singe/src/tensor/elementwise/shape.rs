use crate::{
    backend::registry::{Operation, OperationAttributes, PadAttributes},
    dispatch::call_one,
    error::Result,
    shape::Shape,
    tensor::Tensor,
};

impl Tensor {
    /// Pads the tensor with `value` according to per-axis `(before, after)` pairs.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if padding is invalid or unsupported by the backend.
    pub fn pad(&self, padding: &[(i64, i64)], value: f32) -> Result<Self> {
        call_one(
            Operation::Pad,
            OperationAttributes::Pad(PadAttributes {
                padding: padding.to_vec(),
                value,
            }),
            &[self],
        )
    }

    /// Returns an immutable broadcast view, using zero strides for expanded axes.
    ///
    /// This records a graph operation when called inside graph capture.
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` is not a valid expansion target or the backend cannot expand it.
    pub fn expand(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = &shape.into();
        call_one(
            Operation::Expand,
            OperationAttributes::Shape(shape.clone()),
            &[self],
        )
    }
}
