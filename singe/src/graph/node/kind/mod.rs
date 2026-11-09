#[cfg(test)]
use crate::device::Device;
#[cfg(any(test, feature = "onnx"))]
use crate::{
    backend::registry::{
        FlattenAttributes, OperationAttributeKind, PadAttributes, QuantizationAttributes,
        SliceAttributes,
    },
    data_type::DataType,
};
use crate::{
    backend::registry::{Operation, OperationAttributes, TransposeAttributes},
    graph::{GraphOperationInputs, ir::ValueId},
    shape::Shape,
    tensor::Tensor,
};

#[cfg(feature = "onnx")]
use crate::backend::registry::ClampAttributes;

#[cfg(test)]
mod neural;

/// Construction terminals and canonical operation invocations.
#[derive(Debug, Clone)]
pub enum NodeKind {
    Input { name: String },
    Constant { tensor: Tensor },
    Operation(GraphOperationInputs),
}

impl NodeKind {
    pub(crate) fn operation_invocation(
        operation: Operation,
        attributes: OperationAttributes,
        inputs: Vec<ValueId>,
    ) -> Self {
        Self::Operation(GraphOperationInputs::new(operation, attributes, inputs))
    }
}

impl NodeKind {
    /// Builds a cast node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn cast(input: ValueId, data_type: DataType) -> Self {
        Self::operation_invocation(
            Operation::Cast,
            OperationAttributes::Cast(data_type),
            vec![input],
        )
    }

    /// Builds a copy-to node kind.
    #[cfg(test)]
    pub(crate) fn copy_to(input: ValueId, device: Device) -> Self {
        Self::operation_invocation(
            Operation::CopyTo,
            OperationAttributes::CopyTo(device),
            vec![input],
        )
    }

    /// Builds a clamp node kind.
    #[cfg(feature = "onnx")]
    pub(crate) fn clamp(input: ValueId, min: f32, max: f32) -> Self {
        Self::operation_invocation(
            Operation::Clamp,
            OperationAttributes::Clamp(ClampAttributes { min, max }),
            vec![input],
        )
    }

    /// Builds a lerp node kind.
    #[cfg(test)]
    pub(crate) fn lerp(a: ValueId, b: ValueId, t: f32) -> Self {
        Self::operation_invocation(Operation::Lerp, OperationAttributes::Lerp(t), vec![a, b])
    }

    /// Builds a where node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn where_operation(condition: ValueId, x: ValueId, y: ValueId) -> Self {
        Self::operation_invocation(
            Operation::Where,
            OperationAttributes::None,
            vec![condition, x, y],
        )
    }

    /// Builds a quantization operation node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn quantization_operation(
        operation: Operation,
        input: ValueId,
        scale: ValueId,
        zero_point: ValueId,
        data_type: DataType,
    ) -> Option<Self> {
        operation
            .has_attribute_kind(OperationAttributeKind::Quantization)
            .then_some(Self::operation_invocation(
                operation,
                OperationAttributes::Quantization(QuantizationAttributes {
                    axis: None,
                    block_size: None,
                    data_type: (operation == Operation::Quantize).then_some(data_type),
                    packed: None,
                }),
                vec![input, scale, zero_point],
            ))
    }
}

impl NodeKind {
    /// Builds a graph input node kind.
    pub(crate) fn input(name: String) -> Self {
        Self::Input { name }
    }

    /// Builds an embedded constant node kind.
    pub(crate) fn constant(tensor: Tensor) -> Self {
        Self::Constant { tensor }
    }
}

impl NodeKind {
    /// Builds a reshape node kind.
    pub(crate) fn reshape(input: ValueId, shape: Shape) -> Self {
        Self::operation_invocation(
            Operation::Reshape,
            OperationAttributes::Shape(shape),
            vec![input],
        )
    }

    /// Builds an expand node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn expand(input: ValueId, shape: Shape) -> Self {
        Self::operation_invocation(
            Operation::Expand,
            OperationAttributes::Shape(shape),
            vec![input],
        )
    }

    /// Builds a pad node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn pad(input: ValueId, padding: Vec<(i64, i64)>, value: f32) -> Self {
        Self::operation_invocation(
            Operation::Pad,
            OperationAttributes::Pad(PadAttributes { padding, value }),
            vec![input],
        )
    }

    /// Builds a transpose node kind.
    pub(crate) fn transpose(input: ValueId, dim0: i64, dim1: i64) -> Self {
        Self::operation_invocation(
            Operation::Transpose,
            OperationAttributes::Transpose(TransposeAttributes { dim0, dim1 }),
            vec![input],
        )
    }

    /// Builds a permute node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn permute(input: ValueId, order: Vec<i64>) -> Self {
        Self::operation_invocation(
            Operation::Permute,
            OperationAttributes::Permute(order),
            vec![input],
        )
    }

    /// Builds a slice node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn slice(input: ValueId, dim: i64, start: i64, length: i64) -> Self {
        Self::operation_invocation(
            Operation::Slice,
            OperationAttributes::Slice(SliceAttributes {
                dim,
                start,
                length,
                step: 1,
            }),
            vec![input],
        )
    }

    /// Builds a concat node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn concat(inputs: Vec<ValueId>, dim: i64) -> Self {
        Self::operation_invocation(Operation::Concat, OperationAttributes::Concat(dim), inputs)
    }

    /// Builds an index-select node kind.
    #[cfg(test)]
    pub(crate) fn index_select(input: ValueId, dim: i64, indices: ValueId) -> Self {
        Self::operation_invocation(
            Operation::IndexSelect,
            OperationAttributes::IndexSelect(dim),
            vec![input, indices],
        )
    }

    /// Builds a flatten node kind.
    #[cfg(any(test, feature = "onnx"))]
    pub(crate) fn flatten(input: ValueId, start_dim: i64, end_dim: i64) -> Self {
        Self::operation_invocation(
            Operation::Flatten,
            OperationAttributes::Flatten(FlattenAttributes { start_dim, end_dim }),
            vec![input],
        )
    }

    /// Builds a contiguous node kind.
    pub(crate) fn contiguous(input: ValueId) -> Self {
        Self::operation_invocation(
            Operation::Contiguous,
            OperationAttributes::None,
            vec![input],
        )
    }
}

#[cfg(test)]
mod tests;
