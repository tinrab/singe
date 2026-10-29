use crate::{
    backend::tensor::{BackendTensorReordering, BackendTensorUid, Shape, Tensor},
    data_type::DataType,
    scalar::{SCALAR_STORAGE_BYTES, ScalarValue},
};

/// Hashable declaration of one native tensor for execution-plan cache keys.
///
/// Includes native identity, layout, encoding, placement and addressing metadata.
/// Compile-time constants compare by their encoded bytes, preserving floating-point
/// payloads and signed zero. Runtime scalar values and device addresses are excluded.
/// Building a snapshot clones the checked shape without querying cuDNN.
/// The snapshot retains no tensor, descriptor or allocation owners.
///
/// Combine these snapshots with operand roles, every operation's attributes, the
/// compatible device/context and compilation policy to identify a complete graph.
/// Runtime shape overrides require their own key information.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorProblemKey {
    /// Native tensor identity used in graph operands and execution bindings.
    pub id: BackendTensorUid,
    /// Native scalar or packed tensor encoding.
    pub data_type: DataType,
    /// Checked dimensions and strides in native descriptor element units.
    pub shape: Shape,
    /// Declared byte alignment required of bound addresses.
    pub byte_alignment: i64,
    /// Whether storage is a cuDNN-managed virtual intermediate.
    pub is_virtual: bool,
    /// Includes runtime scalars and compile-time constants.
    pub is_by_value: bool,
    /// Zero-padded native scalar encoding, present only for compile-time constants.
    pub constant_bytes: Option<[u8; SCALAR_STORAGE_BYTES]>,
    /// Optional number of scalar lanes per native descriptor element.
    pub vector_count: Option<i64>,
    /// Optional zero-based dimension whose elements contain the declared vector lanes.
    pub vectorized_dimension: Option<i64>,
    /// Optional native tensor-memory reordering declaration.
    pub reordering: Option<BackendTensorReordering>,
    /// Native identity of the optional ragged-offset tensor.
    pub ragged_offset: Option<BackendTensorUid>,
    /// Number of tensor elements represented by each ragged-offset unit.
    pub ragged_offset_multiplier: i64,
    /// Optional declared physical storage span of ragged tensor memory, in bytes.
    pub ragged_byte_capacity: Option<usize>,
}

impl From<&Tensor> for TensorProblemKey {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: tensor.id(),
            data_type: tensor.data_type(),
            shape: tensor.shape().clone(),
            byte_alignment: tensor.byte_alignment(),
            is_virtual: tensor.is_virtual(),
            is_by_value: tensor.is_by_value(),
            constant_bytes: tensor.constant_value().map(ScalarValue::native_bytes),
            vector_count: tensor.vector_count(),
            vectorized_dimension: tensor.vectorized_dimension(),
            reordering: tensor.reordering(),
            ragged_offset: tensor.ragged_offset(),
            ragged_offset_multiplier: tensor.ragged_offset_multiplier(),
            ragged_byte_capacity: tensor.ragged_byte_capacity(),
        }
    }
}
