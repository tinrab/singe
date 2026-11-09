mod request;
mod view;
mod view_shape;

pub use self::{
    view::{view_output_layout, view_output_shape},
    view_shape::{view_flatten_output_shape, view_start_offset_delta},
};
use crate::{
    backend::registry::{Operation as RegistryOperation, OperationAttributes},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::ir::ValueId,
    shape::Shape,
};

/// Declared metadata for an operation output.
#[derive(Debug, Clone)]
pub struct GraphOperationOutput {
    pub(crate) data_type: DataType,
    pub(crate) shape: Shape,
    pub(crate) device: Device,
}

/// Construction request with canonical operands, attributes, and declared output metadata.
#[derive(Debug, Clone)]
pub struct GraphOperationRequest {
    pub(crate) name: Option<String>,
    pub(crate) inputs: GraphOperationInputs,
    pub(crate) output: GraphOperationOutput,
}

/// Graph-local operands with the same opcode and attributes used by eager requests and normalized IR.
/// Logical inference checks the request at insertion.
#[derive(Debug, Clone)]
pub struct GraphOperationInputs {
    pub(crate) operation: RegistryOperation,
    pub(crate) attributes: OperationAttributes,
    pub(crate) inputs: Vec<ValueId>,
}

impl GraphOperationInputs {
    /// Projects a canonical invocation into the construction graph's node form.
    /// Normalization reunites `TopK`'s value and index projections in the IR.
    pub(crate) fn for_result(
        operation: RegistryOperation,
        attributes: OperationAttributes,
        inputs: Vec<ValueId>,
        index: usize,
    ) -> Result<Self> {
        let operation = match (operation, index) {
            (RegistryOperation::TopK, 0) => RegistryOperation::TopKValues,
            (RegistryOperation::TopK, 1) => RegistryOperation::TopKIndices,
            (RegistryOperation::PagedAttention, 0) => RegistryOperation::PagedAttentionOutput,
            (RegistryOperation::PagedAttention, 1) => RegistryOperation::PagedAttentionLse,
            (RegistryOperation::PagedMlaAttention, 0) => RegistryOperation::PagedMlaAttentionOutput,
            (RegistryOperation::PagedMlaAttention, 1) => RegistryOperation::PagedMlaAttentionLse,
            (operation, 0) => operation,
            _ => return Err(Error::missing_output("for result")),
        };
        Ok(Self::new(operation, attributes, inputs))
    }

    /// Stores a registry operation, typed attributes, and ordered graph operand IDs.
    /// Logical validation occurs when the operation is inserted into a graph.
    #[must_use]
    pub fn new(
        operation: RegistryOperation,
        attributes: OperationAttributes,
        inputs: Vec<ValueId>,
    ) -> Self {
        Self {
            operation,
            attributes,
            inputs,
        }
    }

    /// Returns the registry operation code.
    #[must_use]
    pub fn operation(&self) -> RegistryOperation {
        self.operation
    }

    /// Returns the typed operation attributes.
    #[must_use]
    pub fn attributes(&self) -> &OperationAttributes {
        &self.attributes
    }

    /// Returns operand node IDs in canonical operation input order.
    #[must_use]
    pub fn operands(&self) -> &[ValueId] {
        &self.inputs
    }
}
