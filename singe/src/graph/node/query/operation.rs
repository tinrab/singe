use crate::{
    backend::registry::Operation,
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
};

impl NodeRef<'_> {
    /// Returns matmul inputs.
    pub(crate) fn matmul_inputs(&self) -> Option<[ValueId; 2]> {
        if matches!(
            self.operation(),
            Some(Operation::MatMul | Operation::QuantizedMatMul)
        ) {
            let inputs = self.inputs();
            return Some([*inputs.first()?, *inputs.get(1)?]);
        }
        None
    }

    /// Returns inputs and modes for contraction nodes.
    pub(crate) fn contraction(&self) -> Option<ContractionParameters> {
        if self.operation() != Some(Operation::Contraction) {
            return None;
        }

        let inputs = self.inputs();
        let operation_attributes = operation_attributes(self);
        let attributes = operation_attributes.contraction()?;
        Some((
            *inputs.first()?,
            attributes.a_mode.clone(),
            *inputs.get(1)?,
            attributes.b_mode.clone(),
            *inputs.get(2)?,
            attributes.c_mode.clone(),
            attributes.d_mode.clone(),
            attributes.alpha,
            attributes.beta,
        ))
    }

    /// Returns condition, true-branch, and false-branch inputs for where nodes.
    pub(crate) fn where_inputs(&self) -> Option<(ValueId, ValueId, ValueId)> {
        if self.operation() != Some(Operation::Where) {
            return None;
        }

        let inputs = self.inputs();
        Some((*inputs.first()?, *inputs.get(1)?, *inputs.get(2)?))
    }

    /// Returns endpoints and interpolation parameter for lerp nodes.
    pub(crate) fn lerp_inputs(&self) -> Option<(ValueId, ValueId, f32)> {
        if self.operation() != Some(Operation::Lerp) {
            return None;
        }

        let inputs = self.inputs();
        Some((
            *inputs.first()?,
            *inputs.get(1)?,
            operation_attributes(self).lerp_t()?,
        ))
    }

    /// Returns inputs for binary elementwise nodes.
    pub(crate) fn binary_elementwise_inputs(&self) -> Option<[ValueId; 2]> {
        let operation = self.operation()?;
        if !operation.is_binary_elementwise() {
            return None;
        }

        let inputs = self.inputs();
        Some([*inputs.first()?, *inputs.get(1)?])
    }
}

type ContractionParameters = (
    ValueId,
    Vec<i64>,
    ValueId,
    Vec<i64>,
    ValueId,
    Vec<i64>,
    Vec<i64>,
    f32,
    f32,
);
