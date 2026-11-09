use crate::{
    backend::registry::Operation,
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
    scalar::Scalar,
};

impl NodeRef<'_> {
    /// Returns the source input and scalar for one expected scalar operation.
    pub(crate) fn scalar_operation(
        &self,
        expected_operation: Operation,
    ) -> Option<(ValueId, Scalar)> {
        if self.operation() != Some(expected_operation) {
            return None;
        }
        Some((
            self.scalar_elementwise_input()?,
            operation_attributes(self).scalar()?,
        ))
    }

    /// Returns the input for scalar elementwise nodes.
    pub(crate) fn scalar_elementwise_input(&self) -> Option<ValueId> {
        let operation = self.operation()?;
        if !operation.is_scalar_elementwise() {
            return None;
        }
        self.inputs().first().copied()
    }
}
