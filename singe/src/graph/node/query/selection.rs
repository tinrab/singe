use crate::{
    backend::registry::Operation,
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
};

impl NodeRef<'_> {
    /// Returns the source input and bounds for clamp operations.
    pub(crate) fn clamp_bounds(&self) -> Option<(ValueId, f32, f32)> {
        let input = self.inputs().first().copied()?;
        if self.operation() != Some(Operation::Clamp) {
            return None;
        }

        let clamp = operation_attributes(self).clamp().copied()?;
        Some((input, clamp.min, clamp.max))
    }
}
