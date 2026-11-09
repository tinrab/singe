use crate::{
    backend::registry::Operation,
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
};

impl NodeRef<'_> {
    /// Returns the source input, padding, and fill value for pad operations.
    pub(crate) fn pad(&self) -> Option<PadParameters> {
        if self.operation() != Some(Operation::Pad) {
            return None;
        }

        let input = self.inputs().first().copied()?;
        let operation_attributes = operation_attributes(self);
        let attributes = operation_attributes.pad()?;
        Some((input, attributes.padding.clone(), attributes.value))
    }

    /// Returns concat inputs and axis.
    pub(crate) fn concat(&self) -> Option<(Vec<ValueId>, i64)> {
        if self.operation() != Some(Operation::Concat) {
            return None;
        }
        Some((
            self.inputs().to_vec(),
            operation_attributes(self).concat_dimension()?,
        ))
    }

    /// Returns repeat-kv input and repeat count.
    pub(crate) fn repeat_kv(&self) -> Option<(ValueId, i64)> {
        if self.operation() != Some(Operation::RepeatKv) {
            return None;
        }
        Some((
            self.inputs().first().copied()?,
            operation_attributes(self).repeat_kv_repeats()?,
        ))
    }
}

type PadParameters = (ValueId, Vec<(i64, i64)>, f32);
