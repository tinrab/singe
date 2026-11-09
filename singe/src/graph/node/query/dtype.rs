use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    device::Device,
    graph::{
        ir::{ValueId, operation_attributes},
        node::NodeRef,
    },
};

impl NodeRef<'_> {
    /// Returns copy-to source input and target device.
    pub(crate) fn copy_to(&self) -> Option<(ValueId, Device)> {
        if self.operation() != Some(Operation::CopyTo) {
            return None;
        }

        let input = self.inputs().first().copied()?;
        let OperationAttributes::CopyTo(device) = operation_attributes(self) else {
            return None;
        };
        Some((input, device))
    }

    /// Returns cast source input and target dtype.
    pub(crate) fn cast(&self) -> Option<(ValueId, DataType)> {
        if self.operation() != Some(Operation::Cast) {
            return None;
        }
        Some((
            self.inputs().first().copied()?,
            operation_attributes(self).cast_data_type()?,
        ))
    }
}
