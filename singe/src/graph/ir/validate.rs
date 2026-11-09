use crate::{
    backend::registry::{OperationAttributes, OutputRole},
    data_type::DataType,
    error::{Error, Result},
    graph::{
        guards::GraphGuards,
        ir::{Graph, OperationNode},
        operation::view_output_layout,
    },
    layout::Layout,
};

impl Graph {
    /// Checks the IR's metadata contracts, as the graph passes' trailing validation did.
    ///
    /// Each operation must have an operand count its schema accepts, primary-result metadata that matches its constant or declared output, and operands on one device.
    /// Each view's layout must be the one its operation gives the source layout, and each initializer must satisfy its input guard.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first operation or initializer that breaks a contract.
    pub(crate) fn validate(&self) -> Result<()> {
        for operation in &self.operations {
            self.validate_operation(operation)
                .map_err(|error| error.with_graph_source(operation.source_location()))?;
        }
        GraphGuards::create_from_ir(self)?.validate_initializers_with(|_, _| Ok(()))
    }

    fn validate_operation(&self, operation: &OperationNode) -> Result<()> {
        let name = format!("ir validation operation {}", operation.id().index());
        let Some(value) = operation.result_values().next() else {
            return Ok(());
        };
        let value = self.required_value(value)?;
        if let Some(tensor) = self.constant_tensor(value.id()) {
            if tensor.data_type() != value.data_type() {
                return Err(Error::data_type_mismatch(
                    format!("{name} constant"),
                    tensor.data_type(),
                    value.data_type(),
                ));
            }

            if tensor.device() != value.device() {
                return Err(Error::device_mismatch(
                    format!("{name} constant"),
                    tensor.device(),
                    value.device(),
                ));
            }
            if tensor.shape() != value.shape() {
                return Err(Error::shape_mismatch(
                    format!("{name} constant"),
                    tensor.shape().dimensions().to_vec(),
                    value.shape().dimensions().to_vec(),
                ));
            }
        }

        let (Some(schema), Some(opcode)) = (operation.schema(), operation.opcode()) else {
            return Ok(());
        };
        if !schema.operands().accepts_count(operation.inputs().len()) {
            return Err(Error::unsupported_operation_with_details(
                "ir validation",
                format!(
                    "operation {} {} has {} operands, expected schema {:?}",
                    operation.id().index(),
                    schema.name(),
                    operation.inputs().len(),
                    schema.operands()
                ),
            ));
        }

        if schema.single_output_role() == Some(OutputRole::Mask)
            && value.data_type() != DataType::Bool
        {
            return Err(Error::data_type_mismatch(
                format!("{name} comparison output"),
                DataType::Bool,
                value.data_type(),
            ));
        }

        let declared = schema.declared_output();
        if let Some(data_type) = declared.data_type()
            && data_type != value.data_type()
        {
            return Err(Error::data_type_mismatch(
                format!("{name} {}", schema.name()),
                data_type,
                value.data_type(),
            ));
        }

        if let Some(device) = declared.device()
            && device != value.device()
        {
            return Err(Error::device_mismatch(
                format!("{name} {}", schema.name()),
                device,
                value.device(),
            ));
        }

        if let Some(shape) = declared.shape()
            && shape.dimensions() != value.shape().dimensions()
        {
            return Err(Error::shape_mismatch(
                format!("{name} shape"),
                shape.dimensions().to_vec(),
                value.shape().dimensions().to_vec(),
            ));
        }

        if let Some(&first) = operation.inputs().first() {
            let device = self.required_value(first)?.device();
            for &input in operation.inputs() {
                let input_device = self.required_value(input)?.device();
                if input_device != device {
                    return Err(Error::device_mismatch(
                        format!("{name} operands"),
                        device,
                        input_device,
                    ));
                }
            }

            let expected = match operation.attributes() {
                OperationAttributes::CopyTo(device) => *device,
                _ => device,
            };
            if expected != value.device() {
                return Err(Error::device_mismatch(
                    format!("{name} output"),
                    expected,
                    value.device(),
                ));
            }
        }

        if let Some(alias) = value.alias() {
            let source = self.required_value(alias.source())?;
            let expected = if value.device() == source.device() {
                view_output_layout(opcode, operation.attributes(), source.layout())?
            } else {
                Layout::from_shape(value.shape())
            };
            if &expected != value.layout() {
                return Err(Error::layout_mismatch(
                    format!("{name} alias layout"),
                    expected.dimensions().to_vec(),
                    expected.strides().to_vec(),
                    expected.storage_offset(),
                    value.layout().dimensions().to_vec(),
                    value.layout().strides().to_vec(),
                    value.layout().storage_offset(),
                ));
            }
        }
        Ok(())
    }
}
