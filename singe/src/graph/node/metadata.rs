use crate::{
    backend::registry::{OperationSchema, OutputRole},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::node::{Node, NodeKind, NodeRef},
    layout::Layout,
    tensor::Tensor,
};

impl Node {
    /// Replaces this node's declared output metadata.
    pub fn replace_output_metadata(&mut self, data_type: DataType, layout: Layout, device: Device) {
        self.output_data_type = data_type;
        self.output_layout = layout;
        self.output_device = device;
    }

    /// Replaces this node's declared output data type.
    #[cfg(test)]
    pub fn replace_output_data_type(&mut self, data_type: DataType) {
        self.output_data_type = data_type;
    }

    /// Replaces this node's declared output layout.
    #[cfg(test)]
    pub fn replace_output_layout(&mut self, layout: Layout) {
        self.output_layout = layout;
    }

    /// Replaces this node's declared output device.
    #[cfg(test)]
    pub fn replace_output_device(&mut self, device: Device) {
        self.output_device = device;
    }
}

impl Node {
    /// Updates input node metadata from a concrete resolved tensor.
    pub fn update_input_metadata(&mut self, tensor: &Tensor) -> bool {
        if !matches!(self.kind, NodeKind::Input { .. }) {
            return false;
        }

        self.replace_output_metadata(
            tensor.data_type(),
            Layout::from_tensor(tensor),
            tensor.device(),
        );
        true
    }
}

pub fn validate_constant_metadata(node: &NodeRef<'_>, name: &str) -> Result<()> {
    let Some(tensor) = node.constant_tensor(0) else {
        return Ok(());
    };
    if tensor.data_type() != node.output_data_type() {
        return Err(Error::data_type_mismatch(
            format!("{name} constant"),
            tensor.data_type(),
            node.output_data_type(),
        ));
    }

    if tensor.device() != node.output_device() {
        return Err(Error::device_mismatch(
            format!("{name} constant"),
            tensor.device(),
            node.output_device(),
        ));
    }

    if tensor.shape().dimensions() != node.output_shape().dimensions() {
        return Err(Error::shape_mismatch(
            format!("{name} constant"),
            tensor.shape().dimensions().to_vec(),
            node.output_shape().dimensions().to_vec(),
        ));
    }
    Ok(())
}

impl NodeRef<'_> {
    /// Validates declared terminal and schema metadata for this node.
    ///
    /// # Errors
    ///
    /// Returns an error if a constant tensor disagrees with its declaration or the node output violates its operation schema.
    pub fn validate_declared_metadata(&self, name: impl Into<String>) -> Result<()> {
        let name = name.into();
        self.validate_terminal_declared_metadata(&name)?;
        self.validate_schema_declared_metadata(&name)
    }

    /// Validates declared metadata for terminal input and constant nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if a retained constant tensor has a different declared data type, device, or shape.
    pub fn validate_terminal_declared_metadata(&self, name: &str) -> Result<()> {
        validate_constant_metadata(self, name)?;
        Ok(())
    }

    /// Validates schema-declared output metadata for this node.
    ///
    /// # Errors
    ///
    /// Returns an error if a mask output is not boolean or the declared output data type, device, or shape differs from the schema requirement.
    pub fn validate_schema_declared_metadata(&self, name: &str) -> Result<()> {
        let Some(schema) = self.schema() else {
            return Ok(());
        };
        self.validate_schema_declared_metadata_with_schema(name, &schema)
    }

    /// Validates output metadata against a known operation schema.
    ///
    /// # Errors
    ///
    /// Returns an error if a mask output is not boolean or the declared output data type, device, or shape differs from the schema requirement.
    pub fn validate_schema_declared_metadata_with_schema(
        &self,
        name: &str,
        schema: &OperationSchema,
    ) -> Result<()> {
        if schema.single_output_role() == Some(OutputRole::Mask)
            && self.output_data_type() != DataType::Bool
        {
            return Err(Error::data_type_mismatch(
                format!("{name} comparison output"),
                DataType::Bool,
                self.output_data_type(),
            ));
        }

        if let Some(data_type) = schema.declared_output().data_type()
            && data_type != self.output_data_type()
        {
            return Err(Error::data_type_mismatch(
                format!("{name} {}", schema.name()),
                data_type,
                self.output_data_type(),
            ));
        }

        if let Some(device) = schema.declared_output().device()
            && device != self.output_device()
        {
            return Err(Error::device_mismatch(
                format!("{name} {}", schema.name()),
                device,
                self.output_device(),
            ));
        }

        if let Some(shape) = schema.declared_output().shape()
            && shape.dimensions() != self.output_shape().dimensions()
        {
            return Err(Error::shape_mismatch(
                format!("{name} shape"),
                shape.dimensions().to_vec(),
                self.output_shape().dimensions().to_vec(),
            ));
        }
        Ok(())
    }
}
