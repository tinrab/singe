mod basic;
mod convert;
mod neural;
mod reduction;
mod spatial;

use crate::{
    data_type::DataType,
    device::Device,
    graph::operation::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
    shape::Shape,
};

impl GraphOperationRequest {
    /// Builds a request using canonical catalog attributes and graph-local operands.
    /// Graph insertion checks inference against the declared output before mutation.
    #[must_use]
    pub fn new(
        name: Option<String>,
        inputs: GraphOperationInputs,
        output: GraphOperationOutput,
    ) -> Self {
        Self {
            name,
            inputs,
            output,
        }
    }
}

impl GraphOperationOutput {
    /// Stores the declared output data type, shape, and device without backend validation.
    #[must_use]
    pub fn new(data_type: DataType, shape: Shape, device: Device) -> Self {
        Self {
            data_type,
            shape,
            device,
        }
    }
}
