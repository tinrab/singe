#[cfg(feature = "onnx")]
use crate::{
    backend::registry::OperationAttributes,
    graph::{GraphOperationInputs, GraphOperationOutput, GraphOperationRequest},
};
use crate::{
    backend::registry::{Operation as RegistryOperation, operation_schema},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{Graph, ir::ValueId, node::NodeKind},
    shape::Shape,
    tensor::Tensor,
};

impl Graph {
    /// Adds a named quantization operation selected by registry opcode.
    ///
    /// # Errors
    ///
    /// Returns an error if `operation` is not a supported quantization operation, the input node is missing, or declared output metadata is invalid.
    pub(crate) fn add_named_quantization_operation(
        &mut self,
        name: Option<String>,
        operation: RegistryOperation,
        input: ValueId,
        scale: f32,
        zero_point: i64,
        data_type: DataType,
        shape: Shape,
        device: Device,
    ) -> Result<ValueId> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::InvalidArgument {
                name: "graph quantization scale".into(),
            });
        }

        let (scale, zero_point) = self.add_quantization_constants(scale, zero_point, device)?;
        let kind = NodeKind::quantization_operation(operation, input, scale, zero_point, data_type)
            .ok_or_else(|| Error::unsupported_operation(operation_schema(operation).name()))?;
        self.add_named_operation(name, kind, data_type, shape, device)
    }

    /// Adds a named quantized matrix multiplication with per-tensor scales and zero points.
    ///
    /// # Errors
    ///
    /// Returns an error if an input node is missing or the declared output metadata is invalid.
    #[cfg(feature = "onnx")]
    pub(crate) fn add_named_quantized_matmul_operation(
        &mut self,
        name: Option<String>,
        inputs: [ValueId; 2],
        lhs: (f32, i64),
        rhs: (f32, i64),
        output: GraphOperationOutput,
    ) -> Result<ValueId> {
        let device = output.device;
        let (lhs_scale, lhs_zero_point) = self.add_quantization_constants(lhs.0, lhs.1, device)?;
        let (rhs_scale, rhs_zero_point) = self.add_quantization_constants(rhs.0, rhs.1, device)?;
        self.add_operation_request(GraphOperationRequest::new(
            name,
            GraphOperationInputs::new(
                RegistryOperation::QuantizedMatMul,
                OperationAttributes::None,
                vec![
                    inputs[0],
                    inputs[1],
                    lhs_scale,
                    lhs_zero_point,
                    rhs_scale,
                    rhs_zero_point,
                ],
            ),
            output,
        ))
    }

    /// Adds scalar scale and zero-point constants on `device`.
    fn add_quantization_constants(
        &mut self,
        scale: f32,
        zero_point: i64,
        device: Device,
    ) -> Result<(ValueId, ValueId)> {
        let scale = self.add_constant(Tensor::scalar(device, scale)?);
        let zero_point = self.add_constant(Tensor::scalar(device, zero_point)?);
        Ok((scale, zero_point))
    }
}
