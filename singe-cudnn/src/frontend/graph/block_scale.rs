pub use crate::backend::execution::block_scale::{
    BlockScaleDequantizeInputs, BlockScaleQuantizeOutputs,
};
use crate::{
    backend::tensor::{TensorId, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig},
        operation::{
            BlockScaleDequantizeConfig, BlockScaleOperation, BlockScaleQuantizeConfig, Operation,
        },
    },
};

impl Graph {
    /// Adds a block scale quantize operation with explicit outputs.
    ///
    /// Block scale quantize produces a quantized output tensor and a scale tensor
    /// from a higher precision input.
    /// Explicit layouts are passed to native support selection; axis and transpose
    /// configure inference rather than overriding these declarations.
    ///
    /// # Errors
    ///
    /// Returns an error if an input or output tensor identity is unknown, the block size is nonpositive or does not fit `i32`, or quantized output dimensions differ from the input.
    pub fn block_scale_quantize_into(
        &mut self,
        input: TensorId,
        outputs: BlockScaleQuantizeOutputs,
        config: BlockScaleQuantizeConfig,
    ) -> Result<()> {
        let operation = BlockScaleOperation::Quantize {
            input,
            output: outputs.output,
            scale: outputs.scale,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::BlockScale(operation));

        Ok(())
    }

    /// Adds block scale quantize and creates output and scale tensors.
    ///
    /// Select placement and encoding independently for quantized values and their block scales.
    /// An omitted encoding inherits the graph's I/O or intermediate policy according to placement.
    /// Both results require an explicit or inherited encoding; no quantization format is guessed.
    /// A configured operation scope names the results `name::Y` and `name::scale`.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is unknown, native block size, axis, or inferred layout is invalid or unrepresentable, required value or scale encodings are unspecified, or operation validation fails.
    pub fn block_scale_quantize(
        &mut self,
        input: TensorId,
        outputs: BlockScaleQuantizeOutputs<TensorOutputConfig>,
        config: BlockScaleQuantizeConfig,
    ) -> Result<BlockScaleQuantizeOutputs> {
        let input_tensor = self.tensor_spec(input)?;
        let shapes = config.infer_shapes(&input_tensor.shape)?;
        let output_type = outputs
            .output
            .configured_data_type(self.config.data_type_policy)
            .ok_or(FrontendError::OutputDataTypeRequired {
                output: "block scale quantize output",
            })?;
        let scale_type = outputs
            .scale
            .configured_data_type(self.config.data_type_policy)
            .ok_or(FrontendError::OutputDataTypeRequired {
                output: "block scale quantize scale",
            })?;
        let output_spec =
            outputs
                .output
                .into_spec(self.config.data_type_policy, output_type, shapes.output);
        let scale_spec =
            outputs
                .scale
                .into_spec(self.config.data_type_policy, scale_type, shapes.scale);
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        let scale = self.inferred_tensor(scale_spec, config.name(), "scale");
        let outputs = BlockScaleQuantizeOutputs::new(output, scale);
        if let Err(error) = self.block_scale_quantize_into(input, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    /// Adds a block scale dequantize operation with an explicit output.
    ///
    /// Block scale dequantize combines quantized values with block-broadcast
    /// scale data to produce a dequantized output.
    /// Caller-declared scale geometry, encoding and layout reach native support selection.
    ///
    /// # Errors
    ///
    /// Returns an error if a tensor identity is unknown, block sizes are empty or nonpositive, the output is not virtual, or output dimensions differ from the quantized input.
    pub fn block_scale_dequantize_into(
        &mut self,
        inputs: BlockScaleDequantizeInputs,
        output: TensorId,
        config: BlockScaleDequantizeConfig,
    ) -> Result<()> {
        let operation = BlockScaleOperation::Dequantize {
            input: inputs.input,
            scale: inputs.scale,
            output,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::BlockScale(operation));

        Ok(())
    }

    /// Adds block scale dequantize and creates its required virtual output tensor.
    /// The C++ frontend requires virtual placement for this native operation.
    /// Result encoding follows the explicit inference choice or intermediate policy, defaulting to `F32`.
    /// Compute precision is resolved separately during lowering.
    /// A configured operation scope names the result `name::Y`.
    ///
    /// # Errors
    ///
    /// Returns an error if an input or scale tensor is unknown, block sizes are empty or nonpositive, or the inferred dequantization operation fails validation.
    pub fn block_scale_dequantize(
        &mut self,
        inputs: BlockScaleDequantizeInputs,
        config: BlockScaleDequantizeConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_spec(inputs.input)?;
        let output_data_type = config
            .output_data_type()
            .unwrap_or_else(|| self.effective_intermediate_data_type(DataType::F32));
        let output_spec =
            TensorSpec::new(output_data_type, input_tensor.shape.clone()).virtual_tensor();
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.block_scale_dequantize_into(inputs, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }
}
