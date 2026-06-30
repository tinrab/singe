use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        infer::{
            infer_block_scale_dequantize_shape, infer_block_scale_quantize_output_shape,
            infer_block_scale_shape_with_layout,
        },
        operation::{
            BlockScaleDequantizeConfig, BlockScaleOperation, BlockScaleQuantizeConfig, Operation,
        },
        support,
    },
    tensor::{TensorId, TensorSpec},
    version,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleQuantizeOutputs {
    pub output: TensorId,
    pub scale: TensorId,
}

impl BlockScaleQuantizeOutputs {
    pub fn new(output: TensorId, scale: TensorId) -> Self {
        Self { output, scale }
    }
}

impl Graph {
    /// Adds a block scale quantize operation with explicit outputs.
    ///
    /// Block scale quantize produces a quantized output tensor and a scale tensor
    /// from a higher precision input.
    pub fn block_scale_quantize(
        &mut self,
        input: TensorId,
        output: TensorId,
        scale: TensorId,
        config: BlockScaleQuantizeConfig,
    ) -> Result<()> {
        self.validate_block_scale_quantize_support_surface_for_version(version()?.raw())?;

        let input_tensor = self.tensor_config(input)?.clone();

        let expected_output = infer_block_scale_quantize_output_shape(
            &input_tensor.shape,
            config.axis(),
            config.transpose(),
        )?;
        self.validate_tensor_dimensions(
            output,
            expected_output.dimensions(),
            "block scale quantize output shape",
        )?;
        let expected_scale = infer_block_scale_shape_with_layout(
            &input_tensor.shape,
            config.block_size(),
            config.axis(),
            config.transpose(),
        )?;
        self.validate_tensor_dimensions(
            scale,
            expected_scale.dimensions(),
            "block scale quantize scale shape",
        )?;

        self.operations
            .push(Operation::BlockScale(BlockScaleOperation::Quantize {
                input,
                output,
                scale,
                config,
            }));

        Ok(())
    }

    /// Adds block scale quantize and creates output and scale tensors.
    pub fn block_scale_quantize_infer(
        &mut self,
        input: TensorId,
        config: BlockScaleQuantizeConfig,
    ) -> Result<BlockScaleQuantizeOutputs> {
        let input_tensor = self.tensor_config(input)?.clone();
        let output_data_type = self.effective_intermediate_data_type(config.compute_type());
        let output_shape = infer_block_scale_quantize_output_shape(
            &input_tensor.shape,
            config.axis(),
            config.transpose(),
        )?;
        let scale_shape = infer_block_scale_shape_with_layout(
            &input_tensor.shape,
            config.block_size(),
            config.axis(),
            config.transpose(),
        )?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape).virtual_tensor());
        let scale = self.tensor(TensorSpec::new(output_data_type, scale_shape).virtual_tensor());
        if let Err(error) = self.block_scale_quantize(input, output, scale, config) {
            self.tensors.remove(&output);
            self.tensors.remove(&scale);
            return Err(error);
        }
        Ok(BlockScaleQuantizeOutputs::new(output, scale))
    }

    /// Adds a block scale dequantize operation with an explicit output.
    ///
    /// Block scale dequantize combines quantized values with block-broadcast
    /// scale data to produce a dequantized output.
    pub fn block_scale_dequantize(
        &mut self,
        input: TensorId,
        scale: TensorId,
        output: TensorId,
        config: BlockScaleDequantizeConfig,
    ) -> Result<()> {
        self.validate_block_scale_dequantize_support_surface_for_version(version()?.raw())?;

        let input_tensor = self.tensor_config(input)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();

        if !output_tensor.is_virtual {
            return Err(Error::FrontendBindingTypeMismatch {
                tensor_id: output,
                expected: "virtual tensor".into(),
                actual: "bindable tensor".into(),
            });
        }
        self.validate_tensor_dimensions(
            output,
            input_tensor.shape.dimensions(),
            "block scale dequantize output shape",
        )?;
        let expected_scale =
            infer_block_scale_dequantize_shape(&input_tensor.shape, config.block_sizes())?;
        self.validate_tensor_dimensions(
            scale,
            expected_scale.dimensions(),
            "block scale dequantize scale shape",
        )?;

        self.operations
            .push(Operation::BlockScale(BlockScaleOperation::Dequantize {
                input,
                scale,
                output,
                config,
            }));

        Ok(())
    }

    /// Adds block scale dequantize and creates the output tensor.
    pub fn block_scale_dequantize_infer(
        &mut self,
        input: TensorId,
        scale: TensorId,
        config: BlockScaleDequantizeConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_config(input)?.clone();
        let output_data_type =
            self.effective_intermediate_data_type(config.compute_type().unwrap_or(DataType::F32));
        let output = self
            .tensor(TensorSpec::new(output_data_type, input_tensor.shape.clone()).virtual_tensor());
        if let Err(error) = self.block_scale_dequantize(input, scale, output, config) {
            self.tensors.remove(&output);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn validate_block_scale_quantize_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::BLOCK_SCALE_QUANTIZE.require_frontend_feature(cudnn_version)
    }

    pub(crate) fn validate_block_scale_dequantize_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::BLOCK_SCALE_DEQUANTIZE.require_frontend_feature(cudnn_version)
    }
}
