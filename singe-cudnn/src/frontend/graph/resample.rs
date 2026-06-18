use crate::{
    data_type::DataType,
    error::{Error, Result},
    execution::resample::ResampleMode,
    frontend::{
        graph::Graph,
        infer::infer_resample_output,
        operation::{Operation, ResampleConfig},
    },
    tensor::{TensorId, TensorSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleOutputs {
    pub output: TensorId,
    pub indices: Option<TensorId>,
}

impl ResampleOutputs {
    pub fn new(output: TensorId, indices: Option<TensorId>) -> Self {
        Self { output, indices }
    }
}

impl Graph {
    /// Adds a resampling forward operation with an explicit output tensor.
    ///
    /// Resampling changes spatial dimensions using modes such as average
    /// pooling, max pooling, bilinear, or cubic interpolation.
    /// Max pooling may also produce an index tensor.
    pub fn resample(
        &mut self,
        input: TensorId,
        output: TensorId,
        indices: Option<TensorId>,
        config: ResampleConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let expected_output = infer_resample_output(&input_tensor.shape, &config)?;
        self.validate_tensor_data_type(
            output,
            self.effective_io_data_type(input_tensor.data_type),
            "resample output",
        )?;
        self.validate_tensor_dimensions(
            output,
            expected_output.dimensions(),
            "resample output shape",
        )?;
        self.validate_resample_indices(
            indices,
            expected_output.dimensions(),
            config.mode(),
            "resample indices",
            "resample indices shape",
        )?;

        self.operations.push(Operation::Resample {
            input,
            output,
            indices,
            config,
        });
        Ok(())
    }

    /// Adds a resampling forward operation and creates the output tensor.
    pub fn resample_infer(
        &mut self,
        input: TensorId,
        indices: Option<TensorId>,
        config: ResampleConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_config(input)?.clone();
        let inferred_output = infer_resample_output(&input_tensor.shape, &config)?;
        let output_shape = Self::shape_preserving_input_format(
            &input_tensor.shape,
            inferred_output.dimensions().to_vec(),
        )?;
        self.validate_resample_indices(
            indices,
            output_shape.dimensions(),
            config.mode(),
            "resample indices",
            "resample indices shape",
        )?;
        let output_data_type = self.effective_io_data_type(input_tensor.data_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.resample(input, output, indices, config)?;
        Ok(output)
    }

    /// Adds a resampling forward operation and creates optional max-pool indices.
    pub fn resample_infer_with_indices(
        &mut self,
        input: TensorId,
        config: ResampleConfig,
    ) -> Result<ResampleOutputs> {
        let input_tensor = self.tensor_config(input)?.clone();
        let inferred_output = infer_resample_output(&input_tensor.shape, &config)?;
        let output_shape = Self::shape_preserving_input_format(
            &input_tensor.shape,
            inferred_output.dimensions().to_vec(),
        )?;
        let output_data_type = self.effective_io_data_type(input_tensor.data_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape.clone()));
        let indices = if config.mode() == ResampleMode::MaxPool {
            Some(self.tensor(TensorSpec::new(DataType::I8, output_shape)))
        } else {
            None
        };
        self.resample(input, output, indices, config)?;
        Ok(ResampleOutputs::new(output, indices))
    }

    /// Adds a resampling backward operation with explicit gradient tensors.
    pub fn resample_backward(
        &mut self,
        input: TensorId,
        output: TensorId,
        output_gradient: TensorId,
        input_gradient: TensorId,
        indices: Option<TensorId>,
        config: ResampleConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        self.validate_tensor_data_type(
            input_gradient,
            self.effective_io_data_type(input_tensor.data_type),
            "resample backward input gradient",
        )?;
        self.validate_tensor_data_type(
            output_gradient,
            output_tensor.data_type,
            "resample backward output gradient",
        )?;
        self.validate_tensor_dimensions(
            input_gradient,
            input_tensor.shape.dimensions(),
            "resample backward input gradient shape",
        )?;
        self.validate_tensor_dimensions(
            output_gradient,
            output_tensor.shape.dimensions(),
            "resample backward output gradient shape",
        )?;
        self.validate_resample_indices(
            indices,
            output_tensor.shape.dimensions(),
            config.mode(),
            "resample backward indices",
            "resample backward indices shape",
        )?;

        self.operations.push(Operation::ResampleBackward {
            input,
            output,
            output_gradient,
            input_gradient,
            indices,
            config,
        });
        Ok(())
    }

    pub fn resample_backward_infer(
        &mut self,
        input: TensorId,
        output: TensorId,
        output_gradient: TensorId,
        indices: Option<TensorId>,
        config: ResampleConfig,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let output_tensor = self.tensor_config(output)?.clone();
        self.validate_resample_indices(
            indices,
            output_tensor.shape.dimensions(),
            config.mode(),
            "resample backward indices",
            "resample backward indices shape",
        )?;
        let input_gradient_data_type = self.effective_io_data_type(input_tensor.data_type);
        let input_gradient = self.tensor(TensorSpec::new(
            input_gradient_data_type,
            input_tensor.shape.clone(),
        ));
        if let Err(error) = self.resample_backward(
            input,
            output,
            output_gradient,
            input_gradient,
            indices,
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(input_gradient)
    }

    fn validate_resample_indices(
        &self,
        indices: Option<TensorId>,
        output_dimensions: &[i64],
        mode: ResampleMode,
        mode_error: &str,
        shape_error: &str,
    ) -> Result<()> {
        if let Some(indices) = indices {
            if mode != ResampleMode::MaxPool {
                return Err(Error::DescriptorMismatch {
                    name: mode_error.into(),
                });
            }
            self.validate_tensor_data_type(indices, DataType::I8, shape_error)?;
            let indices_tensor = self.tensor_config(indices)?;
            if indices_tensor.shape.dimensions() != output_dimensions {
                return Err(Error::FrontendTensorDimensionsMismatch {
                    tensor_id: indices,
                    operation: shape_error.into(),
                    expected: output_dimensions.to_vec(),
                    actual: indices_tensor.shape.dimensions().to_vec(),
                });
            }
        }
        Ok(())
    }
}
