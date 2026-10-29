pub use crate::backend::execution::resample::ResampleOutputs;
use crate::{
    backend::{
        execution::resample::ResampleMode,
        tensor::{Shape, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig},
        operation::{Operation, ResampleBackwardInputs, ResampleConfig, ResampleOperation},
    },
};

/// Requested saved-index result for forward resampling inference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ResampleIndexOutput {
    /// Do not produce saved indices.
    #[default]
    None,
    /// Preserve a declared index tensor's geometry, layout, encoding and placement.
    Existing(TensorId),
    /// Create indices with inferred NHWC geometry and the requested placement and encoding.
    /// An omitted encoding defaults to INT8 independently of graph precision policy.
    Inferred(TensorOutputConfig),
}

impl Graph {
    /// Adds a resampling forward operation with an explicit output tensor.
    ///
    /// Resampling changes spatial dimensions using pooling or interpolation.
    /// Max pooling and nearest-neighbor resampling may also produce indices.
    /// Explicit output geometry, types and layouts are passed through to cuDNN.
    /// Native finalization and support selection decide their compatibility with the spatial attributes.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, incompatible input/output ranks or batch/channel dimensions, invalid spatial geometry, unsupported saved-index mode or mismatched index dimensions, or virtual destination placement with nonzero beta.
    pub fn resample_into(
        &mut self,
        input: TensorId,
        outputs: ResampleOutputs,
        config: ResampleConfig,
    ) -> Result<()> {
        let operation = ResampleOperation::Forward {
            input,
            outputs,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Resample(operation));
        Ok(())
    }

    /// Infers a resampling output with NHWC strides and independent placement and encoding.
    /// Indices may be omitted, supplied as an existing tensor or inferred with their own output configuration.
    /// Inferred indices default to INT8 independently of floating-point graph policy.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, incompatible input/output ranks or batch/channel dimensions, invalid spatial geometry, unsupported saved-index mode or mismatched index dimensions, or virtual destination placement with nonzero beta. Inferred output dimensions and strides must also be representable.
    pub fn resample(
        &mut self,
        input: TensorId,
        config: ResampleConfig,
        output: TensorOutputConfig,
        indices: ResampleIndexOutput,
    ) -> Result<ResampleOutputs> {
        output.validate_accumulation(config.beta() != 0.0, "resample")?;
        self.validate_tensor_min_rank(input, 3, "resample rank")?;
        let input_tensor = self.tensor_spec(input)?;
        let geometry = config.resolve_geometry(input_tensor.shape.rank() - 2)?;
        let output_dimensions =
            geometry.infer_forward_dimensions(input_tensor.shape.dimensions())?;
        let output_shape = Shape::channels_last(output_dimensions)?;
        let input_type = input_tensor.data_type;
        let checkpoint = self.mutation_checkpoint();
        let indices = match indices {
            ResampleIndexOutput::None => None,
            ResampleIndexOutput::Existing(tensor) => Some(tensor),
            ResampleIndexOutput::Inferred(output) => {
                let output = output
                    .with_data_type(output.data_type().unwrap_or(DataType::I8))
                    .into_spec(
                        self.config.data_type_policy,
                        DataType::I8,
                        output_shape.clone(),
                    );
                Some(self.inferred_tensor(output, config.name(), "Index"))
            }
        };
        let output = output.into_spec(self.config.data_type_policy, input_type, output_shape);
        let output = self.inferred_tensor(output, config.name(), "Y");
        let outputs = ResampleOutputs::new(output, indices);
        if let Err(error) = self.resample_into(input, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    /// Adds backward resampling with an explicit input gradient.
    /// Saved forward tensors are optional where cuDNN permits their omission.
    /// Explicit gradient geometry reaches native finalization without recomputing an inferred output.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, incompatible gradient ranks or batch/channel dimensions, invalid spatial geometry, mismatched saved forward tensor or index dimensions, unsupported saved-index mode, or virtual gradient placement with nonzero beta.
    pub fn resample_backward_into(
        &mut self,
        inputs: ResampleBackwardInputs<TensorId>,
        input_gradient: TensorId,
        config: ResampleConfig,
    ) -> Result<()> {
        let operation = ResampleOperation::Backward {
            inputs,
            input_gradient,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Resample(operation));
        Ok(())
    }

    /// Creates an input gradient with the requested dimensions and strides.
    /// Forward geometry cannot uniquely determine the original input shape.
    /// Placement selects I/O or intermediate precision, with the output-gradient type as fallback.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, incompatible gradient ranks or batch/channel dimensions, invalid spatial geometry, mismatched saved forward tensor or index dimensions, unsupported saved-index mode, or virtual gradient placement with nonzero beta.
    pub fn resample_backward(
        &mut self,
        inputs: ResampleBackwardInputs<TensorId>,
        input_shape: Shape,
        config: ResampleConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        output.validate_accumulation(config.beta() != 0.0, "resample backward")?;
        let checkpoint = self.mutation_checkpoint();
        let output_gradient = self.tensor_spec(inputs.output_gradient)?;
        let input_gradient = output.into_spec(
            self.config.data_type_policy,
            output_gradient.data_type,
            input_shape,
        );
        let input_gradient = self.inferred_tensor(input_gradient, config.name(), "DX");
        if let Err(error) = self.resample_backward_into(inputs, input_gradient, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(input_gradient)
    }

    pub(crate) fn validate_resample_geometry(
        &self,
        input: TensorId,
        output: TensorId,
        config: &ResampleConfig,
    ) -> Result<()> {
        self.validate_tensor_min_rank(input, 3, "resample input rank")?;
        let input_dimensions = self.tensor_spec(input)?.shape.dimensions();
        self.validate_tensor_rank(output, input_dimensions.len(), "resample output rank")?;
        let output_dimensions = self.tensor_spec(output)?.shape.dimensions();
        if input_dimensions[..2] != output_dimensions[..2] {
            return Err(Error::ShapeMismatch {
                name: "resample batch and channel dimensions".into(),
                expected: input_dimensions[..2].to_vec(),
                actual: output_dimensions[..2].to_vec(),
            });
        }
        config.resolve_geometry(input_dimensions.len() - 2)?;
        Ok(())
    }

    pub(crate) fn validate_resample_indices(
        &self,
        indices: Option<TensorId>,
        output_dimensions: &[i64],
        mode: ResampleMode,
        shape_error: &'static str,
    ) -> Result<()> {
        if let Some(indices) = indices {
            if !matches!(mode, ResampleMode::MaxPool | ResampleMode::Nearest) {
                return Err(FrontendError::ResampleIndicesUnsupportedMode.into());
            }
            self.validate_tensor_dimensions(indices, output_dimensions, shape_error)?;
        }
        Ok(())
    }
}
