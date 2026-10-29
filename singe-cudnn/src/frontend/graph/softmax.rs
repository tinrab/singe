mod expand;

use expand::SoftmaxTensors;

pub use crate::backend::execution::attention::SoftmaxOutputs;
use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{Graph, TensorOutputConfig, infer_binary_pointwise_dimensions},
        operation::{AttentionPrimitiveOperation, Operation, PointwiseOperation, SoftmaxConfig},
    },
};

impl Default for SoftmaxOutputs<TensorOutputConfig> {
    fn default() -> Self {
        Self::new(TensorOutputConfig::new())
    }
}

impl Graph {
    pub(crate) fn validate_softmax_tensors(
        &self,
        input: TensorId,
        outputs: SoftmaxOutputs,
        sink: Option<TensorId>,
    ) -> Result<Vec<i64>> {
        let input_tensor = self.tensor_spec(input)?;
        self.validate_tensor_dimensions(
            outputs.output,
            input_tensor.shape.dimensions(),
            "softmax output",
        )?;
        let reduction_dimensions = reduce_last_axis_dimensions(input_tensor.shape.dimensions())?;
        for (tensor, name) in [
            (outputs.stats, "softmax stats"),
            (outputs.max, "softmax max"),
            (outputs.sum_exp, "softmax sum exp"),
        ] {
            if let Some(tensor) = tensor {
                self.validate_tensor_dimensions(tensor, &reduction_dimensions, name)?;
            }
        }
        if let Some(sink) = sink {
            let sink_dimensions = self.shape(sink)?.dimensions();
            let dimensions =
                infer_binary_pointwise_dimensions(&reduction_dimensions, sink_dimensions)?;
            if dimensions != reduction_dimensions {
                return Err(FrontendError::TensorDimensionsMismatch {
                    tensor_id: sink,
                    operation: "softmax sink broadcast",
                    expected: reduction_dimensions,
                    actual: sink_dimensions.to_vec(),
                }
                .into());
            }
        }
        Ok(reduction_dimensions)
    }

    /// Records the native softmax stage inside an attention composition.
    pub(crate) fn attention_softmax(
        &mut self,
        input: TensorId,
        outputs: SoftmaxOutputs,
        config: &SoftmaxConfig,
    ) -> Result<()> {
        let operation = AttentionPrimitiveOperation::Softmax {
            stats_base: config.stats_base(),
            x: input,
            y: outputs.output,
            stats: outputs.stats,
            max: outputs.max,
            sum_exp: outputs.sum_exp,
            sink: config.sink(),
        };
        operation.validate_graph(self)?;
        self.operations
            .push(Operation::AttentionPrimitive(operation));
        Ok(())
    }

    /// Adds standalone softmax with explicit probability and optional summary outputs.
    /// Tensor encodings and strides remain independent of each other and of graph compute precision.
    /// Expands to pointwise and reduction operations because the native softmax descriptor requires SDPA.
    /// Unrequested Max and `SumExp` use virtual intermediate storage.
    /// Summaries with another encoding are converted after the softmax calculation.
    ///
    /// # Errors
    ///
    /// Returns an error if input, output, summary, or sink identities are unknown, probability or summary dimensions are incompatible, sink logits cannot broadcast to row summaries, an intermediate layout is unrepresentable, or a component pointwise/reduction operation fails validation.
    pub fn softmax_into(
        &mut self,
        input: TensorId,
        outputs: SoftmaxOutputs,
        config: &SoftmaxConfig,
    ) -> Result<()> {
        let reduction_dimensions = self.validate_softmax_tensors(input, outputs, config.sink())?;
        let intermediate_type = self.effective_intermediate_data_type(
            self.resolve_compute_data_type(config.compute_type()),
        );
        let intermediate = |output: Option<TensorId>| -> Result<Option<TensorId>> {
            match output {
                Some(output) if self.tensor_spec(output)?.data_type == intermediate_type => {
                    Ok(Some(output))
                }
                _ => Ok(None),
            }
        };
        let max = intermediate(outputs.max)?;
        let sum = intermediate(outputs.sum_exp)?;
        let checkpoint = self.mutation_checkpoint();
        let (max, sum) = match (max, sum) {
            (Some(max), Some(sum)) => (max, sum),
            (max, sum) => {
                let summary =
                    TensorSpec::new(intermediate_type, Shape::contiguous(reduction_dimensions)?)
                        .virtual_tensor();
                let max = max.unwrap_or_else(|| {
                    self.inferred_tensor(summary.clone(), config.name(), "Max_intermediate")
                });
                let sum = sum.unwrap_or_else(|| {
                    self.inferred_tensor(summary, config.name(), "Sum_exp_intermediate")
                });
                (max, sum)
            }
        };
        let tensors = SoftmaxTensors {
            input,
            output: outputs.output,
            stats: outputs.stats,
            max,
            sum_exp: sum,
        };
        if let Err(error) = self.expand_softmax(tensors, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        for (input, output) in [(max, outputs.max), (sum, outputs.sum_exp)] {
            if let Some(output) = output.filter(|&output| output != input) {
                self.pointwise_into(PointwiseOperation::Unary {
                    name: None,
                    mode: PointwiseMode::Identity,
                    input,
                    output,
                    compute_type: config.compute_type(),
                    nan_propagation: config.nan_propagation(),
                    alpha1: 1.0,
                    axis: None,
                })?;
            }
        }
        Ok(())
    }

    /// Adds standalone softmax and infers the requested outputs.
    /// Probabilities retain the input's dimensions and strides.
    /// Each summary has a unit last dimension and contiguous storage.
    /// Every result selects its placement and encoding independently through its output configuration.
    /// Unset policies fall back to input encoding for probabilities and graph compute precision for summaries.
    /// A configured name scopes probabilities as `name::S` and summaries as `name::Stats`, `name::Max` and `name::Sum_exp`.
    /// Omitting Stats skips its log-sum-exp operations.
    /// Unrequested Max and `SumExp` remain private virtual intermediates.
    ///
    /// # Errors
    /// Returns an error for unknown operands or invalid geometry.
    /// Sink dimensions must broadcast to the row summaries without enlarging them.
    /// A rejected request leaves no inferred tensors or operations in the graph.
    pub fn softmax(
        &mut self,
        input: TensorId,
        outputs: SoftmaxOutputs<TensorOutputConfig>,
        config: &SoftmaxConfig,
    ) -> Result<SoftmaxOutputs> {
        self.infer_softmax_with(input, outputs, config, Self::softmax_into)
    }

    pub(crate) fn attention_softmax_infer(
        &mut self,
        input: TensorId,
        outputs: SoftmaxOutputs<TensorOutputConfig>,
        config: &SoftmaxConfig,
    ) -> Result<SoftmaxOutputs> {
        self.infer_softmax_with(input, outputs, config, Self::attention_softmax)
    }

    fn infer_softmax_with(
        &mut self,
        input: TensorId,
        outputs: SoftmaxOutputs<TensorOutputConfig>,
        config: &SoftmaxConfig,
        record: fn(&mut Self, TensorId, SoftmaxOutputs, &SoftmaxConfig) -> Result<()>,
    ) -> Result<SoftmaxOutputs> {
        let input_tensor = self.tensor_spec(input)?;
        let reduction_shape = reduce_last_axis(&input_tensor.shape)?;
        if let Some(sink) = config.sink() {
            self.tensor_spec(sink)?;
        }
        let output = outputs.output.into_spec(
            self.config.data_type_policy,
            input_tensor.data_type,
            input_tensor.shape.clone(),
        );
        let summary_type = self.resolve_compute_data_type(config.compute_type());
        let summaries = [outputs.stats, outputs.max, outputs.sum_exp].map(|request| {
            request.map(|request| {
                request.into_spec(
                    self.config.data_type_policy,
                    summary_type,
                    reduction_shape.clone(),
                )
            })
        });
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "S");
        let [stats, max, sum_exp] = summaries;
        let outputs = SoftmaxOutputs {
            output,
            stats: stats.map(|spec| self.inferred_tensor(spec, config.name(), "Stats")),
            max: max.map(|spec| self.inferred_tensor(spec, config.name(), "Max")),
            sum_exp: sum_exp.map(|spec| self.inferred_tensor(spec, config.name(), "Sum_exp")),
        };
        if let Err(error) = record(self, input, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}

pub fn reduce_last_axis_dimensions(input: &[i64]) -> Result<Vec<i64>> {
    let mut dimensions = input.to_vec();

    let Some(last_dimension) = dimensions.last_mut() else {
        return Err(Error::InvalidDataShape);
    };
    *last_dimension = 1;

    Ok(dimensions)
}

pub fn reduce_last_axis(shape: &Shape) -> Result<Shape> {
    Shape::contiguous(reduce_last_axis_dimensions(shape.dimensions())?)
}
