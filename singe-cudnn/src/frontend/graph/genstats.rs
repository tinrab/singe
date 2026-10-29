pub use crate::backend::execution::normalization::GenStatsOutputs;
use crate::{
    backend::tensor::{Shape, TensorId},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig, infer_normalization_channel_dimensions},
        operation::{GenStatsConfig, GenStatsOperation, Operation},
    },
};

impl Graph {
    /// Adds a generate-stats operation with explicit output tensors.
    ///
    /// `GenStats` computes per-channel sum and sum-of-squares tensors consumed by
    /// batch normalization finalize.
    /// Explicit output types and strides are passed through to cuDNN.
    /// Math precision follows the operation override, graph policy and then input encoding.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensor identities, input rank below two, or sum and square-sum dimensions that differ from the inferred per-channel shape.
    pub fn genstats_into(
        &mut self,
        input: TensorId,
        outputs: GenStatsOutputs,
        config: GenStatsConfig,
    ) -> Result<()> {
        let operation = GenStatsOperation::new(input, outputs, config);
        operation.validate_graph(self)?;
        self.operations.push(Operation::GenStats(operation));
        Ok(())
    }

    /// Adds a generate-stats operation and creates the sum and square-sum tensors.
    /// Each result selects placement and encoding independently, with an `F32` fallback.
    /// Virtual results inherit intermediate precision; bindable results inherit I/O precision.
    /// Result encodings are independent of `GenStatsConfig` math precision.
    /// A configured scope names results `name_sum_output` and `name_sq_sum_output`, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is unknown, its rank is below two, or inferred per-channel strides cannot be represented.
    pub fn genstats(
        &mut self,
        input: TensorId,
        outputs: GenStatsOutputs<TensorOutputConfig>,
        config: GenStatsConfig,
    ) -> Result<GenStatsOutputs> {
        let input_tensor = self.tensor_spec(input)?;
        let shape =
            Shape::channels_last(infer_normalization_channel_dimensions(&input_tensor.shape)?)?;
        let mut sum =
            outputs
                .sum
                .into_spec(self.config.data_type_policy, DataType::F32, shape.clone());
        let mut sq_sum =
            outputs
                .square_sum
                .into_spec(self.config.data_type_policy, DataType::F32, shape);
        sum.name = config.name().map(|name| format!("{name}_sum_output"));
        sq_sum.name = config.name().map(|name| format!("{name}_sq_sum_output"));
        let checkpoint = self.mutation_checkpoint();
        let sum = self.tensor(sum);
        let sq_sum = self.tensor(sq_sum);
        let outputs = GenStatsOutputs::new(sum, sq_sum);
        if let Err(error) = self.genstats_into(input, outputs, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }
}
