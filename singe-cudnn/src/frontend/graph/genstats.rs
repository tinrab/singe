use crate::{
    data_type::DataType,
    error::Result,
    frontend::{
        graph::Graph,
        infer::infer_batch_norm_channel_shape,
        operation::{GenStatsConfig, GenStatsOperation, Operation},
    },
    tensor::{TensorId, TensorSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenStatsOutputs {
    pub sum: TensorId,
    pub square_sum: TensorId,
}

impl GenStatsOutputs {
    pub fn new(sum: TensorId, square_sum: TensorId) -> Self {
        Self { sum, square_sum }
    }
}

impl Graph {
    /// Adds a generate-stats operation with explicit output tensors.
    ///
    /// GenStats computes per-channel sum and sum-of-squares tensors consumed by
    /// batch normalization finalize.
    pub fn genstats(
        &mut self,
        input: TensorId,
        sum: TensorId,
        sq_sum: TensorId,
        config: GenStatsConfig,
    ) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        let expected = infer_batch_norm_channel_shape(&input_tensor.shape)?;
        let expected_data_type = self.effective_io_data_type(DataType::F32);
        self.validate_tensors_data_type(
            [(sum, "genstats sum"), (sq_sum, "genstats sq_sum")],
            expected_data_type,
        )?;
        self.validate_tensors_dimensions(
            [
                (sum, "genstats sum shape"),
                (sq_sum, "genstats sq_sum shape"),
            ],
            expected.dimensions(),
        )?;
        self.operations
            .push(Operation::GenStats(GenStatsOperation::new(
                input, sum, sq_sum, config,
            )));
        Ok(())
    }

    /// Adds a generate-stats operation and creates the sum and square-sum tensors.
    pub fn genstats_infer(
        &mut self,
        input: TensorId,
        config: GenStatsConfig,
    ) -> Result<GenStatsOutputs> {
        let input_tensor = self.tensor_config(input)?.clone();
        let shape = infer_batch_norm_channel_shape(&input_tensor.shape)?;
        let output_data_type = self.effective_io_data_type(DataType::F32);
        let sum = self.tensor(TensorSpec::new(output_data_type, shape.clone()));
        let sq_sum = self.tensor(TensorSpec::new(output_data_type, shape));
        self.genstats(input, sum, sq_sum, config)?;
        Ok(GenStatsOutputs::new(sum, sq_sum))
    }
}
