use crate::{
    backend::tensor::{Shape, TensorId},
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{Operation, ReshapeConfig, TensorOperation},
    },
};

impl Graph {
    /// Records a reshape while preserving the caller's output geometry and name.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensors, invalid vector metadata, unequal logical element counts, or incompatible encodings in logical mode.
    pub fn reshape_into(
        &mut self,
        input: TensorId,
        output: TensorId,
        config: ReshapeConfig,
    ) -> Result<()> {
        let operation = TensorOperation::Reshape {
            input,
            output,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Tensor(operation));
        Ok(())
    }

    /// Creates a reshape result with the requested dimensions and native mode.
    /// An empty dimension list denotes a scalar and is promoted to `[1]`, matching C++.
    /// Multi-axis outputs use NHWC stride order, while scalar and vector outputs are contiguous.
    /// Placement selects I/O or intermediate encoding defaults; explicit encodings take precedence.
    /// Logical reshape requires the input encoding; view-only reshape permits a different result encoding.
    /// A configured scope names the inferred result `name::Y`.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensors, invalid vector metadata, unequal logical element counts, or incompatible encodings in logical mode. Inferred output dimensions must be positive and their channels-last strides must be representable.
    pub fn reshape(
        &mut self,
        input: TensorId,
        output_dimensions: impl Into<Vec<i64>>,
        config: ReshapeConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_spec(input)?;
        let mut output_dimensions = output_dimensions.into();
        if output_dimensions.is_empty() {
            output_dimensions.push(1);
        }
        let output_shape = Shape::channels_last(output_dimensions)?;
        let output_spec = output.into_spec(
            self.config.data_type_policy,
            input_tensor.data_type,
            output_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output_spec, config.name(), "Y");
        if let Err(error) = self.reshape_into(input, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }
}
