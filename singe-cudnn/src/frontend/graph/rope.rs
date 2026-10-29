use crate::{
    backend::tensor::TensorId,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{Operation, RopeBackwardInputs, RopeConfig, RopeInputs, RopeOperation},
    },
};

impl Graph {
    /// Records native rotary position embedding with an explicit output tensor.
    /// Frequencies contain raw angles, not precomputed sine and cosine values.
    ///
    /// # Errors
    ///
    /// Returns an error if an activation, gradient, frequency, or destination tensor identity is unknown, the rotation dimension is negative, or destination dimensions differ from the activation or output gradient.
    pub fn rope_forward_into(
        &mut self,
        inputs: RopeInputs<TensorId>,
        output: TensorId,
        config: RopeConfig,
    ) -> Result<()> {
        let operation = RopeOperation::Forward {
            inputs,
            output,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Rope(operation));
        Ok(())
    }

    /// Creates a rotated output with the input dimensions and strides.
    /// Select a bindable output to save rotated values for a later attention backward graph.
    /// Omitted encoding follows I/O or intermediate policy according to the requested placement.
    /// A configured scope names the result `name::OUTPUT`; an empty scope leaves it unnamed.
    ///
    /// # Errors
    ///
    /// Returns an error if an activation, gradient, frequency, or destination tensor identity is unknown, the rotation dimension is negative, or destination dimensions differ from the activation or output gradient.
    pub fn rope_forward(
        &mut self,
        inputs: RopeInputs<TensorId>,
        config: RopeConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let input = self.tensor_spec(inputs.input)?;
        let output = output.into_spec(
            self.config.data_type_policy,
            input.data_type,
            input.shape.clone(),
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "OUTPUT");
        if let Err(error) = self.rope_forward_into(inputs, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Records the transpose rotation using the forward operation's scale and frequencies.
    /// The operation produces the input gradient and no frequency gradient.
    ///
    /// # Errors
    ///
    /// Returns an error if an activation, gradient, frequency, or destination tensor identity is unknown, the rotation dimension is negative, or destination dimensions differ from the activation or output gradient.
    pub fn rope_backward_into(
        &mut self,
        inputs: RopeBackwardInputs<TensorId>,
        input_gradient: TensorId,
        config: RopeConfig,
    ) -> Result<()> {
        let operation = RopeOperation::Backward {
            inputs,
            input_gradient,
            config,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Rope(operation));
        Ok(())
    }

    /// Creates an input gradient with the output gradient's dimensions and strides.
    /// Use the same rotation dimension and output scale as in the corresponding forward operation.
    /// Requested placement selects I/O or intermediate precision unless the output encoding is explicit.
    /// A configured scope names the result `name::DX`; an empty scope leaves it unnamed.
    ///
    /// # Errors
    ///
    /// Returns an error if an activation, gradient, frequency, or destination tensor identity is unknown, the rotation dimension is negative, or destination dimensions differ from the activation or output gradient.
    pub fn rope_backward(
        &mut self,
        inputs: RopeBackwardInputs<TensorId>,
        config: RopeConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let gradient = self.tensor_spec(inputs.output_gradient)?;
        let input_gradient = output.into_spec(
            self.config.data_type_policy,
            gradient.data_type,
            gradient.shape.clone(),
        );
        let checkpoint = self.mutation_checkpoint();
        let input_gradient = self.inferred_tensor(input_gradient, config.name(), "DX");
        if let Err(error) = self.rope_backward_into(inputs, input_gradient, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(input_gradient)
    }
}
