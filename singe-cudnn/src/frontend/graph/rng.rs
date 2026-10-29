use crate::{
    backend::tensor::{Shape, TensorId},
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{Operation, RandomNumberGeneratorConfig, RandomNumberGeneratorOperation},
    },
};

impl Graph {
    /// Records random-number generation into an explicitly declared tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if an output, seed, or offset identity is unknown, distribution parameters are invalid, or a seed/offset tensor is not an INT32 or INT64 scalar with the required unit strides.
    pub fn random_number_generator_into(
        &mut self,
        output: TensorId,
        config: RandomNumberGeneratorConfig,
    ) -> Result<()> {
        let operation = RandomNumberGeneratorOperation::new(output, config);
        operation.validate_graph(self)?;
        self.operations
            .push(Operation::RandomNumberGenerator(operation));
        Ok(())
    }

    /// Creates a random tensor with the requested geometry, placement and encoding.
    /// Omitted result encoding inherits I/O or intermediate policy according to placement.
    /// A configured scope names the result `name::Y`; an empty scope leaves it unnamed.
    ///
    /// # Errors
    ///
    /// Returns an error if an output, seed, or offset identity is unknown, distribution parameters are invalid, or a seed/offset tensor is not an INT32 or INT64 scalar with the required unit strides.
    pub fn random_number_generator(
        &mut self,
        shape: Shape,
        config: RandomNumberGeneratorConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let output = output.into_spec(self.config.data_type_policy, DataType::F32, shape);
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "Y");
        if let Err(error) = self.random_number_generator_into(output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }
}
