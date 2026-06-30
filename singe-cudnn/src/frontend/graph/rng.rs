use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        operation::{Operation, RandomNumberGeneratorConfig, RandomNumberGeneratorOperation},
        shape::shape_with_nhwc_strides,
        support,
    },
    tensor::{Shape, TensorId, TensorSpec},
};

impl Graph {
    pub fn random_number_generator(
        &mut self,
        output: TensorId,
        config: RandomNumberGeneratorConfig,
    ) -> Result<()> {
        self.validate_tensor_data_type_supported(
            output,
            &[DataType::F16, DataType::BF16, DataType::F32],
            "rng output",
        )?;
        if let Some(seed) = config.seed_tensor() {
            self.validate_random_number_generator_scalar_tensor(
                seed,
                DataType::I64,
                support::RNG_SEED,
            )?;
            if config.offset().is_none() {
                return Err(Error::FrontendRandomNumberGeneratorSeedOffsetMismatch);
            }
        }
        if config.offset().is_some() && config.seed_tensor().is_none() && config.seed().is_none() {
            return Err(Error::FrontendRandomNumberGeneratorSeedOffsetMismatch);
        }
        if let Some(offset) = config.offset() {
            self.validate_random_number_generator_scalar_tensor(
                offset,
                DataType::I64,
                support::RNG_OFFSET,
            )?;
        }

        self.operations.push(Operation::RandomNumberGenerator(
            RandomNumberGeneratorOperation::new(output, config),
        ));
        Ok(())
    }

    pub fn random_number_generator_infer(
        &mut self,
        shape: Shape,
        config: RandomNumberGeneratorConfig,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let output_data_type = self.effective_io_data_type(DataType::F32);
        let output = self.tensor(TensorSpec::new(output_data_type, shape));
        if let Err(error) = self.random_number_generator(output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub fn random_number_generator_infer_with_dimensions(
        &mut self,
        dimensions: impl Into<Vec<i64>>,
        config: RandomNumberGeneratorConfig,
    ) -> Result<TensorId> {
        let shape = shape_with_nhwc_strides(dimensions)?;
        self.random_number_generator_infer(shape, config)
    }

    fn validate_random_number_generator_scalar_tensor(
        &self,
        tensor: TensorId,
        data_type: DataType,
        operation: &str,
    ) -> Result<()> {
        self.validate_tensor_data_type(tensor, data_type, operation)?;

        let shape = &self.tensor_config(tensor)?.shape;
        let expected_dimensions = vec![1; shape.dimensions().len()];
        self.validate_tensor_dimensions(tensor, &expected_dimensions, operation)?;

        let expected_strides = vec![1; shape.strides().len()];
        self.validate_tensor_strides(tensor, &expected_strides, operation)
    }
}
