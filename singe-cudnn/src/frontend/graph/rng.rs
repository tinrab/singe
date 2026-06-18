use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::Graph,
        operation::{Operation, RandomNumberGeneratorConfig},
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
            self.validate_tensor_data_type(seed, DataType::I64, "rng seed")?;
            self.validate_random_number_generator_scalar_tensor(seed, "rng seed")?;
            if config.offset().is_none() {
                return Err(Error::DescriptorMismatch {
                    name: "rng offset".into(),
                });
            }
        }
        if config.offset().is_some() && config.seed_tensor().is_none() && config.seed().is_none() {
            return Err(Error::DescriptorMismatch {
                name: "rng offset".into(),
            });
        }
        if let Some(offset) = config.offset() {
            self.validate_tensor_data_type(offset, DataType::I64, "rng offset")?;
            self.validate_random_number_generator_scalar_tensor(offset, "rng offset")?;
        }

        self.operations
            .push(Operation::RandomNumberGenerator { output, config });
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
        let shape = Self::default_nhwc_shape(dimensions.into())?;
        self.random_number_generator_infer(shape, config)
    }

    fn validate_random_number_generator_scalar_tensor(
        &self,
        tensor: TensorId,
        operation: &str,
    ) -> Result<()> {
        let shape = &self.tensor_config(tensor)?.shape;
        if !shape.dimensions().iter().all(|dimension| *dimension == 1) {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected: vec![1; shape.dimensions().len()],
                actual: shape.dimensions().to_vec(),
            });
        }
        if !shape.strides().iter().all(|stride| *stride == 1) {
            return Err(Error::FrontendTensorStridesMismatch {
                tensor_id: tensor,
                operation: operation.into(),
                expected: vec![1; shape.strides().len()],
                actual: shape.strides().to_vec(),
            });
        }
        Ok(())
    }
}
