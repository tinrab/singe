use crate::{
    error::Result,
    frontend::{graph::Graph, operation::Operation},
    tensor::{TensorId, TensorSpec},
};

impl Graph {
    pub fn reshape(&mut self, input: TensorId, output: TensorId) -> Result<()> {
        let input_tensor = self.tensor_config(input)?.clone();
        self.validate_tensor_data_type(output, input_tensor.data_type, "reshape output")?;
        self.validate_tensor_element_count(
            output,
            input_tensor.shape.element_count()?,
            "reshape output",
        )?;

        self.operations.push(Operation::Reshape { input, output });
        Ok(())
    }

    pub fn reshape_infer(
        &mut self,
        input: TensorId,
        output_dimensions: impl Into<Vec<i64>>,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let input_tensor = self.tensor_config(input)?.clone();
        let output_data_type = input_tensor.data_type;
        let output_shape = Self::default_nhwc_shape(output_dimensions.into())?;
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        if let Err(error) = self.reshape(input, output) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }
}
