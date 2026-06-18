#[cfg(test)]
use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::graph::Graph,
    tensor::TensorId,
};

#[cfg(test)]
impl Graph {
    #[cfg(test)]
    pub(super) fn validate_sdpa_sequence_lengths(
        &self,
        sequence_length_query: TensorId,
        sequence_length_key_value: TensorId,
        batch_size: i64,
        error_name: &str,
    ) -> Result<()> {
        self.validate_tensor_data_type(sequence_length_query, DataType::I32, error_name)?;
        self.validate_tensor_data_type(sequence_length_key_value, DataType::I32, error_name)?;

        let expected_shape = [batch_size, 1, 1, 1];
        self.validate_tensor_dimensions(
            sequence_length_query,
            &expected_shape,
            "sdpa seq len q shape",
        )?;
        self.validate_tensor_dimensions(
            sequence_length_key_value,
            &expected_shape,
            "sdpa seq len kv shape",
        )?;

        Ok(())
    }

    #[cfg(test)]
    pub(super) fn validate_sdpa_layout_last_stride_one(
        &self,
        tensor_ref: TensorId,
        name: &str,
    ) -> Result<()> {
        let tensor = self.tensor_config(tensor_ref)?;
        self.validate_tensor_rank(tensor_ref, 4, "sdpa rank")?;
        if tensor.shape.strides()[3] != 1 {
            return Err(Error::DescriptorMismatch { name: name.into() });
        }

        Ok(())
    }
}
