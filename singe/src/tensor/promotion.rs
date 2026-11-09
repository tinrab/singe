use crate::{
    backend::registry::{Operation, operation_schema},
    data_type::DataType,
    error::{Error, Result},
    tensor::Tensor,
};

impl Tensor {
    pub(crate) fn binary_operands(
        &self,
        other: &Self,
        operation: Operation,
    ) -> Result<(Self, Self)> {
        if self.device() != other.device() {
            return Err(Error::device_mismatch(
                "binary operands",
                self.device(),
                other.device(),
            ));
        }

        let Some(request) =
            operation_schema(operation).binary_request(self.value_spec(), other.value_spec())
        else {
            return Ok((self.clone(), other.clone()));
        };
        let data_type = request.infer()?.computation_data_type();
        Ok((
            self.promoted_operand(data_type)?,
            other.promoted_operand(data_type)?,
        ))
    }

    fn promoted_operand(&self, data_type: DataType) -> Result<Self> {
        if self.data_type() == data_type {
            Ok(self.clone())
        } else {
            self.cast(data_type)
        }
    }
}
