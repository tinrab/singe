use super::*;

pub fn cpu_tensor<T: TensorDataType>(shape: &Shape, values: Vec<T>) -> Tensor {
    Tensor::from_vec(Device::Cpu, shape, values).unwrap()
}

pub fn graph_operation_output(output: &Tensor) -> GraphOperationOutput {
    GraphOperationOutput::new(output.data_type(), output.shape().clone(), output.device())
}
