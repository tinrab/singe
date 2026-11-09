use crate::{
    backend::{
        cuda::kernel_plan,
        registry::{Operation, OperationAttributes},
    },
    error::Result,
    tensor::Tensor,
};

pub fn softmax(a: &Tensor, dim: i64) -> Result<Tensor> {
    planned_softmax(Operation::Softmax, a, dim)
}

pub fn log_softmax(a: &Tensor, dim: i64) -> Result<Tensor> {
    planned_softmax(Operation::LogSoftmax, a, dim)
}

fn planned_softmax(operation: Operation, input: &Tensor, dim: i64) -> Result<Tensor> {
    kernel_plan::execute(operation, &OperationAttributes::Dimension(dim), &[input])
}
