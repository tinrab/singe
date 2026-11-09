use crate::{
    backend::{
        cuda::{
            eager::reduction::reduction,
            kernel_plan::{self, KernelInput, KernelPlan},
        },
        decomposition,
        registry::{Operation, OperationAttributes, TopKAttributes},
    },
    error::Result,
    tensor::Tensor,
};

pub fn sum(a: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
    reduction(Operation::Sum, a, dimensions, keepdim)
}

pub fn mean(a: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
    reduction(Operation::Mean, a, dimensions, keepdim)
}

pub fn max(a: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
    reduction(Operation::Max, a, dimensions, keepdim)
}

pub fn min(a: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
    reduction(Operation::Min, a, dimensions, keepdim)
}

pub fn product(a: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
    reduction(Operation::Product, a, dimensions, keepdim)
}

pub fn logsumexp(a: &Tensor, dimensions: &[i64], keepdim: bool) -> Result<Tensor> {
    reduction(Operation::LogSumExp, a, dimensions, keepdim)
}

pub fn argmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    arg_reduction(Operation::ArgMax, input, dim)
}

pub fn argmin(input: &Tensor, dim: i64) -> Result<Tensor> {
    arg_reduction(Operation::ArgMin, input, dim)
}

fn arg_reduction(operation: Operation, input: &Tensor, dim: i64) -> Result<Tensor> {
    kernel_plan::execute(operation, &OperationAttributes::Dimension(dim), &[input])
}

pub fn argsort(input: &Tensor, dim: i64, descending: bool) -> Result<Tensor> {
    let input = input.contiguous()?;
    let plan = KernelPlan::argsort(&KernelInput::from_source(&input), dim, descending)?;
    kernel_plan::run(&plan, &[&input])
}

pub fn cumsum(input: &Tensor, dim: i64) -> Result<Tensor> {
    cumulative(input, dim, Operation::Cumsum)
}

pub fn cumprod(input: &Tensor, dim: i64) -> Result<Tensor> {
    cumulative(input, dim, Operation::Cumprod)
}

fn cumulative(input: &Tensor, dim: i64, operation: Operation) -> Result<Tensor> {
    kernel_plan::execute(operation, &OperationAttributes::Dimension(dim), &[input])
}

// `topk`, `topk_values`, or `topk_indices` as a selection where it picks few values from a contiguous row, and as its decomposition, a sort, otherwise.
pub fn topk(operation: Operation, input: &Tensor, topk: TopKAttributes) -> Result<Vec<Tensor>> {
    let Some(plan) = KernelPlan::topk_select(&KernelInput::from_source(input), &topk)? else {
        return decomposition::decompose(operation, &OperationAttributes::TopK(topk), &[input]);
    };
    let mut outputs = kernel_plan::run_outputs(&plan, &[input])?;
    Ok(match operation {
        Operation::TopKValues => vec![outputs.swap_remove(0)],
        Operation::TopKIndices => vec![outputs.swap_remove(1)],
        _ => outputs,
    })
}
