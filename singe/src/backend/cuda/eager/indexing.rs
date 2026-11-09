use std::iter::once;

use crate::{
    backend::{
        cuda::{
            kernel_plan::{self, KernelInput, KernelPlanner},
            utility::ensure_same_device,
        },
        registry::{Operation, OperationAttributes},
    },
    error::Result,
    tensor::Tensor,
};

pub fn embedding(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    indexing(
        Operation::Embedding,
        &OperationAttributes::None,
        weight,
        indices,
    )
}

pub fn indexing(
    operation: Operation,
    attributes: &OperationAttributes,
    input: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    ensure_same_device(input, indices)?;
    // The kernel checks every index on the device and reports out-of-range ones.
    kernel_plan::execute(operation, attributes, &[input, indices])
}

/// Selects elements of the consecutive dimensions starting at `dim` with broadcast `I64` index tensors.
pub fn index(input: &Tensor, indices: &[Tensor], dim: i64) -> Result<Tensor> {
    let metadata = indices
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let plan = KernelPlanner::default().index(&KernelInput::from_source(input), &metadata, dim)?;
    let operands = once(input).chain(indices).collect::<Vec<_>>();
    kernel_plan::run(&plan, &operands)
}

/// Returns a copy of `input` with `values` written into, or with `accumulate` added to, the elements [`index`] selects.
pub fn index_put(
    input: &Tensor,
    values: &Tensor,
    indices: &[Tensor],
    dim: i64,
    accumulate: bool,
) -> Result<Tensor> {
    let metadata = indices
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let plan = KernelPlanner::default().index_put(
        &KernelInput::from_source(input),
        &KernelInput::from_source(values),
        &metadata,
        dim,
        accumulate,
    )?;
    let output = kernel_plan::execute(Operation::Copy, &OperationAttributes::None, &[input])?;
    let operands = once(values).chain(indices).collect::<Vec<_>>();
    kernel_plan::run_into(&plan, &operands, &output)?;
    Ok(output)
}

// Writes `values` into, or with `accumulate` adds them to, the elements [`index`] selects of `destination`, where they are when it is contiguous, else through a copy it is then assigned.
pub fn index_write(
    destination: &Tensor,
    values: &Tensor,
    indices: &[Tensor],
    dim: i64,
    accumulate: bool,
) -> Result<()> {
    if !destination.is_contiguous() {
        let updated = index_put(destination, values, indices, dim, accumulate)?;
        return kernel_plan::assign(destination, &updated);
    }

    let metadata = indices
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let plan = KernelPlanner::default().index_put(
        &KernelInput::from_source(destination),
        &KernelInput::from_source(values),
        &metadata,
        dim,
        accumulate,
    )?;
    let operands = once(values).chain(indices).collect::<Vec<_>>();
    kernel_plan::run_into(&plan, &operands, destination)
}
