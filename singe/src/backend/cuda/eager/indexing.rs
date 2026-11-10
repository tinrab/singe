use std::{iter::once, sync::Arc};

use crate::{
    backend::{
        cuda::{
            kernel_plan::{self, KernelPlan, PlanKind},
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
    let operands = once(input).chain(indices).collect::<Vec<_>>();
    let plans = kernel_plan::cached_plans(
        PlanKind::Index,
        &OperationAttributes::None,
        &[dim],
        &operands,
        |planner, inputs| Ok(vec![planner.index(&inputs[0], &inputs[1..], dim)?]),
    )?;
    kernel_plan::run(&plans[0], &operands)
}

// The plan that writes `values` into, or with `accumulate` adds them to, the elements of `destination` that `indices` select.
fn index_put_plan(
    destination: &Tensor,
    values: &Tensor,
    indices: &[Tensor],
    dim: i64,
    accumulate: bool,
) -> Result<Arc<[KernelPlan]>> {
    let planned = [destination, values]
        .into_iter()
        .chain(indices)
        .collect::<Vec<_>>();
    kernel_plan::cached_plans(
        PlanKind::IndexPut,
        &OperationAttributes::None,
        &[dim, i64::from(accumulate)],
        &planned,
        |planner, inputs| {
            Ok(vec![planner.index_put(
                &inputs[0],
                &inputs[1],
                &inputs[2..],
                dim,
                accumulate,
            )?])
        },
    )
}

/// Returns a copy of `input` with `values` written into, or with `accumulate` added to, the elements [`index`] selects.
pub fn index_put(
    input: &Tensor,
    values: &Tensor,
    indices: &[Tensor],
    dim: i64,
    accumulate: bool,
) -> Result<Tensor> {
    let plans = index_put_plan(input, values, indices, dim, accumulate)?;
    let output = kernel_plan::execute(Operation::Copy, &OperationAttributes::None, &[input])?;
    let operands = once(values).chain(indices).collect::<Vec<_>>();
    kernel_plan::run_into(&plans[0], &operands, &output)?;
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

    let plans = index_put_plan(destination, values, indices, dim, accumulate)?;
    let operands = once(values).chain(indices).collect::<Vec<_>>();
    kernel_plan::run_into(&plans[0], &operands, destination)
}
