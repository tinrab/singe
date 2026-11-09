//! Gathers and scatters through index tensors, lowered to kernel IR accesses that check their index values.

use std::iter;

use crate::{
    backend::{
        cuda::kernel_plan::source::{KernelInput, KernelPlan, KernelPlanner},
        index_problem::{IndexBounds, indexing_problem},
        kernel_lowering::{self, Selection},
        registry::{Operation, OperationAttributes},
        spec::{MultiIndexProblem, multi_index_problem},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
};

/// Plans `gather`, `index_select`, or `embedding` of `inputs`, the input and its index tensor.
pub fn plan(
    planner: &KernelPlanner,
    operation: Operation,
    attributes: &OperationAttributes,
    inputs: &[KernelInput],
) -> Result<KernelPlan> {
    let [input, indices] = inputs else {
        return Err(Error::missing_input("indexing plan"));
    };
    if !matches!(indices.data_type, DataType::I32 | DataType::I64)
        || (operation != Operation::Take && indices.data_type != DataType::I64)
    {
        return Err(Error::unsupported_data_type("cuda indexing operands"));
    }

    let (problem, bounds) = indexing_problem(
        operation,
        attributes,
        input.layout.shape(),
        indices.layout.shape(),
    )?;
    let extent = input.layout.dimensions()[problem.axis];
    let selection = Selection {
        axis: problem.axis,
        extents: &[extent],
        index_shape: indices.layout.shape(),
        indices: &[(indices.data_type, &indices.layout)],
        wrap: matches!(bounds, IndexBounds::WrapNegative),
    };
    planner.lowered(
        kernel_lowering::gather(
            &format!("singe_{}", operation.name()),
            input.data_type,
            &input.layout,
            &selection,
        )?,
        inputs,
        &[operation],
    )
}

impl KernelPlanner {
    /// Plans `index`: the input elements its index tensors select along consecutive dimensions from `dim`.
    pub(crate) fn index(
        &self,
        input: &KernelInput,
        indices: &[KernelInput],
        dim: i64,
    ) -> Result<KernelPlan> {
        let problem = multi_index_plan(input, indices, dim)?;
        let layouts = layouts(indices);
        self.lowered(
            kernel_lowering::gather(
                "singe_index",
                input.data_type,
                &input.layout,
                &selection(&problem, &layouts),
            )?,
            iter::once(input).chain(indices),
            &[Operation::Index],
        )
    }

    /// Plans `index_put` into a contiguous copy of `input`, which the caller makes first: each element of `values` is written or added at the position its index tensors select.
    pub(crate) fn index_put(
        &self,
        input: &KernelInput,
        values: &KernelInput,
        indices: &[KernelInput],
        dim: i64,
        accumulate: bool,
    ) -> Result<KernelPlan> {
        let problem = multi_index_plan(input, indices, dim)?;
        if values.data_type != input.data_type {
            return Err(Error::data_type_mismatch(
                "cuda index put values",
                input.data_type,
                values.data_type,
            ));
        }

        let layouts = layouts(indices);
        self.lowered(
            kernel_lowering::scatter(
                "singe_index_put",
                input.value_spec(),
                &values.layout,
                &selection(&problem, &layouts),
                accumulate,
            )?,
            iter::once(values).chain(indices),
            &[Operation::IndexPut],
        )
    }
}

/// Checks the index operands and infers their problem.
fn multi_index_plan(
    input: &KernelInput,
    indices: &[KernelInput],
    dim: i64,
) -> Result<MultiIndexProblem> {
    if indices.is_empty() {
        return Err(Error::unsupported_operation(
            "cuda index without index tensors",
        ));
    }

    if indices.iter().any(|index| index.data_type != DataType::I64) {
        return Err(Error::unsupported_data_type(
            "cuda index tensors must be i64",
        ));
    }

    let shapes = indices
        .iter()
        .map(|index| index.layout.shape())
        .collect::<Vec<_>>();
    multi_index_problem(input.layout.shape(), &shapes, dim)
}

fn layouts(indices: &[KernelInput]) -> Vec<(DataType, &Layout)> {
    indices
        .iter()
        .map(|index| (index.data_type, &index.layout))
        .collect()
}

/// Returns the selection of `problem`, whose negative indices count from the end of their dimensions.
fn selection<'a>(
    problem: &'a MultiIndexProblem,
    indices: &'a [(DataType, &'a Layout)],
) -> Selection<'a> {
    Selection {
        axis: problem.axis,
        extents: &problem.indexed_dimensions,
        index_shape: &problem.index_shape,
        indices,
        wrap: true,
    }
}

#[cfg(test)]
mod tests;
