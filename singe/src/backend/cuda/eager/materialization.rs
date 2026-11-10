use crate::{
    backend::{
        cuda::{
            eager::shape::noop_contiguous,
            kernel_plan::{self, KernelInput, PlanKind},
            utility::ensure_same_data_type,
        },
        registry::{Operation, OperationAttributes},
        spec,
    },
    error::{Error, Result},
    layout::Layout,
    tensor::Tensor,
};

pub fn contiguous(input: &Tensor) -> Result<Tensor> {
    if let Some(output) = noop_contiguous(input) {
        return Ok(output);
    }
    kernel_plan::execute(Operation::Contiguous, &OperationAttributes::None, &[input])
}

pub fn concat(tensors: &[Tensor], dim: i64) -> Result<Tensor> {
    let first = tensors.first().ok_or(Error::MissingConcatInput)?;
    for tensor in tensors {
        ensure_same_data_type(first, tensor)?;
    }

    let shapes = tensors.iter().map(Tensor::shape).collect::<Vec<_>>();
    let problem = spec::concat_problem(&shapes, dim, "cuda concat")?;
    let output = KernelInput::new(first.data_type(), Layout::from_shape(&problem.output_shape));
    let plans = kernel_plan::cached_plans(
        PlanKind::Concat,
        &OperationAttributes::None,
        &[i64::try_from(problem.axis).map_err(|_| Error::out_of_range("concat axis"))?],
        &tensors.iter().collect::<Vec<_>>(),
        |planner, inputs| planner.concat(&output, inputs, problem.axis),
    )?;
    let steps = plans
        .iter()
        .zip(
            tensors
                .iter()
                .filter(|tensor| tensor.shape().element_count() != 0),
        )
        .map(|(plan, tensor)| (plan, vec![tensor]))
        .collect::<Vec<_>>();
    kernel_plan::run_steps(&steps, first, first.data_type(), &problem.output_shape)
}

pub fn repeat_kv(input: &Tensor, repeats: i64) -> Result<Tensor> {
    let problem = spec::repeat_kv_problem(input.shape(), repeats, "cuda repeat_kv")?;
    if repeats == 1 {
        return Ok(input.clone());
    }

    let output = KernelInput::new(input.data_type(), Layout::from_shape(&problem.output_shape));
    let plans = kernel_plan::cached_plans(
        PlanKind::RepeatKv,
        &OperationAttributes::None,
        &[repeats],
        &[input],
        |planner, inputs| Ok(vec![planner.repeat_kv(&output, &inputs[0], repeats)?]),
    )?;
    kernel_plan::run_steps(
        &[(&plans[0], vec![input])],
        input,
        input.data_type(),
        &problem.output_shape,
    )
}
