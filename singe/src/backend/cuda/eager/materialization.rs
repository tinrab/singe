use crate::{
    backend::{
        cuda::{
            eager::shape::noop_contiguous,
            kernel_plan::{self, KernelInput, KernelPlanner},
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
    let inputs = tensors
        .iter()
        .map(KernelInput::from_source)
        .collect::<Vec<_>>();
    let steps = KernelPlanner::default()
        .concat(&output, &inputs, problem.axis)?
        .into_iter()
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
    let plan =
        KernelPlanner::default().repeat_kv(&output, &KernelInput::from_source(input), repeats)?;
    kernel_plan::run_steps(
        &[(plan, vec![input])],
        input,
        input.data_type(),
        &problem.output_shape,
    )
}
