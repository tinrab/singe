use crate::{
    backend::{
        cuda::{
            self,
            cutensor::{create_cutensor_contraction_plan, launch_cutensor_contraction_f32},
            diagnostics::eager_operation_error,
            stream::ensure_not_stream_capturing,
            utility::{ensure_f32, ensure_same_device},
        },
        registry::Operation,
        spec,
    },
    data_type::DataType,
    error::Result,
    tensor::Tensor,
};

pub fn contraction(
    a: &Tensor,
    a_mode: &[i64],
    b: &Tensor,
    b_mode: &[i64],
    c: &Tensor,
    c_mode: &[i64],
    d_mode: &[i64],
    alpha: f32,
    beta: f32,
) -> Result<Tensor> {
    ensure_same_device(a, b)?;
    ensure_same_device(a, c)?;
    let input_data_types = [a.data_type(), b.data_type(), c.data_type()];
    ensure_f32(a).map_err(|error| {
        eager_operation_error(
            error,
            Operation::Contraction,
            a.data_type(),
            a.shape(),
            &input_data_types,
        )
    })?;
    ensure_f32(b).map_err(|error| {
        eager_operation_error(
            error,
            Operation::Contraction,
            b.data_type(),
            b.shape(),
            &input_data_types,
        )
    })?;
    ensure_f32(c).map_err(|error| {
        eager_operation_error(
            error,
            Operation::Contraction,
            c.data_type(),
            c.shape(),
            &input_data_types,
        )
    })?;
    ensure_not_stream_capturing(a.device(), "cuda eager contraction during stream capture")?;

    let spec = spec::contraction_problem(
        a.shape(),
        a_mode,
        b.shape(),
        b_mode,
        c.shape(),
        c_mode,
        d_mode,
    )?;
    let output_count = spec.output_shape.element_count();
    let context = cuda::execution_context(a.device())?;
    context.bind()?;
    let output = context.allocate_output(DataType::F32, &spec.output_shape)?;
    if output_count == 0 {
        // SAFETY: the empty output has no elements to initialize.
        return Ok(unsafe { output.assume_init() });
    }

    context.with_cutensor(|cutensor, stream| {
        let (plan, mut workspace) = create_cutensor_contraction_plan(
            cutensor,
            a,
            a_mode,
            b,
            b_mode,
            c,
            c_mode,
            d_mode,
            output.region(),
        )?;
        stream.raw().sync_scope(|_| {
            Ok({
                // SAFETY: the plan is built for these tensors; eager execution retains storage through the stream's ordered reuse.
                unsafe {
                    launch_cutensor_contraction_f32(
                        &plan,
                        stream,
                        a,
                        b,
                        c,
                        output.region(),
                        alpha,
                        beta,
                        workspace.as_mut(),
                    )
                }
            })
        })?
    })?;
    // SAFETY: the completed contraction writes every element of the contiguous output.
    Ok(unsafe { output.assume_init() })
}
