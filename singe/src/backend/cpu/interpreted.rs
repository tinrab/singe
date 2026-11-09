//! Element-wise operations and reductions on the CPU through the kernel IR interpreter.
//!
//! Every element-wise operation is one kernel IR expression and every reduction one row kernel, which CUDA emits and the CPU interprets here, so the two backends compute the same values for every data type.
//! The native CPU kernels remain fast paths for `F32` and `F64`, whose results the conformance harness checks against the same references.

use singe_codegen::interpret;

use crate::{
    backend::{
        kernel_lowering::{
            ElementGroup, ElementWiseKernel, batch_norm as batch_norm_kernel, check_status_code,
            normalization as normalization_kernel, reduction as reduction_kernel,
            softmax as softmax_kernel,
        },
        registry::Operation,
        spec,
    },
    data_type::DataType,
    device::Device,
    dispatch::KernelCall,
    error::{Error, Result},
    layout::Layout,
    tensor::Tensor,
};

/// Returns whether the CPU computes `operation` with a kernel IR kernel where no native kernel covers a call's types.
#[must_use]
pub fn interprets(operation: Operation) -> bool {
    is_reduction(operation) || is_row(operation) || is_element_wise(operation)
}

// Softmaxes and normalizations, whose native CPU kernels compute only in `F32`.
fn is_row(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::Softmax
            | Operation::LogSoftmax
            | Operation::LayerNorm
            | Operation::RmsNorm
            | Operation::BatchNorm
    )
}

fn is_reduction(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::Sum
            | Operation::Mean
            | Operation::Max
            | Operation::Min
            | Operation::Product
            | Operation::LogSumExp
    )
}

fn is_element_wise(operation: Operation) -> bool {
    (operation.is_unary_elementwise()
        && !matches!(operation, Operation::Copy | Operation::Contiguous))
        || operation.is_binary_elementwise()
        || operation.is_scalar_elementwise()
        || matches!(
            operation,
            Operation::Clamp | Operation::Lerp | Operation::Where | Operation::Cast
        )
}

/// Returns whether a native CPU kernel computes `call`: one in a type the operation's native kernel computes in, `F32` and for most operations `F64`, a logical operation, or a cast between floating and Boolean types.
#[must_use]
pub fn is_native(call: &KernelCall<'_>) -> bool {
    let request = call.request();
    let operation = request.operation();
    if is_row(operation) {
        return call
            .inputs()
            .iter()
            .all(|input| input.data_type() == DataType::F32);
    }

    let floating = |data_type: DataType| {
        data_type == DataType::F32
            || (data_type == DataType::F64
                && !matches!(
                    operation,
                    Operation::Clamp | Operation::Lerp | Operation::SwiGlu
                ))
    };
    if operation == Operation::Cast {
        // Native casts round floats as the interpreter does, but narrow integers differently and hold no 8-bit floats.
        let integer = |data_type: DataType| {
            (!data_type.is_float() && data_type != DataType::Bool)
                || matches!(
                    data_type,
                    DataType::F8E4M3 | DataType::F8E5M2 | DataType::F4E2M1
                )
        };
        return call
            .inputs()
            .iter()
            .all(|input| !integer(input.data_type()))
            && !integer(request.inference().outputs()[0].data_type());
    }

    if matches!(
        operation,
        Operation::LogicalAnd | Operation::LogicalOr | Operation::LogicalNot
    ) {
        return true;
    }
    floating(request.inference().computation_data_type())
        && call
            .inputs()
            .iter()
            .all(|input| floating(input.data_type()) || input.data_type() == DataType::Bool)
}

/// Computes `call` by interpreting its kernel IR kernel.
///
/// # Errors
///
/// Returns an error if the kernel cannot be built for the call's types, or the interpreter rejects it, as for an integer division by zero.
pub fn run(call: &KernelCall<'_>) -> Result<Tensor> {
    let request = call.request();
    let operation = request.operation();
    let mut buffers = call
        .inputs()
        .iter()
        .map(Tensor::to_bytes)
        .collect::<Result<Vec<_>>>()?;
    // Every input is read from its contiguous bytes.
    let layouts = call
        .inputs()
        .iter()
        .map(|input| Layout::from_shape(input.shape()))
        .collect::<Vec<_>>();
    let operands = call
        .inputs()
        .iter()
        .zip(&layouts)
        .map(|(input, layout)| (input.data_type(), layout))
        .collect::<Vec<_>>();
    let inferred = request
        .inference()
        .outputs()
        .first()
        .ok_or_else(|| Error::missing_output("interpreted run"))?
        .clone();
    if inferred.shape().element_count() == 0 {
        return Tensor::zeroed_on(Device::Cpu, inferred.data_type(), inferred.shape());
    }

    let (kernel, bindings, output) = if is_row(operation) {
        let (first, parameters) = operands
            .split_first()
            .ok_or_else(|| Error::missing_input("interpreted run"))?;
        let group = ElementGroup::input(first.0, first.1);
        let attributes = call.attributes();
        let lowered = if let Some(batch) = attributes.batch_normalization() {
            batch_norm_kernel(batch, &operands)?
        } else if let Some(normalization) = attributes.layer_normalization() {
            normalization_kernel(operation, normalization, &group, parameters)?
        } else {
            let dim = attributes
                .dimension()
                .ok_or_else(|| Error::invalid_argument("cpu softmax attributes"))?;
            softmax_kernel(operation, &group, dim, None)?
        };
        let output = lowered
            .outputs
            .first()
            .ok_or_else(|| Error::missing_output("interpreted run"))?
            .clone();
        (lowered.kernel, lowered.bindings, output)
    } else if is_reduction(operation) {
        let reduction = call
            .attributes()
            .reduction()
            .ok_or_else(|| Error::invalid_argument("cpu reduction attributes"))?;
        let [(data_type, layout)] = operands[..] else {
            return Err(Error::missing_input("interpreted run"));
        };
        // A reduction over no elements is the operation's identity, which needs no kernel.
        let problem =
            spec::reduction_problem(layout.shape(), &reduction.dimensions, reduction.keepdim)?;
        if let Some(fill) = spec::empty_reduction_fill(layout.shape(), &problem, operation)? {
            return fill.full(Device::Cpu, inferred.data_type(), inferred.shape());
        }

        let lowered = reduction_kernel(
            operation,
            &ElementGroup::input(data_type, layout),
            &reduction.dimensions,
            reduction.keepdim,
            None,
        )?;
        let output = lowered
            .outputs
            .first()
            .ok_or_else(|| Error::missing_output("interpreted run"))?
            .clone();
        (lowered.kernel, lowered.bindings, output)
    } else {
        let output = inferred;
        let name = format!("singe_{}", operation.name());
        let (mut kernel, values) = ElementWiseKernel::create(&name, output.shape(), &operands)?;
        let value = kernel.apply(operation, call.attributes(), &values)?;
        kernel.store(value, output.data_type())?;
        let (kernel, bindings, _) = kernel.finish();
        (kernel, bindings, output)
    };
    let bytes = output
        .shape()
        .element_count()
        .checked_mul(output.data_type().size_bytes())
        .ok_or_else(|| Error::out_of_range("cpu interpreted output size"))?;
    buffers.push(vec![0; bytes]);
    check_status_code(interpret::run(&kernel, &bindings, &mut buffers)?)?;
    let bytes = buffers
        .pop()
        .ok_or_else(|| Error::missing_output("interpreted run"))?;
    Tensor::from_bytes(
        Device::Cpu,
        output.data_type(),
        output.shape().clone(),
        &bytes,
    )
}
