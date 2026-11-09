//! Routes operation calls through dispatch keys to registered kernels.
//!
//! Every tensor operation, graph interpreter step, and module call builds an [`OperationRequest`] and enters [`call`].
//! The dispatcher runs the highest active key that has a kernel: [`DispatchKey::Autocast`], then [`DispatchKey::Capture`], then [`DispatchKey::Autograd`], then the backend key chosen from the input device.
//! While a kernel runs, its key and every key above it are excluded on the thread, so [`KernelCall::redispatch`] and nested calls reach the keys below.
//! Backends run their own kernel for an operation, or its composite kernel when they have none.

mod autocast;
mod custom;
mod kernel;
mod key;
mod registry;

pub use self::{
    autocast::{AutocastGuard, autocast, autocast_data_type},
    custom::{CustomOperation, CustomOperationDefinition},
    kernel::{Kernel, KernelCall},
    key::{DispatchKey, DispatchKeySet},
    registry::RegistryBuilder,
};
pub(crate) use self::{
    key::{ExcludeGuard, excluded},
    registry::Registry,
};
use crate::{
    activity, autograd,
    backend::{
        meta,
        registry::{
            AliasBehavior, BackendExecution, CheckedOperationRequest, Operation,
            OperationAttributes, OperationRequest, operation_schema,
            operation_schema_for_attributes,
        },
    },
    device::Device,
    error::{Error, Result, UnsupportedTarget},
    graph::capture,
    tensor::Tensor,
};

/// Runs `operation` on `inputs` and returns its results in output order.
///
/// # Errors
///
/// Returns an error if the request is invalid for the inputs, no kernel handles it on the input device, or a kernel fails.
pub fn call(
    operation: Operation,
    attributes: OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let request = OperationRequest::new(
        operation,
        attributes,
        inputs.iter().map(|input| input.value_spec()).collect(),
    )
    .check()?;
    let inputs = inputs
        .iter()
        .map(|input| (*input).clone())
        .collect::<Vec<_>>();
    dispatch_checked(&request, &inputs)
}

/// Runs an operation with exactly one result.
///
/// # Errors
///
/// Returns an error if [`call`] fails or the operation does not return one result.
pub fn call_one(
    operation: Operation,
    attributes: OperationAttributes,
    inputs: &[&Tensor],
) -> Result<Tensor> {
    // A mistaken call must fail before it submits anything.
    if operation_schema_for_attributes(operation, &attributes).output_count() != 1 {
        return Err(Error::invalid_argument("expected one operation result"));
    }

    let mut outputs = call(operation, attributes, inputs)?;
    if outputs.len() != 1 {
        return Err(Error::missing_output("call one"));
    }
    outputs
        .pop()
        .ok_or_else(|| Error::missing_output("call one"))
}

/// Runs a checked request on `inputs` with every frontend key excluded, as the graph interpreter does.
pub(crate) fn execute(request: &CheckedOperationRequest, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    let _guard = ExcludeGuard::new(DispatchKeySet::frontend());
    dispatch_checked(request, inputs)
}

/// Runs a checked request through every active key that is not excluded on this thread.
pub(crate) fn dispatch_checked(
    request: &CheckedOperationRequest,
    inputs: &[Tensor],
) -> Result<Vec<Tensor>> {
    let device = inputs
        .first()
        .ok_or_else(|| Error::missing_input("dispatch checked"))?
        .device();
    let registry = Registry::get();
    let entry = registry.entry(request.operation())?;
    let excluded = excluded();
    for key in DispatchKey::FRONTEND {
        if excluded.contains(key) || !frontend_active(key, inputs) {
            continue;
        }

        let kernel = entry
            .as_ref()
            .and_then(|entry| entry.kernel(key))
            .or_else(|| registry.fallback(key));
        if let Some(kernel) = kernel {
            let _guard = ExcludeGuard::new(key.and_higher());
            return kernel.call(&KernelCall::new(request, inputs, key));
        }
    }

    if device.is_meta() {
        let _guard = ExcludeGuard::new(DispatchKeySet::frontend());
        return meta::run(request, inputs);
    }

    let backend = backend_key(device)?;
    let kernel = entry
        .as_ref()
        .and_then(|entry| entry.backend_kernel(backend))
        .ok_or_else(|| Error::Unsupported {
            operation: request.operation().name().into(),
            target: UnsupportedTarget::Operation,
            details: format!("no kernel registered for the {backend} backend").into(),
        })?;
    let _guard = ExcludeGuard::new(DispatchKeySet::frontend());
    execute_backend(request, inputs, backend, kernel)
}

/// Returns whether `operation` has a kernel at backend `key`, its own or the composite one.
pub(crate) fn has_backend_kernel(operation: Operation, key: DispatchKey) -> bool {
    Registry::get()
        .entry(operation)
        .ok()
        .flatten()
        .is_some_and(|entry| entry.backend_kernel(key).is_some())
}

fn frontend_active(key: DispatchKey, inputs: &[Tensor]) -> bool {
    match key {
        DispatchKey::Autocast => autocast_data_type().is_some(),
        DispatchKey::Capture => capture::is_active(),
        DispatchKey::Autograd => autograd::records(inputs),
        DispatchKey::Cpu | DispatchKey::Cuda => false,
    }
}

/// Returns the backend key for `device`, which runs kernels.
fn backend_key(device: Device) -> Result<DispatchKey> {
    match device {
        Device::Cpu => Ok(DispatchKey::Cpu),
        Device::Cuda(id) if id < 0 => Err(Error::negative_cuda_device_id(id)),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Ok(DispatchKey::Cuda),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(Error::cuda_support_disabled()),
        Device::Meta => Err(Error::unsupported_device("meta tensors run no kernels")),
    }
}

/// Validates a backend call, converts inputs to the kernel's types, runs the kernel, and checks its results against the request.
fn execute_backend(
    request: &CheckedOperationRequest,
    inputs: &[Tensor],
    backend: DispatchKey,
    kernel: &Kernel,
) -> Result<Vec<Tensor>> {
    activity::invocation();
    if inputs.len() != request.inputs().len() {
        return Err(Error::invalid_argument("checked operation input count"));
    }

    let device = inputs
        .first()
        .ok_or_else(|| Error::missing_input("execute backend"))?
        .device();
    for (input, spec) in inputs.iter().zip(request.inputs()) {
        if input.value_spec() != *spec {
            return Err(Error::invalid_argument("checked operation input metadata"));
        }
        if input.device() != device {
            return Err(Error::device_mismatch(
                "operation operands",
                device,
                input.device(),
            ));
        }
    }

    request.validate_execution(match backend {
        DispatchKey::Cuda => BackendExecution::CudaEager,
        _ => BackendExecution::CpuEager,
    })?;
    let converted = inputs
        .iter()
        .zip(request.inference().kernel_input_data_types())
        .map(|(input, data_type)| {
            if input.data_type() == *data_type {
                Ok(input.clone())
            } else {
                input.cast(*data_type)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let writes =
        operation_schema(request.operation()).alias_behavior() == AliasBehavior::WritesInput;
    if writes {
        check_write(&converted)?;
    }

    let outputs = kernel.call(&KernelCall::new(request, &converted, backend))?;
    check_outputs(request, &outputs, device)?;
    if writes {
        converted[0].storage().bump_version();
    }
    Ok(outputs)
}

/// Rejects a write whose destination addresses an element twice or whose other inputs overlap it.
fn check_write(inputs: &[Tensor]) -> Result<()> {
    let destination = inputs
        .first()
        .ok_or_else(|| Error::missing_input("check write"))?;
    if destination.layout().has_internal_overlap() {
        return Err(Error::invalid_argument(
            "write destination addresses some elements more than once",
        ));
    }

    let identity = destination.storage().storage_identity();
    for input in &inputs[1..] {
        if input.storage().storage_identity() == identity
            && input.layout().may_overlap(destination.layout())
        {
            return Err(Error::invalid_argument(
                "write source overlaps its destination",
            ));
        }
    }
    Ok(())
}

fn check_outputs(
    request: &CheckedOperationRequest,
    outputs: &[Tensor],
    device: Device,
) -> Result<()> {
    let name = request.operation().name();
    if outputs.len() != request.inference().outputs().len() {
        return Err(Error::missing_output("check outputs"));
    }

    let device = match request.attributes() {
        OperationAttributes::CopyTo(device) => *device,
        _ => device,
    };
    for (output, expected) in outputs.iter().zip(request.inference().outputs()) {
        if output.data_type() != expected.data_type() {
            return Err(Error::data_type_mismatch(
                name,
                expected.data_type(),
                output.data_type(),
            ));
        }

        if output.shape() != expected.shape() {
            return Err(Error::shape_mismatch(
                name,
                expected.shape().dimensions().to_vec(),
                output.shape().dimensions().to_vec(),
            ));
        }
        if output.device() != device {
            return Err(Error::device_mismatch(name, device, output.device()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
