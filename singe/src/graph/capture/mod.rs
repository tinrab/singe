mod options;
mod recorder;

use self::recorder::{with_active_session, with_session};

pub use self::options::{CaptureBindings, CaptureInfo, CaptureOptions};
pub(crate) use self::recorder::{dimension, evaluate_dimensions, is_active, record_reshape};
use crate::{
    backend::registry::{AliasBehavior, operation_schema},
    device::Device,
    dispatch::{DispatchKey, DispatchKeySet, ExcludeGuard, Kernel, KernelCall, RegistryBuilder},
    error::{Error, Result},
    random::Generator,
    storage::{cpu::CpuStorage, meta::MetaStorage},
    tensor::Tensor,
};

/// Registers recording as the fallback kernel of the capture key.
pub(crate) fn register(builder: &mut RegistryBuilder) {
    builder.fallback(DispatchKey::Capture, Kernel::new(record));
}

/// Runs the operation below the capture key and records it with its results.
///
/// Operations that kernels run internally execute with capture excluded, so only this call is recorded.
fn record(call: &KernelCall<'_>) -> Result<Vec<Tensor>> {
    if operation_schema(call.operation()).alias_behavior() == AliasBehavior::WritesInput {
        let outputs = call.redispatch()?;
        with_session(|session| session.record_write(call.input(0)?, call.input(1)?))?;
        return Ok(outputs);
    }

    with_session(|session| {
        for input in call.inputs() {
            session.ensure_tensor(input)?;
        }
        Ok(())
    })?;
    let outputs = call.redispatch()?;
    with_session(|session| session.record_invocation(call.request(), call.inputs(), &outputs))?;
    Ok(outputs)
}

/// Registers an immutable literal created on the recording thread.
/// Closed-over replaceable tensors belong in `CaptureBindings` instead.
///
/// # Errors
///
/// Returns an error if no capture is active, the active session is still borrowed, or constant registration rejects the name or tensor binding.
pub fn constant(name: impl Into<String>, tensor: &Tensor) -> Result<Tensor> {
    recorder::register_constant(name.into(), tensor)
}

/// Returns the bound state of `generator` and its block offset when a capture is active.
pub(crate) fn generator_input(
    generator: &Generator,
    device: Device,
    blocks: u64,
) -> Result<Option<(Tensor, u64)>> {
    let mut bound = None;
    with_active_session(|session| {
        bound = Some(session.generator_input(generator, device, blocks)?);
        Ok(())
    })?;
    Ok(bound)
}

/// Introduces the known literal of a framework operation decomposition, created on `device` by `create`.
///
/// On the meta device the literal is created on the CPU and kept as the host value of a meta tensor, so a graph traced on meta tensors can be placed on a device with values.
pub(crate) fn literal(
    device: Device,
    create: impl FnOnce(Device) -> Result<Tensor>,
) -> Result<Tensor> {
    let tensor = if device.is_meta() {
        let host = create(Device::Cpu)?;
        let value = CpuStorage::from_bytes(host.data_type(), host.to_bytes()?)?;
        Tensor::create(
            MetaStorage::with_value(value),
            host.data_type(),
            host.shape().clone(),
        )?
    } else {
        create(device)?
    };
    with_active_session(|session| session.record_literal(&tensor))?;
    Ok(tensor)
}

// Returns the leading rows of `bound` that the `I64` scalar `count` names, for an operation whose result extent depends on values.
// The count is read once for the eager result, and a captured graph returns the result narrowed to its count after each run.
pub(crate) fn counted(bound: &Tensor, count: &Tensor) -> Result<Tensor> {
    let length = {
        let _guard = ExcludeGuard::new(DispatchKeySet::frontend());
        count
            .to_vec::<i64>()?
            .first()
            .copied()
            .ok_or_else(|| Error::missing_output("counted"))?
    };
    let narrowed = bound.strided_view_unrecorded(bound.layout().slice(0, 0, length, 1)?)?;
    with_active_session(|session| session.record_counted(&narrowed, bound, count))?;
    Ok(narrowed)
}

#[cfg(test)]
mod tests;
