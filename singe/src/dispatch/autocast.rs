use std::cell::Cell;

use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    device::Device,
    dispatch::{DispatchKey, Kernel, KernelCall, RegistryBuilder, call_one},
    error::{Error, Result},
    tensor::Tensor,
};

thread_local! {
    static AUTOCAST: Cell<Option<DataType>> = const { Cell::new(None) };
}

/// Returns the reduced-precision type of the active autocast scope on this thread, if any.
#[must_use]
pub fn autocast_data_type() -> Option<DataType> {
    AUTOCAST.with(Cell::get)
}

/// Runs CUDA matrix multiplications of FP32 operands in `data_type` until the guard drops.
///
/// Results keep the reduced precision, as `torch.autocast` does.
/// Scopes nest; the innermost one applies.
///
/// # Errors
///
/// Returns an error if `data_type` is not [`DataType::F16`] or [`DataType::Bf16`].
pub fn autocast(data_type: DataType) -> Result<AutocastGuard> {
    if !matches!(data_type, DataType::F16 | DataType::Bf16) {
        return Err(Error::unsupported_data_type(
            "autocast type must be f16 or bf16",
        ));
    }

    let previous = AUTOCAST.with(|autocast| autocast.replace(Some(data_type)));
    Ok(AutocastGuard { previous })
}

/// Active autocast scope, returned by [`autocast`].
#[derive(Debug)]
#[must_use = "autocast applies only while the guard is alive"]
pub struct AutocastGuard {
    previous: Option<DataType>,
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST.with(|autocast| autocast.set(self.previous));
    }
}

pub fn register(builder: &mut RegistryBuilder) {
    builder.kernel(
        Operation::MatMul,
        DispatchKey::Autocast,
        Kernel::new(matmul),
    );
}

/// Casts FP32 CUDA operands to the autocast type, and passes every other call through.
fn matmul(call: &KernelCall<'_>) -> Result<Vec<Tensor>> {
    let Some(data_type) = autocast_data_type() else {
        return call.redispatch();
    };
    let applies = call.inputs().iter().all(|input| {
        matches!(input.device(), Device::Cuda(_)) && input.data_type() == DataType::F32
    });
    if !applies {
        return call.redispatch();
    }

    let lhs = call.input(0)?.cast(data_type)?;
    let rhs = call.input(1)?.cast(data_type)?;
    Ok(vec![call_one(
        Operation::MatMul,
        OperationAttributes::None,
        &[&lhs, &rhs],
    )?])
}
