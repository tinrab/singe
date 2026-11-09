use crate::{
    backend::registry::{BackendExecution, Operation, OperationExecutionFamily, operation_support},
    data_type::DataType,
};

/// Returns the execution family for `operation` in a backend execution mode.
///
/// This is the coarse ownership boundary used by CUDA eager and graph diagnostics: metadata-only view operations, layout materialization, storage copies, backend compute, or unsupported.
#[must_use]
pub fn operation_execution_family(
    operation: Operation,
    execution: BackendExecution,
) -> OperationExecutionFamily {
    operation_support(operation).execution_family(execution)
}

/// Returns the CUDA eager execution family for `operation`.
#[must_use]
pub fn eager_execution_family(operation: Operation) -> OperationExecutionFamily {
    operation_execution_family(operation, BackendExecution::CudaEager)
}

/// Returns whether `data_type` is accepted as either an input or output type for `operation`.
///
/// Use [`supports_operation_input`] and [`supports_operation_output`] when an operation has different input and output dtype policies, such as `argmax` and `argmin`.
#[must_use]
pub fn supports_operation(
    operation: Operation,
    execution: BackendExecution,
    data_type: DataType,
) -> bool {
    let support = operation_support(operation);
    support.supports(execution)
        && (support.supports_input_data_type(execution, data_type)
            || support.supports_data_type(execution, data_type))
}

/// Returns whether an input data type is accepted for an operation in a backend execution mode.
#[must_use]
pub fn supports_operation_input(
    operation: Operation,
    execution: BackendExecution,
    data_type: DataType,
) -> bool {
    let support = operation_support(operation);
    support.supports(execution) && support.supports_input_data_type(execution, data_type)
}

/// Returns whether an output data type can be produced for an operation in a backend execution mode.
#[must_use]
pub fn supports_operation_output(
    operation: Operation,
    execution: BackendExecution,
    data_type: DataType,
) -> bool {
    let support = operation_support(operation);
    support.supports(execution) && support.supports_data_type(execution, data_type)
}
