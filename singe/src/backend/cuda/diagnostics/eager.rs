use std::fmt::Write as _;

use crate::{
    backend::registry::{BackendExecution, Operation, operation_name, operation_support},
    data_type::DataType,
    error::{Error, UnsupportedTarget},
    shape::Shape,
};

/// Adds operation, provider, input type, and shape details to an unsupported-data-type error.
/// Other error variants are returned unchanged.
#[must_use]
pub fn eager_operation_error(
    error: Error,
    operation: Operation,
    data_type: DataType,
    shape: &Shape,
    input_data_types: &[DataType],
) -> Error {
    if !matches!(
        error,
        Error::Unsupported {
            target: UnsupportedTarget::DataType,
            ..
        }
    ) {
        return error;
    }

    let support = operation_support(operation);
    let mut details = format!(
        "providers [{}], execution family {}, dtype {:?}, supported dtypes [{}], layout {}, shape {:?}",
        support.provider_list(),
        support.execution_family_name(BackendExecution::CudaEager),
        data_type,
        support.data_type_list(BackendExecution::CudaEager),
        support.layout_name(),
        shape.dimensions(),
    );
    if support.has_distinct_input_data_types(BackendExecution::CudaEager) {
        write!(
            details,
            ", supported input dtypes [{}]",
            support.input_data_type_list(BackendExecution::CudaEager)
        )
        .expect("formatting into a string cannot fail");
    }

    if !input_data_types.is_empty() {
        write!(details, ", input dtypes {input_data_types:?}")
            .expect("formatting into a string cannot fail");
    }
    Error::Unsupported {
        operation: operation_name(operation).into(),
        target: UnsupportedTarget::DataType,
        details: details.into(),
    }
}

/// Creates an unsupported-operation error with CUDA eager provider, type, layout, and shape details.
#[must_use]
pub fn eager_unsupported_operation(
    operation: Operation,
    data_type: DataType,
    shape: &Shape,
    input_data_types: &[DataType],
) -> Error {
    let support = operation_support(operation);
    let mut details = format!(
        "providers [{}], execution family {}, dtype {:?}, supported dtypes [{}], layout {}, shape {:?}, cuda eager implementation missing",
        support.provider_list(),
        support.execution_family_name(BackendExecution::CudaEager),
        data_type,
        support.data_type_list(BackendExecution::CudaEager),
        support.layout_name(),
        shape.dimensions(),
    );
    if support.has_distinct_input_data_types(BackendExecution::CudaEager) {
        write!(
            details,
            ", supported input dtypes [{}]",
            support.input_data_type_list(BackendExecution::CudaEager)
        )
        .expect("formatting into a string cannot fail");
    }

    if !input_data_types.is_empty() {
        write!(details, ", input dtypes {input_data_types:?}")
            .expect("formatting into a string cannot fail");
    }
    Error::unsupported_operation_with_details(operation_name(operation), details)
}
