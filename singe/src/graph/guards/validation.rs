use std::{collections::HashMap, result::Result as StdResult};

use crate::{
    data_type::DataType,
    device::Device,
    graph::{
        InputDimensionGuard,
        guards::{GraphGuardFailure, storage::storage_span_bytes},
    },
    layout::Layout,
    tensor::Tensor,
};

pub fn check_tensor_metadata(
    name: &str,
    expected_data_type: DataType,
    expected_layout: &Layout,
    expected_device: Device,
    dimension_guards: &[InputDimensionGuard],
    symbol_values: &mut HashMap<String, i64>,
    tensor: &Tensor,
) -> StdResult<(), GraphGuardFailure> {
    if tensor.data_type() != expected_data_type {
        return Err(GraphGuardFailure::DataTypeMismatch {
            name: name.to_owned(),
            expected: expected_data_type,
            actual: tensor.data_type(),
        });
    }

    check_dimensions(
        name,
        expected_layout.dimensions(),
        tensor.shape().dimensions(),
        dimension_guards,
        symbol_values,
    )?;
    let actual_layout = Layout::from_tensor(tensor);
    if !layout_matches_expected_shape_guard(expected_layout, &actual_layout, dimension_guards) {
        return Err(GraphGuardFailure::LayoutMismatch {
            name: name.to_owned(),
            expected_dimensions: expected_layout.dimensions().into(),
            expected_strides: expected_layout.strides().into(),
            expected_offset: expected_layout.storage_offset(),
            actual_dimensions: actual_layout.dimensions().into(),
            actual_strides: actual_layout.strides().into(),
            actual_offset: actual_layout.storage_offset(),
        });
    }

    if dimension_guards.is_empty() && actual_layout.dimensions() != expected_layout.dimensions() {
        return Err(GraphGuardFailure::ShapeMismatch {
            name: name.to_owned(),
            expected: expected_layout.dimensions().to_vec(),
            actual: tensor.shape().dimensions().to_vec(),
        });
    }

    if tensor.device() != expected_device {
        return Err(GraphGuardFailure::DeviceMismatch {
            name: name.to_owned(),
            expected: expected_device,
            actual: tensor.device(),
        });
    }

    let required = storage_span_bytes(tensor.layout(), tensor.data_type()).map_err(|_| {
        GraphGuardFailure::OutOfRange {
            name: format!("{name} storage span"),
        }
    })?;
    if required > tensor.storage().byte_size() {
        return Err(GraphGuardFailure::LengthMismatch {
            name: format!("{name} storage bytes"),
            expected: required,
            actual: tensor.storage().byte_size(),
        });
    }
    Ok(())
}

fn layout_matches_expected_shape_guard(
    expected_layout: &Layout,
    actual_layout: &Layout,
    dimension_guards: &[InputDimensionGuard],
) -> bool {
    if actual_layout.storage_offset() != expected_layout.storage_offset() {
        return false;
    }

    if actual_layout.strides() == expected_layout.strides() {
        return true;
    }

    // A dimension of one has no meaningful stride, so views that differ only there address the same elements.
    if actual_layout.dimensions() == expected_layout.dimensions()
        && actual_layout
            .dimensions()
            .iter()
            .zip(
                actual_layout
                    .strides()
                    .iter()
                    .zip(expected_layout.strides()),
            )
            .all(|(&dimension, (actual, expected))| dimension == 1 || actual == expected)
    {
        return true;
    }
    !dimension_guards.is_empty() && expected_layout.is_contiguous() && actual_layout.is_contiguous()
}

pub fn check_dimensions(
    input_name: &str,
    expected: &[i64],
    actual: &[i64],
    dimension_guards: &[InputDimensionGuard],
    symbol_values: &mut HashMap<String, i64>,
) -> StdResult<(), GraphGuardFailure> {
    if expected.len() != actual.len() {
        return Err(GraphGuardFailure::ShapeMismatch {
            name: input_name.to_owned(),
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }

    for (axis, (&expected_dimension, &actual_dimension)) in expected.iter().zip(actual).enumerate()
    {
        if let Some(guard) = dimension_guards.iter().find(|guard| guard.axis() == axis) {
            check_guard_range(guard, actual_dimension)?;
            check_guard_symbol(guard, actual_dimension, symbol_values)?;
        } else if expected_dimension != actual_dimension {
            return Err(GraphGuardFailure::ShapeMismatch {
                name: input_name.to_owned(),
                expected: expected.to_vec(),
                actual: actual.to_vec(),
            });
        }
    }
    Ok(())
}

fn check_guard_range(
    guard: &InputDimensionGuard,
    actual_dimension: i64,
) -> StdResult<(), GraphGuardFailure> {
    if let Some(min) = guard.min()
        && actual_dimension < min
    {
        return Err(input_dimension_out_of_range(guard, actual_dimension));
    }

    if let Some(max) = guard.max()
        && actual_dimension > max
    {
        return Err(input_dimension_out_of_range(guard, actual_dimension));
    }
    Ok(())
}

fn input_dimension_out_of_range(
    guard: &InputDimensionGuard,
    actual_dimension: i64,
) -> GraphGuardFailure {
    GraphGuardFailure::InputDimensionOutOfRange {
        input: guard.input().to_string(),
        axis: guard.axis(),
        name: guard.name().to_string(),
        min: guard.min(),
        max: guard.max(),
        actual: actual_dimension,
    }
}

fn check_guard_symbol(
    guard: &InputDimensionGuard,
    actual_dimension: i64,
    symbol_values: &mut HashMap<String, i64>,
) -> StdResult<(), GraphGuardFailure> {
    match symbol_values.get(guard.name()).copied() {
        Some(expected_symbol) if expected_symbol != actual_dimension => {
            return Err(GraphGuardFailure::ShapeMismatch {
                name: format!("symbolic dimension `{}`", guard.name()),
                expected: vec![expected_symbol],
                actual: vec![actual_dimension],
            });
        }
        Some(_) => {}
        None => {
            symbol_values.insert(guard.name().to_string(), actual_dimension);
        }
    }
    Ok(())
}
