use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, UnsupportedTarget},
    graph::ValueId,
};

/// Structured runtime guard validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphGuardFailure {
    /// A required graph input was missing.
    MissingInput,
    /// A runtime input name was not part of the compiled graph surface.
    UnexpectedInput(String),
    /// A runtime input attempted to override a static initializer.
    StaticInputOverride(String),
    /// A static initializer's storage identity changed after guard creation.
    StaticInputStorageMismatch {
        /// Graph input name.
        name: String,
    },
    /// A runtime input had mismatched dtype.
    DataTypeMismatch {
        /// Name of the input or value being validated.
        name: String,
        /// Expected data type.
        expected: DataType,
        /// Actual data type.
        actual: DataType,
    },
    /// A runtime input had mismatched shape.
    ShapeMismatch {
        /// Name of the input or value being validated.
        name: String,
        /// Expected dimensions.
        expected: Vec<i64>,
        /// Actual dimensions.
        actual: Vec<i64>,
    },
    /// A runtime input had mismatched layout.
    LayoutMismatch {
        /// Name of the input or value being validated.
        name: String,
        /// Expected dimensions.
        expected_dimensions: Box<[i64]>,
        /// Expected strides.
        expected_strides: Box<[i64]>,
        /// Expected storage element offset.
        expected_offset: usize,
        /// Actual dimensions.
        actual_dimensions: Box<[i64]>,
        /// Actual strides.
        actual_strides: Box<[i64]>,
        /// Actual storage element offset.
        actual_offset: usize,
    },
    /// A runtime input lived on the wrong device.
    DeviceMismatch {
        /// Name of the input or value being validated.
        name: String,
        /// Expected device.
        expected: Device,
        /// Actual device.
        actual: Device,
    },
    /// A runtime input did not provide enough backing storage.
    LengthMismatch {
        /// Name of the buffer being validated.
        name: String,
        /// Required length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A symbolic or ranged guard was violated.
    OutOfRange {
        /// Name of the guarded value.
        name: String,
    },
    /// A named input dimension was outside its guarded range.
    InputDimensionOutOfRange {
        /// Graph input name.
        input: String,
        /// Zero-based input dimension axis.
        axis: usize,
        /// Symbolic dimension name.
        name: String,
        /// Inclusive minimum allowed dimension value.
        min: Option<i64>,
        /// Inclusive maximum allowed dimension value.
        max: Option<i64>,
        /// Runtime dimension value.
        actual: i64,
    },
    MissingValue(ValueId),
}

impl From<GraphGuardFailure> for Error {
    fn from(failure: GraphGuardFailure) -> Self {
        match failure {
            GraphGuardFailure::MissingInput => Self::missing_input("graph input guard"),
            GraphGuardFailure::UnexpectedInput(name) => Self::UnexpectedInput(name),
            GraphGuardFailure::StaticInputOverride(name) => Self::Unsupported {
                target: UnsupportedTarget::Operation,
                operation: "static graph input override".into(),
                details: format!("input `{name}` is a static initializer").into(),
                source: None,
            },
            GraphGuardFailure::StaticInputStorageMismatch { name } => Self::Unsupported {
                target: UnsupportedTarget::Operation,
                operation: "static graph input storage identity".into(),
                details: format!("input `{name}` static initializer storage changed").into(),
                source: None,
            },
            GraphGuardFailure::DataTypeMismatch {
                name,
                expected,
                actual,
            } => Self::data_type_mismatch(name, expected, actual),
            GraphGuardFailure::ShapeMismatch {
                name,
                expected,
                actual,
            } => Self::shape_mismatch(name, expected, actual),
            GraphGuardFailure::LayoutMismatch {
                name,
                expected_dimensions,
                expected_strides,
                expected_offset,
                actual_dimensions,
                actual_strides,
                actual_offset,
            } => Self::layout_mismatch(
                name,
                expected_dimensions.into_vec(),
                expected_strides.into_vec(),
                expected_offset,
                actual_dimensions.into_vec(),
                actual_strides.into_vec(),
                actual_offset,
            ),
            GraphGuardFailure::DeviceMismatch {
                name,
                expected,
                actual,
            } => Self::device_mismatch(name, expected, actual),
            GraphGuardFailure::LengthMismatch {
                name,
                expected,
                actual,
            } => Self::length_mismatch(name, expected, actual),
            GraphGuardFailure::OutOfRange { name } => Self::OutOfRange { name: name.into() },
            GraphGuardFailure::InputDimensionOutOfRange {
                input,
                axis,
                name,
                min,
                max,
                actual,
            } => Self::OutOfRange {
                name: format!(
                    "{input} dimension `{name}` axis {axis} expected range {}..={} got {actual}",
                    min.map_or_else(|| "-inf".to_string(), |min| min.to_string()),
                    max.map_or_else(|| "inf".to_string(), |max| max.to_string())
                )
                .into(),
            },
            GraphGuardFailure::MissingValue(value) => {
                Self::missing_node(format!("graph guard of value {}", value.0))
            }
        }
    }
}
