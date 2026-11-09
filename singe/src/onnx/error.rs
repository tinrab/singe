use std::{path::PathBuf, result::Result as StdResult};

use singe_onnx::{Error as OnnxError, TensorDataType};
use thiserror::Error;

use crate::error::Error as SingeError;

/// Error returned by Singe ONNX loading, import, and execution APIs.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The logical program has semantics outside the documented export subset.
    #[error("unsupported onnx export: {reason}")]
    UnsupportedExport {
        /// Program or graph semantic requirement that the ONNX export subset cannot represent.
        reason: String,
    },
    /// Invalid model-level declarations, such as duplicate or missing opsets.
    #[error("invalid onnx model: {reason}")]
    InvalidModel {
        /// Declaration that could not be validated.
        reason: String,
    },
    /// Failed to read or parse an ONNX model.
    #[error(transparent)]
    Read(#[from] OnnxError),

    /// Failed while building or running the Singe graph.
    #[error(transparent)]
    Singe(#[from] SingeError),

    /// ONNX model data refers to an unsupported operator.
    #[error("unsupported onnx operator {domain}{operator} opset {version}: {reason}")]
    UnsupportedOperator {
        /// Operator domain.
        domain: String,
        /// Operator name.
        operator: String,
        /// Imported opset version.
        version: i64,
        /// Diagnostic reason.
        reason: String,
    },

    /// ONNX value or tensor type is unsupported.
    #[error("unsupported onnx type for `{name}`: {data_type:?} ({reason})")]
    UnsupportedType {
        /// Value name.
        name: String,
        /// ONNX tensor data type.
        data_type: TensorDataType,
        /// Diagnostic reason.
        reason: String,
    },

    /// ONNX shape cannot be represented by Singe for import.
    #[error("unsupported onnx shape for `{name}`: {reason}")]
    UnsupportedShape {
        /// Value name.
        name: String,
        /// Diagnostic reason.
        reason: String,
    },

    /// ONNX value was referenced before being declared.
    #[error("missing onnx value `{0}`")]
    MissingValue(String),

    /// ONNX value namespace contains a duplicate value.
    #[error("duplicate onnx value `{0}`")]
    DuplicateValue(String),

    /// ONNX attribute is invalid for a supported operator.
    #[error("invalid onnx attribute `{attribute}` on `{node}`: {reason}")]
    InvalidAttribute {
        /// Node name or operator name.
        node: String,
        /// Attribute name.
        attribute: String,
        /// Diagnostic reason.
        reason: String,
    },

    /// ONNX initializer data is invalid or unsupported.
    #[error("invalid onnx initializer `{name}`: {reason}")]
    InvalidInitializer {
        /// Initializer name.
        name: String,
        /// Diagnostic reason.
        reason: String,
    },

    /// ONNX external tensor data cannot be read.
    #[error("invalid onnx external data `{name}` at {path}: {reason}")]
    InvalidExternalData {
        /// Tensor name.
        name: String,
        /// External data path.
        path: PathBuf,
        /// Diagnostic reason.
        reason: String,
    },
}

impl Error {
    pub(crate) fn unsupported_operator(
        domain: Option<&str>,
        operator: impl Into<String>,
        version: i64,
        reason: impl Into<String>,
    ) -> Self {
        let domain = match domain {
            Some("") | None => String::new(),
            Some(domain) => format!("{domain}::"),
        };
        Self::UnsupportedOperator {
            domain,
            operator: operator.into(),
            version,
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_attribute(
        node: impl Into<String>,
        attribute: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidAttribute {
            node: node.into(),
            attribute: attribute.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_initializer(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInitializer {
            name: name.into(),
            reason: reason.into(),
        }
    }
}

/// Result type used by Singe ONNX APIs.
pub type Result<T> = StdResult<T, Error>;
