use std::fmt;

use serde::{Deserialize, Serialize};

use crate::backend::{attribute::BackendAttributeId, tensor::BackendTensorUid};

/// Native execution-intermediate identity, separate from tensor UIDs and frontend tensor identities.
/// Preserves every native INT64 value without imposing tensor-identity constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct IntermediateId(i64);

impl IntermediateId {
    /// Preserves a native signed 64-bit intermediate identity without tensor-ID validation.
    #[must_use]
    pub const fn new(value: i64) -> Self {
        Self(value)
    }

    /// Returns the exact native signed 64-bit identity.
    #[must_use]
    pub const fn as_i64(self) -> i64 {
        self.0
    }
}

impl From<i64> for IntermediateId {
    fn from(value: i64) -> Self {
        Self::new(value)
    }
}

impl From<IntermediateId> for i64 {
    fn from(value: IntermediateId) -> Self {
        value.as_i64()
    }
}

impl fmt::Display for IntermediateId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

/// Native storage and dependency metadata for an intermediate retained between plan executions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntermediateInfo {
    /// Native execution-intermediate ID, separate from tensor UIDs.
    pub unique_id: IntermediateId,
    /// Device storage required to retain this intermediate between executions.
    pub size: usize,
    /// Tensors whose data determines whether the retained intermediate can be reused.
    pub dependent_tensor_uids: Vec<BackendTensorUid>,
    /// Operation-graph attribute dependencies, when the native query is supported.
    /// `None` means unavailable, while `Some(Vec::new())` reports no such dependencies.
    /// Numeric IDs remain available when newer runtimes report names absent from the checked-in headers.
    pub dependent_attributes: Option<Vec<BackendAttributeId>>,
}
