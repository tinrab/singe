use serde::{Deserialize, Serialize};

use crate::data_type::DataType;

/// Graph-level default data types used by inference and inherited operation precision.
/// Each omitted field uses its operation's fallback or enclosing score graph's default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataTypePolicy {
    io: Option<DataType>,
    intermediate: Option<DataType>,
    compute: Option<DataType>,
}

impl DataTypePolicy {
    /// Leaves I/O, intermediate, and compute defaults unset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets caller-bound result precision, or clears this default with `None`.
    #[must_use]
    pub fn with_io(mut self, data_type: impl Into<Option<DataType>>) -> Self {
        self.io = data_type.into();
        self
    }

    /// Returns the optional encoding default for caller-bound inferred outputs.
    #[must_use]
    pub fn io(self) -> Option<DataType> {
        self.io
    }

    pub(crate) fn effective_io(self, default: DataType) -> DataType {
        self.io.unwrap_or(default)
    }

    /// Sets virtual result precision, or clears this default with `None`.
    #[must_use]
    pub fn with_intermediate(mut self, data_type: impl Into<Option<DataType>>) -> Self {
        self.intermediate = data_type.into();
        self
    }

    /// Returns the optional encoding default for virtual inferred outputs.
    #[must_use]
    pub fn intermediate(self) -> Option<DataType> {
        self.intermediate
    }

    pub(crate) fn effective_intermediate(self, default: DataType) -> DataType {
        self.intermediate.unwrap_or(default)
    }

    /// Sets inherited operation precision, or clears this default with `None`.
    /// Explicit operation precision remains independent of this default.
    #[must_use]
    pub fn with_compute(mut self, data_type: impl Into<Option<DataType>>) -> Self {
        self.compute = data_type.into();
        self
    }

    /// Returns the optional arithmetic-precision default inherited by operations.
    #[must_use]
    pub fn compute(self) -> Option<DataType> {
        self.compute
    }

    pub(crate) fn effective_compute(self, default: DataType) -> DataType {
        self.compute.unwrap_or(default)
    }

    pub(crate) fn resolve_compute(self, requested: Option<DataType>) -> DataType {
        requested.or(self.compute).unwrap_or(DataType::F32)
    }

    /// Fills omitted defaults without changing this graph's explicit choices.
    pub(crate) fn with_defaults(self, parent: Self) -> Self {
        Self {
            io: self.io.or(parent.io),
            intermediate: self.intermediate.or(parent.intermediate),
            compute: self.compute.or(parent.compute),
        }
    }
}
