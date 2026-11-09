use crate::graph::ir::ValueId;

/// Alias group reported by CUDA graph compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphCompileAliasGroupReport {
    /// Root value for this alias group.
    pub(crate) root_value: ValueId,
    /// Values in the alias group.
    pub(crate) values: Vec<ValueId>,
    /// Whether this group contains a graph runtime or static input.
    pub(crate) contains_input: bool,
    /// Whether this group contains a graph constant.
    pub(crate) contains_constant: bool,
    /// Whether this group contains a graph output.
    pub(crate) contains_output: bool,
}

impl GraphCompileAliasGroupReport {
    pub(crate) fn new(
        root_value: ValueId,
        values: Vec<ValueId>,
        contains_input: bool,
        contains_constant: bool,
        contains_output: bool,
    ) -> Self {
        Self {
            root_value,
            values,
            contains_input,
            contains_constant,
            contains_output,
        }
    }

    /// Returns the root value for this alias group.
    #[must_use]
    pub fn root_value(&self) -> ValueId {
        self.root_value
    }

    /// Returns values in the alias group.
    #[must_use]
    pub fn values(&self) -> &[ValueId] {
        &self.values
    }

    /// Returns whether this group contains a graph runtime or static input.
    #[must_use]
    pub fn contains_input(&self) -> bool {
        self.contains_input
    }

    /// Returns whether this group contains a graph constant.
    #[must_use]
    pub fn contains_constant(&self) -> bool {
        self.contains_constant
    }

    /// Returns whether this group contains a graph output.
    #[must_use]
    pub fn contains_output(&self) -> bool {
        self.contains_output
    }
}
