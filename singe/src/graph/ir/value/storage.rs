use crate::graph::ir::ValueId;

/// Alias relationship for a graph value.
/// The value owns its exposed layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueAlias {
    source: ValueId,
    metadata_only: bool,
}

impl ValueAlias {
    /// Records an alias source value and whether the alias changes metadata only.
    /// The constructor does not validate the source against an IR graph.
    #[must_use]
    pub fn new(source: ValueId, metadata_only: bool) -> Self {
        Self {
            source,
            metadata_only,
        }
    }

    /// Returns the source value that owns or defines the aliased storage.
    #[must_use]
    pub fn source(&self) -> ValueId {
        self.source
    }

    /// Returns this alias with another source value, as an IR rewrite renumbers values.
    pub(crate) fn with_source(&self, source: ValueId) -> Self {
        Self {
            source,
            metadata_only: self.metadata_only,
        }
    }
}
