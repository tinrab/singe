use crate::{
    data_type::DataType,
    device::Device,
    graph::{
        ValueId,
        guards::{ConstantGuard, ConstantGuardKind},
    },
    layout::Layout,
};

impl ConstantGuard {
    /// Returns the IR value guarded by this constant.
    #[must_use]
    pub fn value(&self) -> ValueId {
        self.value
    }

    /// Returns the guarded constant data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the guarded constant layout.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the host content digest when this constant is cacheable by value.
    #[must_use]
    pub fn host_content_digest(&self) -> Option<u64> {
        match self.kind {
            ConstantGuardKind::HostContent { digest } => Some(digest),
            ConstantGuardKind::GraphLocal { .. } => None,
        }
    }

    /// Returns the graph-local device when this constant must be guarded by graph identity.
    #[must_use]
    pub fn graph_local_device(&self) -> Option<Device> {
        match self.kind {
            ConstantGuardKind::HostContent { .. } => None,
            ConstantGuardKind::GraphLocal { device } => Some(device),
        }
    }
}
