use crate::graph::guards::StaticInputStorageGuard;

impl StaticInputStorageGuard {
    /// Returns the process-local storage owner identity.
    #[must_use]
    pub fn storage_identity(self) -> usize {
        self.storage_identity
    }

    /// Returns the process-local allocation data pointer identity, when available.
    #[must_use]
    pub fn data_pointer_identity(self) -> Option<usize> {
        self.data_pointer_identity
    }

    /// Returns the tensor storage element offset.
    #[must_use]
    pub fn storage_offset(self) -> usize {
        self.offset
    }

    /// Returns the backing storage allocation byte size.
    #[must_use]
    pub fn storage_byte_size(self) -> usize {
        self.storage_byte_size
    }

    /// Returns the storage write version when the guard was taken.
    #[must_use]
    pub fn version(self) -> u64 {
        self.version
    }
}
