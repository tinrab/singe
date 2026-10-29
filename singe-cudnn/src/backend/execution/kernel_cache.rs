use serde::{Deserialize, Serialize};

use crate::{
    backend::{
        attribute::{BackendAttributeName, BackendAttributeType},
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::{EngineConfig, OperationGraph},
    },
    error::{Error, Result},
    try_ffi,
    utility::bind_context,
};

/// cuDNN's native JSON serialization of a kernel cache.
///
/// The format is defined by cuDNN and is opaque to this crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelCacheJson(String);

impl KernelCacheJson {
    /// Wraps opaque native kernel-cache JSON without parsing or validating it.
    pub fn new(json: impl Into<String>) -> Self {
        Self(json.into())
    }

    /// Borrows the opaque native JSON string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the wrapper and returns the opaque native JSON string.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<[u8]> for KernelCacheJson {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

/// Owns a native kernel-cache descriptor that must be finalized before queries or plan use.
#[derive(Debug)]
pub struct KernelCache {
    descriptor: BackendDescriptor,
}

impl KernelCache {
    /// Creates an empty native kernel-cache descriptor without finalizing it.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN cannot create the descriptor or returns a null handle.
    pub fn create() -> Result<Self> {
        Ok(Self {
            descriptor: BackendDescriptor::create(BackendDescriptorType::KernelCache)?,
        })
    }

    /// Restores the native cache payload without finalizing it.
    /// Call [`Self::build`] with an optional operation graph before querying or using the cache.
    ///
    /// # Errors
    ///
    /// Returns an error if descriptor creation fails, the supplied payload is empty or its byte count does not fit `i64`, or cuDNN rejects the cache attribute assignment.
    pub fn deserialize(data: impl AsRef<[u8]>) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::KernelCache)?;
        descriptor.set_attribute_char_buffer(
            BackendAttributeName::KernelCacheJsonRepresentation,
            data.as_ref(),
        )?;
        Ok(Self { descriptor })
    }

    /// Queries whether the given engine configuration is cached.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache is not finalized, the engine configuration’s CUDA context cannot be bound, or the native cache query fails.
    pub fn is_engine_config_cached(&self, config: &EngineConfig) -> Result<bool> {
        if !self.is_finalized() {
            return Err(Error::DescriptorRequiredFinalized);
        }
        bind_context(config.cuda_context())?;
        let mut raw = config.descriptor().as_raw();
        let mut count = 0;
        // SAFETY: `cudnnBackendGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(singe_cudnn_sys::cudnnBackendGetAttribute(
                self.descriptor.as_raw(),
                BackendAttributeName::KernelCacheIsEngineCfgKernelCached.into(),
                BackendAttributeType::BackendDescriptor.into(),
                1,
                &raw mut count,
                (&raw mut raw).cast(),
            ))?;
        }
        Ok(count > 0)
    }

    /// Counter that shows whether the contents of this cache changed.
    /// Compare values from the same descriptor; serialization does not retain this count.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache is not finalized or the native attribute query fails or does not return one value.
    pub fn revision(&self) -> Result<i64> {
        self.descriptor
            .attribute_i64(BackendAttributeName::KernelCacheRevision)
    }

    /// Number of entries currently held in the cache.
    /// Replacing an entry changes the revision but not this count.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache is not finalized or the native attribute query fails or does not return one value.
    pub fn entry_count(&self) -> Result<i64> {
        self.descriptor
            .attribute_i64(BackendAttributeName::KernelCacheEntryCount)
    }

    /// Returns the finalized cache’s opaque native JSON representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache is not finalized or native payload querying fails, including invalid or persistently changing counts.
    pub fn json_representation(&self) -> Result<KernelCacheJson> {
        self.descriptor
            .attribute_char_string(BackendAttributeName::KernelCacheJsonRepresentation)
            .map(KernelCacheJson)
    }

    /// Saves the exact native character payload, including any native trailing NUL.
    /// Variable-size reads retry when a concurrent native cache update grows the payload.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache is not finalized or native payload querying fails, including invalid or persistently changing counts.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        self.descriptor
            .attribute_char_buffer(BackendAttributeName::KernelCacheJsonRepresentation)
    }

    /// Finalizes the cache, optionally associating it with an operation graph.
    /// Restored caches can be finalized without reconstructing a graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph’s retained CUDA context cannot be bound or cuDNN rejects graph association or cache finalization. An already finalized cache returns successfully without rebinding or changing attributes.
    pub fn build(&mut self, operation_graph: Option<&OperationGraph>) -> Result<()> {
        if self.descriptor.is_finalized() {
            return Ok(());
        }

        bind_context(operation_graph.and_then(OperationGraph::cuda_context))?;
        if let Some(operation_graph) = operation_graph {
            self.descriptor.set_attribute_descriptor(
                BackendAttributeName::KernelCacheOperationGraph,
                operation_graph.descriptor(),
            )?;
        }
        self.descriptor.finalize()
    }

    /// Returns whether the cache descriptor has been finalized.
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.descriptor.is_finalized()
    }

    /// Borrows the native kernel-cache descriptor, which may still be unfinalized.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
