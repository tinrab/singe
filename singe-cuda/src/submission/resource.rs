use std::{any::Any, fmt, ops::Range, sync::Arc};

use crate::{
    context::Context,
    error::{Error, Result},
    memory::{CudaHostSlice, PinnedHostMemory},
    stream::Stream,
    submission::Completion,
    view::{
        AllocationId, DeviceRegion, DeviceRegionProvider, DeviceRepr, HostRegion, OwnedDeviceRegion,
    },
};

/// Access to a byte range by a submitted operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Reads the declared bytes without modifying them. Concurrent tracked reads do not conflict.
    Read,
    /// Writes the declared bytes without requiring their previous contents. Conflicts with overlapping tracked reads and writes.
    Write,
    /// Reads and modifies the declared bytes. Conflicts with overlapping tracked reads and writes.
    ReadWrite,
}

#[derive(Debug, Clone, Copy)]
enum AccessLocation {
    Device,
    Host { portable: bool },
}

/// An access declaration preserves allocation identity across aliases and subranges.
#[derive(Debug, Clone)]
pub struct Access {
    pub(crate) context: Arc<Context>,
    location: AccessLocation,
    pub(crate) id: AllocationId,
    pub(crate) range: Range<usize>,
    pub(crate) mode: AccessMode,
    pub(crate) readiness: Option<Stream>,
}

impl Access {
    /// Declares access to a device region, preserving its allocation identity, byte range, context, and producer readiness stream.
    ///
    /// The declaration does not retain the allocation or establish initialization.
    #[must_use]
    pub fn new(region: DeviceRegion<'_>, mode: AccessMode) -> Self {
        Self {
            context: Arc::clone(region.context()),
            location: AccessLocation::Device,
            id: region.allocation_id(),
            range: region.byte_range(),
            mode,
            readiness: region.stream().cloned(),
        }
    }

    /// Replaces the ordering declaration's mode while preserving its range and producer readiness.
    /// This does not grant memory access permissions or establish initialization.
    #[must_use]
    pub fn with_mode(mut self, mode: AccessMode) -> Self {
        self.mode = mode;
        self
    }

    pub(crate) fn allocation(
        context: &Arc<Context>,
        id: AllocationId,
        range: Range<usize>,
        mode: AccessMode,
    ) -> Self {
        Self {
            context: Arc::clone(context),
            id,
            range,
            mode,
            location: AccessLocation::Device,
            readiness: None,
        }
    }

    /// Declares access to a host region while preserving its allocation identity, byte range, context, and portability.
    ///
    /// The declaration does not retain the allocation or establish initialization.
    #[must_use]
    pub fn host(region: HostRegion<'_>, mode: AccessMode) -> Self {
        Self {
            context: Arc::clone(region.context()),
            location: AccessLocation::Host {
                portable: region.is_portable(),
            },
            id: region.allocation_id(),
            range: region.byte_range(),
            mode,
            readiness: None,
        }
    }

    pub(crate) fn validate_context(&self, context: &Context) -> Result<()> {
        if !matches!(self.location, AccessLocation::Host { portable: true })
            && self.context.as_ref() != context
        {
            return Err(Error::MemoryContextMismatch);
        }
        Ok(())
    }
}

/// Owners retained independently of an operation's output and its awaiting future.
/// Access declarations describe ordering, not initialization or permission to alias writes.
#[derive(Clone, Default)]
pub struct Resources {
    pub(crate) owners: Vec<Arc<dyn Any + Send + Sync>>,
    pub(crate) accesses: Vec<Access>,
    pub(crate) dependencies: Vec<Completion>,
}

impl fmt::Debug for Resources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resources")
            .field("owners", &self.owners.len())
            .field("accesses", &self.accesses)
            .field("dependencies", &self.dependencies.len())
            .finish()
    }
}

impl Resources {
    /// Creates an empty resource ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retains an allocation, native handle, workspace or other execution resource.
    pub fn retain<T: Any + Send + Sync>(&mut self, owner: Arc<T>) {
        self.owners.push(owner);
    }

    /// Retains a checked owned range and declares its access in one step.
    pub fn retain_access<O: DeviceRegionProvider + Send + Sync + 'static>(
        &mut self,
        region: OwnedDeviceRegion<O>,
        mode: AccessMode,
    ) {
        self.access(region.region(), mode);
        self.retain(region.into_owner());
    }

    /// Retains pinned staging and declares its host byte range without requiring a device mapping.
    pub fn retain_pinned<T: DeviceRepr>(&mut self, memory: &PinnedHostMemory<T>, mode: AccessMode) {
        self.host_access(memory.host_region(), mode);
        self.retain(Arc::clone(memory.allocation()));
    }

    /// Declares an access to a resource such as an array that has no linear device pointer.
    pub fn declare_access(&mut self, access: Access) {
        self.accesses.push(access);
    }

    /// Adds a device-region access declaration without retaining its allocation.
    ///
    /// Retain the owner separately or keep borrowed storage live through completion.
    pub fn access(&mut self, region: DeviceRegion<'_>, mode: AccessMode) {
        self.accesses.push(Access::new(region, mode));
    }

    /// Adds a host-region access declaration without retaining its allocation.
    ///
    /// Retain the owner separately or keep borrowed storage live through completion.
    pub fn host_access(&mut self, region: HostRegion<'_>, mode: AccessMode) {
        self.accesses.push(Access::host(region, mode));
    }

    /// Orders a producer's CUDA work and propagates its checked result.
    /// Retain the actual input allocations separately through consumer completion.
    pub fn depend_on(&mut self, completion: Completion) {
        self.dependencies.push(completion);
    }

    /// Moves another ledger's owners, access declarations, and dependencies into this ledger.
    pub fn extend(&mut self, resources: Self) {
        self.owners.extend(resources.owners);
        self.accesses.extend(resources.accesses);
        self.dependencies.extend(resources.dependencies);
    }
}
