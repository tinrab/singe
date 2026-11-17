use std::{
    mem::{self, MaybeUninit},
    ptr,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    error::{Error as CudaError, Result as CudaResult},
    memory::{DeviceMemory, PinnedVec},
    view::{DeviceRegion, DeviceRegionProvider},
};

use crate::{
    error::{Error, Result},
    mp::{
        context::{ContextRef, validate_same_context},
        plan::Workspace,
    },
    utility::to_usize,
};

/// Owns pending local device workspace and optional pinned host workspace.
///
/// First use retains the plan context and checks both capacities.
/// Later uses must share that context.
/// Dropping used workspace clears cached communication registrations before releasing its allocations.
#[derive(Debug)]
pub struct WorkspaceMemory<'comm> {
    context: Option<ContextRef<'comm>>,
    device: DeviceMemory<MaybeUninit<u8>>,
    host: HostWorkspace,
}

#[derive(Debug)]
struct HostWorkspace {
    memory: Option<PinnedVec<u8>>,
}

impl<'comm> WorkspaceMemory<'comm> {
    /// Allocates local device and pinned host byte storage for the supplied workspace requirements.
    ///
    /// Zero host requirements use no pinned allocation.
    /// The allocation is not associated with a multi-process handle until prepared for execution.
    ///
    /// # Errors
    ///
    /// Returns an error if a workspace byte count does not fit `usize` or CUDA device or pinned host allocation fails.
    pub fn create(allocation_context: &Arc<CudaContext>, workspace: Workspace) -> Result<Self> {
        Ok(Self {
            context: None,
            device: DeviceMemory::<u8>::create_uninit(
                allocation_context,
                to_usize(workspace.device_size(), "device workspace")?,
            )?,
            host: HostWorkspace::create(
                allocation_context,
                to_usize(workspace.host_size(), "host workspace")?,
            )?,
        })
    }

    pub(crate) fn prepare(
        &mut self,
        context: &ContextRef<'comm>,
        required: Workspace,
    ) -> Result<()> {
        if self.device.context().as_ref() != context.cuda_context().as_ref() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        if let Some(existing) = &self.context {
            validate_same_context(existing, context, "workspace")?;
        }

        let actual = self.device.byte_len() as u64;
        if actual < required.device_size() {
            return Err(Error::workspace_too_small(
                "device workspace",
                required.device_size(),
                actual,
            ));
        }

        let actual = self.host.memory.as_ref().map_or(0, Vec::capacity) as u64;
        if actual < required.host_size() {
            return Err(Error::workspace_too_small(
                "host workspace",
                required.host_size(),
                actual,
            ));
        }

        self.context = Some(context.clone());
        Ok(())
    }

    pub(crate) fn device_region(&self) -> DeviceRegion<'_> {
        self.device.region()
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "writable native pointers require an exclusive workspace borrow"
    )]
    pub(crate) fn device_mut_ptr(&mut self) -> *mut () {
        self.device.as_mut_ptr().cast()
    }

    pub(crate) fn host_mut_ptr(&mut self) -> *mut () {
        self.host.as_mut_ptr()
    }
}

impl HostWorkspace {
    fn create(context: &Arc<CudaContext>, size: usize) -> CudaResult<Self> {
        let memory = if size == 0 {
            None
        } else {
            Some(context.pinned_host().with_capacity(size)?)
        };
        Ok(Self { memory })
    }

    fn as_mut_ptr(&mut self) -> *mut () {
        self.memory
            .as_mut()
            .map_or_else(ptr::null_mut, |memory| memory.as_mut_ptr().cast())
    }

    fn release_after<'a>(&mut self, contexts: impl IntoIterator<Item = &'a Arc<CudaContext>>) {
        // Native work may still use the workspace, and the pinned pool hands a freed block out again at once.
        let Some(memory) = &self.memory else {
            return;
        };

        let waited = contexts
            .into_iter()
            .try_for_each(|context| context.synchronize())
            .and_then(|()| memory.allocator().context().synchronize());
        if let Err(error) = waited {
            report_cleanup_failure(
                "synchronize the cuda contexts to release",
                "cutensor host workspace",
                &error,
            );
            mem::forget(self.memory.take());
        }
    }
}

impl Drop for WorkspaceMemory<'_> {
    fn drop(&mut self) {
        self.host.release_after([self.device.context()]);
        if let Some(context) = &self.context
            && let Err(error) = context.clear_operation_cache()
        {
            report_cleanup_failure("clear", "cutensormp workspace registrations", &error);
        }
    }
}
