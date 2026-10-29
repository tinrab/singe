//! Native NPP allocation ownership and context-associated byte regions.

use std::{mem::ManuallyDrop, ptr, sync::Arc};

use singe_cuda::{
    context::Context,
    view::{AllocationId, DeviceRegion, DeviceRegionProvider},
};
use singe_npp_sys as sys;

use crate::error::{Error, Result};

/// The native allocator whose matching free function owns an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationKind {
    /// Signal storage released by `nppsFree`.
    Signal,
    /// Image storage released by `nppiFree`.
    Image,
}

/// An NPP allocation retained independently of its typed views.
/// Retention grants neither initialized input access nor permission to mutate it.
#[derive(Debug)]
pub struct Allocation {
    parts: RawAllocation,
}

/// Native allocation ownership, full bounds, and actual CUDA context.
///
/// Free the pointer with the matching NPP allocator before dropping its context,
/// or transfer the complete value through [`Allocation::from_raw`].
#[derive(Debug)]
pub struct RawAllocation {
    /// Native allocation base pointer to be released by the allocator selected in `kind`.
    pub pointer: *mut u8,
    /// Full allocation capacity in bytes, including any native image pitch padding.
    pub bytes: usize,
    /// Actual CUDA context that owns the native allocation.
    pub context: Arc<Context>,
    /// Stable allocation identity shared by every region that aliases this allocation.
    pub id: AllocationId,
    /// Native allocator family used to allocate and release this pointer.
    pub kind: AllocationKind,
}

// SAFETY: metadata access does not dereference device memory. Destruction binds its context.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "native owners transfer their handles and storage borrows under the adjacent safety contract"
)]
unsafe impl Send for Allocation {}

unsafe impl Sync for Allocation {}

impl Allocation {
    /// Adopts unique native ownership after validating the pointer range.
    ///
    /// Validation errors do not free the native pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if the pointer is null, the byte count is zero, or the pointer range
    /// cannot form a valid device region.
    ///
    /// # Safety
    ///
    /// `parts.pointer` must uniquely own `parts.bytes` accessible bytes allocated by the declared
    /// NPP allocator in `parts.context`. Preserve `parts.id` for every alias and retain an owner
    /// through all queued accesses and captured graph replay. Native allocation liveness,
    /// context, capacity, and allocator family must be accurate.
    pub unsafe fn from_raw(parts: RawAllocation) -> Result<Self> {
        if parts.bytes == 0 || parts.pointer.is_null() {
            return Err(Error::out_of_range("npp allocation"));
        }
        // SAFETY: the caller establishes native liveness; this checks address arithmetic.
        unsafe {
            DeviceRegion::from_raw_parts(&parts.context, parts.id, parts.pointer, parts.bytes)?;
        }
        Ok(Self { parts })
    }

    /// Returns the CUDA context that owns this allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.parts.context
    }

    /// Returns the full allocation capacity in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.parts.bytes
    }

    /// Returns the allocator family that will release this allocation.
    #[must_use]
    pub fn kind(&self) -> AllocationKind {
        self.parts.kind
    }

    /// Returns the native allocation base pointer without transferring ownership.
    ///
    /// The pointer grants no initialization or memory-access permission by itself.
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.parts.pointer
    }

    /// Transfers native ownership with its context and allocation identity.
    #[must_use]
    pub fn into_raw(self) -> RawAllocation {
        let owner = ManuallyDrop::new(self);
        // SAFETY: transfer the only owned fields while suppressing native destruction.
        unsafe { ptr::read(&raw const owner.parts) }
    }
}

// SAFETY: raw import checks bounds and retains the actual allocation/context pair.
unsafe impl DeviceRegionProvider for Allocation {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: the allocation owns this pointer range in the stored context.
        unsafe {
            DeviceRegion::from_raw_parts(
                &self.parts.context,
                self.parts.id,
                self.parts.pointer,
                self.parts.bytes,
            )
        }
        .expect("owned npp allocation metadata is valid")
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Err(error) = self.parts.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!("failed to bind npp allocation context: {error}");
            return;
        }
        // SAFETY: the last owner releases the native base using its matching allocator.
        unsafe {
            match self.parts.kind {
                AllocationKind::Signal => sys::nppsFree(self.parts.pointer.cast()),
                AllocationKind::Image => sys::nppiFree(self.parts.pointer.cast()),
            }
        }
    }
}
