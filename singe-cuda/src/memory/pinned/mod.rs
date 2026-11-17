//! Page-locked host memory that each CUDA context pools behind [`std::alloc::Allocator`].

mod pool;
mod slice;

use std::{
    alloc::{AllocError, Allocator, Layout},
    ptr::{self, NonNull},
    result::Result as StdResult,
    sync::Arc,
};

pub(crate) use self::pool::PinnedPool;
use self::pool::same_class;
pub use self::{pool::PinnedStats, slice::PinnedVecExt};
use crate::{
    context::Context,
    error::{Error, Result},
    view::{DeviceRegion, HostRegion, ZeroableDeviceRepr},
};

/// An allocator for page-locked host memory, drawn from a pool its CUDA context owns.
///
/// Get one from [`Context::pinned_host`].
/// A `Vec` or `Box` that uses it as its allocator lives in pinned memory, which CUDA copies read and write directly instead of staging through pageable memory.
/// [`PinnedVec`] is such a vector, and it is what this crate's pinned copies accept.
/// Cloning is cheap, and all clones from one context share its pool, so a buffer may be freed through any of them.
///
/// # GPU work
///
/// The pool does not know which buffers queued GPU work still uses.
/// Keep a buffer alive until that work has finished; scoped copies and this crate's submissions do so.
///
/// # Allocation failure
///
/// A `Vec` that cannot grow aborts the process, as it does with any allocator.
/// When the size is known up front, allocate with [`PinnedHost::with_capacity`], [`PinnedHost::from_slice`], or [`PinnedHost::zeroed`], which return an error instead.
///
/// # How it works
///
/// The pool takes chunks of at least 2 MiB from `cudaHostAlloc` and splits them into blocks.
/// Blocks up to 2 MiB come in power-of-two sizes from 256 bytes, and a larger block gets a chunk of its own, rounded up to a multiple of 2 MiB.
/// Dropping a buffer returns its block to the pool without a CUDA call, so it is cheap and allowed inside a stream callback.
/// [`PinnedHost::release_cached`] returns unused chunks to CUDA.
/// Chunks are portable and mapped, so any context can copy from them.
#[derive(Debug, Clone)]
pub struct PinnedHost(Arc<Context>);

/// A vector in page-locked host memory, which CUDA copies read and write without staging.
pub type PinnedVec<T> = Vec<T, PinnedHost>;

impl Context {
    /// Returns the allocator for this context's pinned host memory pool.
    #[must_use]
    pub fn pinned_host(self: &Arc<Self>) -> PinnedHost {
        PinnedHost(Arc::clone(self))
    }
}

impl PinnedHost {
    /// Returns the context whose pool this allocator uses.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.0
    }

    /// Allocates an empty vector with room for exactly `capacity` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte size overflows, `T` needs an alignment above 4096 bytes, the context cannot be bound, or CUDA cannot allocate page-locked memory.
    pub fn with_capacity<T>(&self, capacity: usize) -> Result<PinnedVec<T>> {
        let layout = Layout::array::<T>(capacity)
            .map_err(|_| Error::out_of_range("pinned vector capacity"))?;
        let block = self.allocate_block(layout)?;
        // SAFETY: the block holds `capacity` values of `T`, allocated by this allocator with `T`'s array layout.
        Ok(
            unsafe {
                Vec::from_raw_parts_in(block.cast::<T>().as_ptr(), 0, capacity, self.clone())
            },
        )
    }

    /// Allocates a vector holding a copy of `values`.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails, for the reasons [`PinnedHost::with_capacity`] gives.
    pub fn from_slice<T: Copy>(&self, values: &[T]) -> Result<PinnedVec<T>> {
        let mut vector = self.with_capacity(values.len())?;
        vector.extend_from_slice(values);
        Ok(vector)
    }

    /// Allocates a vector of `length` zero values.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails, for the reasons [`PinnedHost::with_capacity`] gives.
    pub fn zeroed<T: ZeroableDeviceRepr>(&self, length: usize) -> Result<PinnedVec<T>> {
        let mut vector = self.with_capacity::<T>(length)?;
        // SAFETY: the capacity holds `length` values, and every all-zero `T` is valid.
        unsafe {
            ptr::write_bytes(vector.as_mut_ptr(), 0, length);
            vector.set_len(length);
        }
        Ok(vector)
    }

    /// Frees the pool's chunks whose blocks are all free.
    ///
    /// Chunks with a live block stay.
    /// The pool also frees idle single-block chunks on its own, when an allocation finds no free block of its size.
    ///
    /// # Errors
    ///
    /// Returns an error if the context cannot be bound or CUDA fails to free a chunk; every idle chunk leaves the pool either way.
    pub fn release_cached(&self) -> Result<()> {
        self.0.pinned.release_cached(&self.0)
    }

    /// Returns how many bytes and chunks the pool holds, and how many bytes are in use.
    #[must_use]
    pub fn stats(&self) -> PinnedStats {
        self.0.pinned.stats()
    }

    pub(crate) fn host_region(&self, pointer: *const u8, bytes: usize) -> HostRegion<'_> {
        self.0.pinned.host_region(&self.0, pointer, bytes)
    }

    pub(crate) fn device_region(
        &self,
        pointer: *const u8,
        bytes: usize,
    ) -> Result<DeviceRegion<'_>> {
        self.0.pinned.device_region(&self.0, pointer, bytes)
    }

    // Requires what `Allocator::grow` or `Allocator::shrink` requires: `block` is live from this allocator with layout `old`.
    unsafe fn resize(
        &self,
        block: NonNull<u8>,
        old: Layout,
        new: Layout,
        zeroed: bool,
    ) -> StdResult<NonNull<[u8]>, AllocError> {
        // A block keeps serving a layout of its own class, and `deallocate` with the new layout finds that class.
        if old.size() != 0 && new.size() != 0 && same_class(old, new) {
            if zeroed && new.size() > old.size() {
                // SAFETY: the block's class covers `new.size()` bytes.
                unsafe {
                    ptr::write_bytes(block.add(old.size()).as_ptr(), 0, new.size() - old.size());
                };
            }
            return Ok(NonNull::slice_from_raw_parts(block, new.size()));
        }

        let moved = if zeroed {
            self.allocate_zeroed(new)?
        } else {
            self.allocate(new)?
        };
        // SAFETY: both blocks hold at least the smaller size, and distinct live blocks never overlap.
        unsafe {
            ptr::copy_nonoverlapping(
                block.as_ptr(),
                moved.cast::<u8>().as_ptr(),
                old.size().min(new.size()),
            );
            self.deallocate(block, old);
        }
        Ok(moved)
    }

    fn allocate_block(&self, layout: Layout) -> Result<NonNull<u8>> {
        if layout.size() == 0 {
            return NonNull::new(ptr::without_provenance_mut(layout.align()))
                .ok_or(Error::NullHandle);
        }
        self.0.pinned.allocate(&self.0, layout)
    }
}

// SAFETY: every clone shares one context and so one pool, which outlives every block because each block's owner holds the context.
// Blocks are disjoint pieces of page-locked host allocations, and the pool neither reads them nor unwinds.
/// Allocates blocks from the context's pool.
/// A zero-sized layout takes no memory.
/// Growing or shrinking keeps a block in place while the new layout falls in the block's size class.
unsafe impl Allocator for PinnedHost {
    fn allocate(&self, layout: Layout) -> StdResult<NonNull<[u8]>, AllocError> {
        let block = self.allocate_block(layout).map_err(|_| AllocError)?;
        Ok(NonNull::slice_from_raw_parts(block, layout.size()))
    }

    unsafe fn deallocate(&self, block: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            self.0.pinned.deallocate(block, layout);
        }
    }

    unsafe fn grow(
        &self,
        block: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> StdResult<NonNull<[u8]>, AllocError> {
        // SAFETY: `grow`'s caller passes a live `block` of this allocator with layout `old`, which is what `resize` requires.
        unsafe { self.resize(block, old, new, false) }
    }

    unsafe fn grow_zeroed(
        &self,
        block: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> StdResult<NonNull<[u8]>, AllocError> {
        // SAFETY: `grow_zeroed`'s caller passes a live `block` of this allocator with layout `old`, which is what `resize` requires.
        unsafe { self.resize(block, old, new, true) }
    }

    unsafe fn shrink(
        &self,
        block: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> StdResult<NonNull<[u8]>, AllocError> {
        // SAFETY: `shrink`'s caller passes a live `block` of this allocator with layout `old`, which is what `resize` requires.
        unsafe { self.resize(block, old, new, false) }
    }
}
