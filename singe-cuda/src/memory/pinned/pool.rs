use std::{
    alloc::Layout,
    collections::BTreeMap,
    fmt::{self, Debug, Formatter},
    mem,
    ptr::{self, NonNull},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::free_host_raw,
    try_ffi,
    view::{AllocationId, HostRegion},
};

const SMALLEST_BLOCK: usize = 256;

const LARGEST_ALIGNMENT: usize = 4096;

const CHUNK_BYTES: usize = 2 << 20;

#[derive(Default)]
pub struct PinnedPool {
    state: Mutex<PoolState>,
}

/// Stats for context's pinned host memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PinnedStats {
    /// Bytes the pool holds from CUDA, in free and live blocks.
    pub reserved_bytes: usize,
    /// Bytes in live blocks, each counted at its size class.
    pub active_bytes: usize,
    /// Allocations the pool holds from CUDA.
    pub chunks: usize,
}

#[derive(Debug, Default)]
struct PoolState {
    chunks: BTreeMap<usize, Chunk>,
    free: BTreeMap<usize, Vec<NonNull<u8>>>,
    active_bytes: usize,
}

#[derive(Debug)]
struct Chunk {
    base: NonNull<u8>,
    bytes: usize,
    class: usize,
    id: AllocationId,
    free_blocks: usize,
}

// SAFETY: the pool's mutex guards this state, and the pool never dereferences its pointers.
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "block pointers are addresses the pool hands out, never dereferenced"
)]
unsafe impl Send for PoolState {}

impl Debug for PinnedPool {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        // The free lists hold thousands of pointers, which would flood the `Debug` output of every `Context`.
        formatter
            .debug_struct("PinnedPool")
            .field("stats", &self.stats())
            .finish()
    }
}

impl PinnedPool {
    pub(crate) fn allocate(&self, context: &Context, layout: Layout) -> Result<NonNull<u8>> {
        let class = class(layout)?;
        let cached = self.lock().take(class);
        if let Some(block) = cached {
            return Ok(block);
        }

        // CUDA runs outside the lock, since `cudaFreeHost` may wait for a stream callback that drops a pinned buffer.
        context.bind()?;
        // A failed free leaves the chunk out of the pool, and this allocation does not depend on it.
        let dedicated = self.lock().take_idle(Chunk::is_dedicated);
        let _ = free_chunks(dedicated);
        let chunk = if let Ok(chunk) = Chunk::allocate(class) {
            chunk
        } else {
            let idle = self.lock().take_idle(|_| true);
            let _ = free_chunks(idle);
            Chunk::allocate(class)?
        };

        let mut state = self.lock();
        state.insert(chunk);
        state
            .take(class)
            .ok_or(Error::InvalidMemoryAllocationRequest)
    }

    pub(crate) fn deallocate(&self, block: NonNull<u8>, layout: Layout) {
        // Never calls CUDA, so dropping a pinned buffer is cheap and works inside a stream callback.
        if let Ok(class) = class(layout) {
            self.lock().give_back(block, class);
        }
    }

    pub(crate) fn release_cached(&self, context: &Context) -> Result<()> {
        context.bind()?;
        let idle = self.lock().take_idle(|_| true);
        free_chunks(idle)
    }

    pub(crate) fn release_all(&self, context: &Context) -> Result<()> {
        let chunks = {
            let mut state = self.lock();
            debug_assert_eq!(
                state.active_bytes, 0,
                "pinned blocks outlived their context"
            );
            state.free.clear();
            mem::take(&mut state.chunks)
        };
        if chunks.is_empty() {
            return Ok(());
        }

        context.bind()?;
        free_chunks(chunks.into_values().collect())
    }

    pub(crate) fn stats(&self) -> PinnedStats {
        let state = self.lock();
        PinnedStats {
            reserved_bytes: state.chunks.values().map(|chunk| chunk.bytes).sum(),
            active_bytes: state.active_bytes,
            chunks: state.chunks.len(),
        }
    }

    pub(crate) fn host_region<'a>(
        &self,
        context: &'a Arc<Context>,
        pointer: *const u8,
        bytes: usize,
    ) -> HostRegion<'a> {
        let chunk = self
            .lock()
            .chunk_holding(pointer.addr())
            .map(|chunk| (chunk.id, chunk.bytes, chunk.base.addr().get()));
        let Some((id, chunk_bytes, base)) = chunk else {
            // SAFETY: an empty range has no bytes for any alias to share.
            return unsafe { HostRegion::from_raw_parts(context, AllocationId::create(), 0, true) };
        };

        let offset = pointer.addr() - base;
        // SAFETY: the chunk stays allocated while the context lives, and it is portable and mapped.
        unsafe { HostRegion::from_raw_parts(context, id, chunk_bytes, true) }
            .slice(offset..offset + bytes)
            .expect("pinned block lies inside its chunk")
    }

    fn lock(&self) -> MutexGuard<'_, PoolState> {
        // A panic elsewhere must not make `deallocate` unwind, and the state stays consistent at every unlock.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl PoolState {
    fn take(&mut self, class: usize) -> Option<NonNull<u8>> {
        let block = self.free.get_mut(&class)?.pop()?;
        if let Some(chunk) = self.chunk_holding_mut(block.addr().get()) {
            chunk.free_blocks -= 1;
        }

        self.active_bytes += class;
        Some(block)
    }

    fn give_back(&mut self, block: NonNull<u8>, class: usize) {
        if let Some(chunk) = self.chunk_holding_mut(block.addr().get()) {
            chunk.free_blocks += 1;
        }

        self.active_bytes -= class;
        self.free.entry(class).or_default().push(block);
    }

    fn insert(&mut self, chunk: Chunk) {
        let blocks = self.free.entry(chunk.class).or_default();
        blocks.extend((0..chunk.free_blocks).map(|index| {
            // SAFETY: every block offset lies inside the chunk.
            unsafe { chunk.base.add(index * chunk.class) }
        }));
        self.chunks.insert(chunk.base.addr().get(), chunk);
    }

    fn take_idle(&mut self, selected: impl Fn(&Chunk) -> bool) -> Vec<Chunk> {
        let idle = self
            .chunks
            .extract_if(.., |_, chunk| chunk.is_idle() && selected(chunk))
            .map(|(_, chunk)| chunk)
            .collect::<Vec<_>>();
        for chunk in &idle {
            if let Some(blocks) = self.free.get_mut(&chunk.class) {
                blocks.retain(|block| !chunk.holds(block.addr().get()));
            }
        }
        idle
    }

    fn chunk_holding(&self, address: usize) -> Option<&Chunk> {
        let (_, chunk) = self.chunks.range(..=address).next_back()?;
        chunk.holds(address).then_some(chunk)
    }

    fn chunk_holding_mut(&mut self, address: usize) -> Option<&mut Chunk> {
        let (_, chunk) = self.chunks.range_mut(..=address).next_back()?;
        chunk.holds(address).then_some(chunk)
    }
}

impl Chunk {
    fn allocate(class: usize) -> Result<Self> {
        let bytes = class.max(CHUNK_BYTES);
        let mut pointer = ptr::null_mut();
        // SAFETY: `cudaHostAlloc` receives a valid output location and a nonzero size.
        unsafe {
            try_ffi!(runtime::cudaHostAlloc(
                &raw mut pointer,
                bytes as _,
                runtime::cudaHostAllocPortable | runtime::cudaHostAllocMapped,
            ))?;
        }

        let chunk = Self {
            base: NonNull::new(pointer.cast::<u8>()).ok_or(Error::NullHandle)?,
            bytes,
            class,
            id: AllocationId::create(),
            free_blocks: bytes / class,
        };
        // Block alignment rests on page-aligned chunks, which CUDA gives in practice but does not document.
        if !chunk.base.addr().get().is_multiple_of(LARGEST_ALIGNMENT) {
            chunk.free()?;
            return Err(Error::UnsupportedPinnedAlignment {
                align: LARGEST_ALIGNMENT,
            });
        }
        Ok(chunk)
    }

    const fn is_idle(&self) -> bool {
        self.free_blocks == self.bytes / self.class
    }

    const fn is_dedicated(&self) -> bool {
        // A chunk of one block is released when idle at the next miss, so large buffers do not stay pinned for the context's life.
        self.class >= CHUNK_BYTES
    }

    fn holds(&self, address: usize) -> bool {
        (self.base.addr().get()..self.base.addr().get() + self.bytes).contains(&address)
    }

    fn free(&self) -> Result<()> {
        // SAFETY: the chunk came from `cudaHostAlloc`, its pool frees it once, and none of its blocks is live.
        unsafe { free_host_raw(self.base.as_ptr().cast()) }
    }
}

fn free_chunks(chunks: Vec<Chunk>) -> Result<()> {
    let mut result = Ok(());
    for chunk in chunks {
        if let Err(error) = chunk.free() {
            result = Err(error);
        }
    }
    result
}

fn class(layout: Layout) -> Result<usize> {
    // A layout always maps to the same class, since `deallocate` receives the layout `allocate` did.
    // Small classes are powers of two, so a block is aligned to its class up to a page; large ones round to whole chunks, wasting less than a power of two would.
    if layout.align() > LARGEST_ALIGNMENT {
        return Err(Error::UnsupportedPinnedAlignment {
            align: layout.align(),
        });
    }

    let bytes = layout.size().max(layout.align()).max(SMALLEST_BLOCK);
    if bytes > CHUNK_BYTES {
        bytes.checked_next_multiple_of(CHUNK_BYTES)
    } else {
        bytes.checked_next_power_of_two()
    }
    .ok_or(Error::InvalidMemoryAllocationRequest)
}
