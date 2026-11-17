//! Device chunk tables and status buffers for batched nvCOMP operations.

use std::{marker::PhantomData, mem::MaybeUninit, ptr, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    memory::DeviceMemory,
    stream::Stream,
    submission::{Access, AccessMode},
    types::DevicePtr,
    view::{
        DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegion, DeviceRegionProvider,
        DeviceSlice, DeviceView,
    },
};

use crate::{
    error::{DeviceStatus, Error, Result, Status},
    utility::{is_alignment_satisfied, nonempty_ptr, validate_alignment},
};

/// Device arrays of chunk pointers and sizes that describe the input of a batched nvCOMP operation.
///
/// The table borrows the buffers it describes, so they outlive it.
/// Its chunks are initialized: the safe constructors take initialized device slices, and [`create`](Self::create) requires initialized regions of its caller.
#[derive(Debug)]
pub struct DeviceChunkTable<'a> {
    device_chunk_ptrs: DeviceMemory<DevicePtr>,
    device_chunk_bytes: DeviceMemory<usize>,
    regions: Vec<DeviceRegion<'a>>,
    max_chunk_bytes: usize,
    total_bytes: usize,
    ptr_alignment: usize,
}

impl<'a> DeviceChunkTable<'a> {
    /// Describes all of `buffer` as one chunk.
    ///
    /// The device tables are allocated in `context`.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous<B: DeviceSlice<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a B,
    ) -> Result<Self> {
        Self::from_regions(context, vec![checked_region(context, buffer)?])
    }

    /// Splits `buffer` into chunks of `chunk_bytes` bytes; the last chunk may be shorter.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunk_bytes` is zero, a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous_chunks<B: DeviceSlice<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a B,
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::from_regions(
            context,
            split_regions(checked_region(context, buffer)?, chunk_bytes)?,
        )
    }

    /// Describes consecutive chunks of `buffer` with the sizes in `chunk_bytes`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sizes add up to more than the buffer length, a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous_chunk_sizes<B: DeviceSlice<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a B,
        chunk_bytes: &[usize],
    ) -> Result<Self> {
        Self::from_regions(
            context,
            adjacent_regions(checked_region(context, buffer)?, chunk_bytes)?,
        )
    }

    /// Describes one chunk at the start of each `stride_bytes`-byte slot of `buffer`, with the sizes in `chunk_bytes`.
    ///
    /// # Errors
    ///
    /// Returns an error if `stride_bytes` is zero while `chunk_bytes` is not empty, a chunk is larger than its slot, the last chunk ends past the buffer, a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous_strided_chunks<B: DeviceSlice<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a B,
        stride_bytes: usize,
        chunk_bytes: &[usize],
    ) -> Result<Self> {
        Self::from_regions(
            context,
            strided_regions(checked_region(context, buffer)?, stride_bytes, chunk_bytes)?,
        )
    }

    /// Describes each buffer in `buffers` as one chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_device_slices<B: DeviceSlice<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffers: &[&'a B],
    ) -> Result<Self> {
        let regions = buffers
            .iter()
            .map(|buffer| checked_region(context, *buffer))
            .collect::<Result<Vec<_>>>()?;
        Self::from_regions(context, regions)
    }

    /// Describes each region in `regions` as one chunk.
    ///
    /// # Safety
    ///
    /// Every byte of every region must be initialized whenever an operation that reads the table runs, including every replay of a captured graph.
    ///
    /// # Errors
    ///
    /// Returns an error if a region belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub unsafe fn create(
        context: &Arc<CudaContext>,
        regions: Vec<DeviceRegion<'a>>,
    ) -> Result<Self> {
        Self::from_regions(context, regions)
    }

    fn from_regions(context: &Arc<CudaContext>, regions: Vec<DeviceRegion<'a>>) -> Result<Self> {
        let (ptrs, sizes, ptr_alignment) = table_metadata(context, &regions)?;
        Ok(Self {
            device_chunk_ptrs: DeviceMemory::from_slice(context, &ptrs)?,
            device_chunk_bytes: DeviceMemory::from_slice(context, &sizes)?,
            max_chunk_bytes: sizes.iter().copied().max().unwrap_or(0),
            total_bytes: total_chunk_bytes(&sizes)?,
            regions,
            ptr_alignment,
        })
    }

    pub(crate) fn read_accesses(&self) -> Vec<Access> {
        let mut accesses = self
            .regions
            .iter()
            .map(|region| Access::new(*region, AccessMode::Read))
            .collect::<Vec<_>>();
        if !self.is_empty() {
            accesses.push(Access::new(
                self.device_chunk_ptrs.region(),
                AccessMode::Read,
            ));
            accesses.push(Access::new(
                self.device_chunk_bytes.region(),
                AccessMode::Read,
            ));
        }
        accesses
    }

    pub(crate) fn require_element_size(&self, bytes: usize) -> Result<()> {
        if self
            .regions
            .iter()
            .any(|region| !region.byte_len().is_multiple_of(bytes))
        {
            return Err(Error::out_of_range("compression chunk element count"));
        }
        Ok(())
    }

    /// Returns the CUDA context of the chunks and tables.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.device_chunk_ptrs.context()
    }

    /// Returns a view of the device tables for raw nvCOMP calls.
    #[must_use]
    pub fn view(&self) -> DeviceChunkTableView<'_> {
        DeviceChunkTableView {
            device_chunk_ptrs: nonempty_ptr(&self.device_chunk_ptrs),
            device_chunk_bytes: nonempty_ptr(&self.device_chunk_bytes),
            regions: &self.regions,
            context: self.context(),
            ptr_alignment: self.ptr_alignment,
        }
    }

    /// Returns the number of chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Returns whether the table has no chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Returns the largest chunk size in bytes, or 0 for an empty table.
    #[must_use]
    pub const fn max_chunk_bytes(&self) -> usize {
        self.max_chunk_bytes
    }

    /// Returns the sum of the chunk sizes in bytes.
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Returns whether every chunk starts at a multiple of `alignment` bytes.
    ///
    /// Empty tables and alignments of 0 or 1 always return `true`.
    /// Chunk sizes are not checked.
    #[must_use]
    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        is_alignment_satisfied(self.len(), self.ptr_alignment, alignment)
    }
}

/// Device arrays of chunk pointers, capacities, and written sizes that describe the output of a batched nvCOMP operation.
///
/// The table borrows the buffers it describes mutably, so nothing else can use them while it exists.
/// Neither the chunks nor the written sizes are initialized until an operation fills them.
#[derive(Debug)]
pub struct DeviceChunkTableMut<'a> {
    device_chunk_ptrs: DeviceMemory<DevicePtr>,
    device_buffer_bytes: DeviceMemory<usize>,
    device_chunk_bytes: DeviceMemory<MaybeUninit<usize>>,
    regions: Vec<DeviceRegion<'a>>,
    min_buffer_bytes: usize,
    max_buffer_bytes: usize,
    total_buffer_bytes: usize,
    ptr_alignment: usize,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> DeviceChunkTableMut<'a> {
    /// Describes all of `buffer` as one output chunk.
    ///
    /// Creating the table initializes neither the output chunks nor the device array that receives their written sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous<B: DeviceOutput<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a mut B,
    ) -> Result<Self> {
        Self::from_regions(context, vec![checked_region(context, buffer)?])
    }

    /// Splits `buffer` into output chunks of `chunk_bytes` bytes; the last chunk may be shorter.
    ///
    /// Creating the table initializes neither the output chunks nor the device array that receives their written sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if `chunk_bytes` is zero, a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous_chunks<B: DeviceOutput<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a mut B,
        chunk_bytes: usize,
    ) -> Result<Self> {
        Self::from_regions(
            context,
            split_regions(checked_region(context, buffer)?, chunk_bytes)?,
        )
    }

    /// Describes consecutive output chunks of `buffer` with the capacities in `chunk_bytes`.
    ///
    /// Creating the table initializes neither the output chunks nor the device array that receives their written sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the capacities add up to more than the buffer length, a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_contiguous_chunk_sizes<B: DeviceOutput<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffer: &'a mut B,
        chunk_bytes: &[usize],
    ) -> Result<Self> {
        Self::from_regions(
            context,
            adjacent_regions(checked_region(context, buffer)?, chunk_bytes)?,
        )
    }

    /// Describes each buffer in `buffers` as one output chunk.
    ///
    /// Creating the table initializes neither the output chunks nor the device array that receives their written sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer belongs to a CUDA context other than `context`, a size overflows, or the device tables cannot be allocated or uploaded.
    pub fn from_device_slices<B: DeviceOutput<u8> + ?Sized>(
        context: &Arc<CudaContext>,
        buffers: &'a mut [&'a mut B],
    ) -> Result<Self> {
        let regions = buffers
            .iter()
            .map(|buffer| checked_region(context, &**buffer))
            .collect::<Result<Vec<_>>>()?;
        Self::from_regions(context, regions)
    }

    fn from_regions(context: &Arc<CudaContext>, regions: Vec<DeviceRegion<'a>>) -> Result<Self> {
        let (ptrs, sizes, ptr_alignment) = table_metadata(context, &regions)?;
        Ok(Self {
            device_chunk_ptrs: DeviceMemory::from_slice(context, &ptrs)?,
            device_buffer_bytes: DeviceMemory::from_slice(context, &sizes)?,
            device_chunk_bytes: DeviceMemory::<usize>::create_uninit(context, regions.len())?,
            min_buffer_bytes: sizes.iter().copied().min().unwrap_or(0),
            max_buffer_bytes: sizes.iter().copied().max().unwrap_or(0),
            total_buffer_bytes: total_chunk_bytes(&sizes)?,
            regions,
            ptr_alignment,
            _marker: PhantomData,
        })
    }

    /// Returns the CUDA context of the chunks and tables.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.device_chunk_ptrs.context()
    }

    /// Returns a view of the chunk pointers and capacities for raw nvCOMP calls.
    #[must_use]
    pub fn buffers(&self) -> DeviceChunkTableView<'_> {
        DeviceChunkTableView {
            device_chunk_ptrs: nonempty_ptr(&self.device_chunk_ptrs),
            device_chunk_bytes: nonempty_ptr(&self.device_buffer_bytes),
            regions: &self.regions,
            context: self.context(),
            ptr_alignment: self.ptr_alignment,
        }
    }

    /// Returns a view of the chunk pointers and written sizes for raw nvCOMP calls.
    ///
    /// The written sizes are valid only after an operation that fills them has finished.
    #[must_use]
    pub fn written_chunks(&self) -> DeviceChunkTableView<'_> {
        DeviceChunkTableView {
            device_chunk_bytes: nonempty_ptr(&self.device_chunk_bytes),
            ..self.buffers()
        }
    }

    /// Returns the device array that receives each chunk's written size, or null for an empty table.
    ///
    /// Read it only after an operation that fills it has finished.
    #[must_use]
    pub fn written_bytes_device_ptr(&self) -> DevicePtr {
        nonempty_ptr(&self.device_chunk_bytes)
    }

    /// Synchronizes `stream` and returns each chunk's written size.
    ///
    /// # Safety
    ///
    /// An operation ordered on `stream` must have written every size, and nothing may write to them during the copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the table and `stream` belong to different CUDA contexts, stream capture is active or cannot be queried, synchronization or the copy fails, or a written size exceeds its chunk's capacity.
    pub unsafe fn copy_written_bytes(&self, stream: &Stream) -> Result<Vec<usize>> {
        self.buffers().validate_context(stream)?;
        stream.ensure_not_capturing()?;
        stream.synchronize()?;
        // SAFETY: the caller guarantees work on `stream` wrote every size slot, and the synchronization above finished it.
        let sizes = unsafe {
            DeviceView::<usize>::from_raw_parts(self.device_chunk_bytes.region(), self.len())?
        }
        .copy_to_host_vec()?;
        for (index, (&actual, region)) in sizes.iter().zip(&self.regions).enumerate() {
            if actual > region.byte_len() {
                return Err(Error::ChunkLengthMismatch {
                    index,
                    name: "written chunk bytes".into(),
                    expected: region.byte_len(),
                    actual,
                });
            }
        }
        Ok(sizes)
    }

    pub(crate) fn write_accesses(&self) -> Vec<Access> {
        let mut accesses = self
            .regions
            .iter()
            .map(|region| Access::new(*region, AccessMode::Write))
            .collect::<Vec<_>>();
        if !self.is_empty() {
            accesses.push(Access::new(
                self.device_chunk_ptrs.region(),
                AccessMode::Read,
            ));
            accesses.push(Access::new(
                self.device_buffer_bytes.region(),
                AccessMode::Read,
            ));
            accesses.push(Access::new(
                self.device_chunk_bytes.region(),
                AccessMode::Write,
            ));
        }
        accesses
    }

    /// Returns the number of chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Returns whether the table has no chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Returns the largest chunk capacity in bytes, or 0 for an empty table.
    #[must_use]
    pub const fn max_buffer_bytes(&self) -> usize {
        self.max_buffer_bytes
    }

    /// Returns the smallest chunk capacity in bytes, or 0 for an empty table.
    #[must_use]
    pub const fn min_buffer_bytes(&self) -> usize {
        self.min_buffer_bytes
    }

    /// Returns the sum of the chunk capacities in bytes.
    #[must_use]
    pub const fn total_buffer_bytes(&self) -> usize {
        self.total_buffer_bytes
    }

    /// Returns whether every chunk starts at a multiple of `alignment` bytes.
    ///
    /// Empty tables and alignments of 0 or 1 always return `true`.
    /// Capacities are not checked.
    #[must_use]
    pub fn is_aligned_to(&self, alignment: usize) -> bool {
        is_alignment_satisfied(self.len(), self.ptr_alignment, alignment)
    }
}

/// Borrowed view of a chunk table's device arrays, for raw nvCOMP calls.
///
/// The view does not claim that the chunks are initialized, and it grants no write access.
#[derive(Debug, Clone, Copy)]
pub struct DeviceChunkTableView<'a> {
    device_chunk_ptrs: DevicePtr,
    device_chunk_bytes: DevicePtr,
    regions: &'a [DeviceRegion<'a>],
    context: &'a Arc<CudaContext>,
    ptr_alignment: usize,
}

impl<'a> DeviceChunkTableView<'a> {
    /// Returns the device array of chunk pointers, or null for an empty table.
    #[must_use]
    pub const fn device_chunk_ptrs(self) -> DevicePtr {
        self.device_chunk_ptrs
    }

    /// Returns the device array of chunk sizes, or null for an empty table.
    ///
    /// For an input table these are the chunk sizes.
    /// For [`DeviceChunkTableMut::buffers`] they are capacities, and for [`DeviceChunkTableMut::written_chunks`] they are written sizes, valid only after an operation fills them.
    #[must_use]
    pub const fn device_chunk_bytes(self) -> DevicePtr {
        self.device_chunk_bytes
    }

    /// Returns the CUDA context of the chunks and tables.
    #[must_use]
    pub fn context(self) -> &'a Arc<CudaContext> {
        self.context
    }

    /// Returns the borrowed memory regions of the chunks.
    ///
    /// For an output table, the regions cover the full capacities, not the written sizes.
    #[must_use]
    pub fn regions(self) -> &'a [DeviceRegion<'a>] {
        self.regions
    }

    /// Returns the number of chunks.
    #[must_use]
    pub fn len(self) -> usize {
        self.regions.len()
    }

    /// Returns whether the table has no chunks.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.regions.is_empty()
    }

    /// Returns whether every chunk starts at a multiple of `alignment` bytes.
    ///
    /// Empty tables and alignments of 0 or 1 always return `true`.
    #[must_use]
    pub fn is_aligned_to(self, alignment: usize) -> bool {
        is_alignment_satisfied(self.len(), self.ptr_alignment, alignment)
    }

    pub(crate) fn validate_context(self, stream: &Stream) -> Result<()> {
        if self.context != stream.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        Ok(())
    }

    pub(crate) fn join_regions(self, stream: &Stream) -> Result<()> {
        self.validate_context(stream)?;
        Ok(stream.join_regions(self.regions)?)
    }
}

fn checked_region<'a, B: DeviceRange<u8> + ?Sized>(
    context: &Arc<CudaContext>,
    buffer: &'a B,
) -> Result<DeviceRegion<'a>> {
    let region = buffer.region().slice(..buffer.len())?;
    if region.context() != context {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    Ok(region)
}

fn table_metadata(
    context: &Arc<CudaContext>,
    regions: &[DeviceRegion<'_>],
) -> Result<(Vec<DevicePtr>, Vec<usize>, usize)> {
    let mut ptrs = Vec::with_capacity(regions.len());
    let mut sizes = Vec::with_capacity(regions.len());
    let mut alignment = 0;
    for region in regions {
        if region.context() != context {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        // SAFETY: the pointer comes from a region checked to be in `context`, whose borrow keeps the allocation alive.
        ptrs.push(unsafe { DevicePtr::from_raw(region.as_ptr().cast()) });
        sizes.push(region.byte_len());
        alignment = common_alignment(alignment, region.as_ptr().addr());
    }
    Ok((ptrs, sizes, alignment))
}

fn split_regions(region: DeviceRegion<'_>, chunk_bytes: usize) -> Result<Vec<DeviceRegion<'_>>> {
    if chunk_bytes == 0 {
        return Err(Error::NotPositive {
            name: "chunk bytes".into(),
        });
    }
    (0..region.byte_len())
        .step_by(chunk_bytes)
        .map(|start| {
            Ok(region.slice(start..start.saturating_add(chunk_bytes).min(region.byte_len()))?)
        })
        .collect()
}

fn adjacent_regions<'a>(
    region: DeviceRegion<'a>,
    sizes: &[usize],
) -> Result<Vec<DeviceRegion<'a>>> {
    let total = total_chunk_bytes(sizes)?;
    if total > region.byte_len() {
        return Err(Error::BufferTooSmall {
            name: "chunk bytes".into(),
            required: total,
            actual: region.byte_len(),
        });
    }

    let mut offset = 0usize;
    let mut regions = Vec::with_capacity(sizes.len());
    for &size in sizes {
        let end = offset.checked_add(size).ok_or_else(|| Error::OutOfRange {
            name: "chunk offset".into(),
        })?;
        regions.push(region.slice(offset..end)?);
        offset = end;
    }
    Ok(regions)
}

fn strided_regions<'a>(
    region: DeviceRegion<'a>,
    stride: usize,
    sizes: &[usize],
) -> Result<Vec<DeviceRegion<'a>>> {
    let required = required_strided_bytes(stride, sizes)?;
    if stride == 0 && !sizes.is_empty() {
        return Err(Error::NotPositive {
            name: "stride bytes".into(),
        });
    }

    if required > region.byte_len() {
        return Err(Error::BufferTooSmall {
            name: "strided buffer bytes".into(),
            required,
            actual: region.byte_len(),
        });
    }
    sizes
        .iter()
        .enumerate()
        .map(|(index, &size)| {
            if size > stride {
                return Err(Error::BufferTooSmall {
                    name: "stride bytes".into(),
                    required: size,
                    actual: stride,
                });
            }

            let start = index * stride;
            Ok(region.slice(start..start + size)?)
        })
        .collect()
}

fn common_alignment(current: usize, address: usize) -> usize {
    if current == 0 {
        return address;
    }

    gcd(current, address)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }

    a
}

fn required_strided_bytes(stride_bytes: usize, sizes: &[usize]) -> Result<usize> {
    let Some(&last) = sizes.last() else {
        return Ok(0);
    };
    (sizes.len() - 1)
        .checked_mul(stride_bytes)
        .and_then(|offset| offset.checked_add(last))
        .ok_or_else(|| Error::OutOfRange {
            name: "strided buffer bytes".into(),
        })
}

fn total_chunk_bytes(chunk_bytes: &[usize]) -> Result<usize> {
    chunk_bytes.iter().try_fold(0usize, |total, bytes| {
        total.checked_add(*bytes).ok_or_else(|| Error::OutOfRange {
            name: "total chunk bytes".into(),
        })
    })
}

/// Device buffer with one [`DeviceStatus`] per chunk, which batched nvCOMP operations fill.
#[derive(Debug)]
pub struct DeviceStatuses {
    statuses: DeviceMemory<MaybeUninit<DeviceStatus>>,
}

impl DeviceStatuses {
    /// Allocates an uninitialized status buffer with `chunks` slots in `allocation_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer size overflows or the allocation fails.
    pub fn create(allocation_context: &Arc<CudaContext>, chunks: usize) -> Result<Self> {
        Ok(Self {
            statuses: DeviceMemory::<DeviceStatus>::create_uninit(allocation_context, chunks)?,
        })
    }

    /// Uses an existing device allocation as the status buffer, one slot per element.
    #[must_use]
    pub fn from_memory(statuses: DeviceMemory<MaybeUninit<DeviceStatus>>) -> Self {
        Self { statuses }
    }

    /// Returns the CUDA context of the status buffer.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.statuses.context()
    }

    /// Returns the number of status slots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.statuses.len()
    }

    /// Returns whether the buffer has no slots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.statuses.is_empty()
    }

    /// Checks that the buffer has exactly `chunks` slots.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LengthMismatch`] if the slot count differs from `chunks`.
    pub fn require(&self, chunks: usize) -> Result<()> {
        if self.len() != chunks {
            return Err(Error::LengthMismatch {
                name: "device statuses".into(),
                expected: chunks,
                actual: self.len(),
            });
        }

        Ok(())
    }

    pub(crate) fn validate_context(&self, stream: &Stream) -> Result<()> {
        if self.context() != stream.context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }
        Ok(())
    }

    pub(crate) fn region(&self) -> DeviceRegion<'_> {
        self.statuses.region()
    }

    /// Returns the device pointer to pass to nvCOMP.
    pub fn as_device_ptr(&mut self) -> DevicePtr {
        if self.is_empty() {
            DevicePtr::null()
        } else {
            self.statuses.as_device_ptr()
        }
    }

    /// Synchronizes `stream`, copies the statuses to the host, and checks them.
    ///
    /// # Safety
    ///
    /// An operation ordered on `stream` must have written every slot, and nothing may write to them during the copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer and `stream` belong to different CUDA contexts, stream capture is active or cannot be queried, synchronization or the copy fails, or a chunk reports an error status.
    pub unsafe fn check(&self, stream: &Stream) -> Result<Vec<Status>> {
        self.validate_context(stream)?;

        stream.ensure_not_capturing()?;
        stream.synchronize()?;
        // SAFETY: the caller guarantees work on `stream` wrote every status slot, and the synchronization above finished it.
        unsafe { self.copy_and_check_completed() }
    }

    unsafe fn copy_and_check_completed(&self) -> Result<Vec<Status>> {
        // SAFETY: the caller guarantees every status slot was written by finished work.
        let view = unsafe {
            DeviceView::<DeviceStatus>::from_raw_parts(self.statuses.region(), self.len())?
        };
        let statuses = view
            .copy_to_host_vec()?
            .into_iter()
            .map(Status::from)
            .collect::<Vec<_>>();

        for (index, status) in statuses.iter().copied().enumerate() {
            if status != Status::Success {
                return Err(Error::OperationFailed {
                    operation: format!("chunk {index}").into(),
                    status,
                });
            }
        }

        Ok(statuses)
    }

    /// Returns the underlying device allocation.
    #[must_use]
    pub fn into_device_memory(self) -> DeviceMemory<MaybeUninit<DeviceStatus>> {
        self.statuses
    }
}

/// Handle to an enqueued batched operation, used to wait for it and check its statuses.
///
/// The handle borrows the status buffer and stream.
/// Dropping it neither waits for the work nor keeps the operation's buffers alive.
#[derive(Debug)]
#[must_use = "call check() to complete the launch and validate its device statuses"]
pub struct AsyncBatchResult<'a> {
    statuses: &'a DeviceStatuses,
    stream: &'a Stream,
}

impl<'a> AsyncBatchResult<'a> {
    /// Creates a handle for work on `stream` that fills `statuses`.
    ///
    /// # Safety
    ///
    /// Work ordered on `stream` must write every status slot.
    /// Every buffer the work uses must stay alive, and nothing may write to it, until the work finishes.
    pub const unsafe fn from_submission(statuses: &'a DeviceStatuses, stream: &'a Stream) -> Self {
        Self { statuses, stream }
    }

    /// Returns the status buffer, whose slots hold results only after the work finishes.
    #[must_use]
    pub const fn statuses(&self) -> &'a DeviceStatuses {
        self.statuses
    }

    /// Returns the stream that runs the work.
    #[must_use]
    pub const fn stream(&self) -> &'a Stream {
        self.stream
    }

    /// Waits for the work and returns the statuses if every chunk succeeded.
    ///
    /// This does not extend the lifetime of any buffer the work uses.
    ///
    /// # Errors
    ///
    /// Returns an error if the status buffer and stream belong to different CUDA contexts, stream capture is active or cannot be queried, synchronization or the status copy fails, or a chunk reports an error status.
    pub fn check(self) -> Result<Vec<Status>> {
        // SAFETY: `from_submission` requires work on `self.stream` that writes every status slot.
        unsafe { self.statuses.check(self.stream) }
    }
}

pub(crate) fn join_operands(
    stream: &Stream,
    input: &[DeviceRegion<'_>],
    output: &[DeviceRegion<'_>],
    workspace: Option<&DeviceOutputView<'_, u8>>,
    statuses: Option<&DeviceStatuses>,
) -> Result<()> {
    let mut regions = Vec::with_capacity(input.len() + output.len() + 2);
    regions.extend_from_slice(input);
    regions.extend_from_slice(output);
    regions.extend(workspace.map(|workspace| workspace.region()));
    regions.extend(statuses.map(DeviceStatuses::region));
    Ok(stream.join_regions(&regions)?)
}

pub(crate) fn check_workspace(
    workspace: Option<&DeviceOutputView<'_, u8>>,
    required: usize,
    alignment: usize,
    stream: &Stream,
) -> Result<()> {
    if let Some(workspace) = workspace
        && workspace.context() != stream.context()
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    let actual = workspace.map_or(0, DeviceOutputView::len);
    if actual < required {
        return Err(Error::BufferTooSmall {
            name: "workspace".into(),
            required,
            actual,
        });
    }

    if let Some(workspace) = workspace
        && !workspace.is_empty()
    {
        let address = workspace.region().as_ptr().addr();
        validate_alignment(
            "workspace",
            is_alignment_satisfied(1, address, alignment),
            alignment,
        )?;
    }
    Ok(())
}

// nvCOMP takes a null pointer for an empty workspace.
pub(crate) fn workspace_parts(
    workspace: &mut Option<DeviceOutputView<'_, u8>>,
) -> (*mut u8, usize) {
    match workspace {
        Some(workspace) if !workspace.is_empty() => (workspace.as_mut_ptr(), workspace.len()),
        _ => (ptr::null_mut(), 0),
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use singe_cuda::view::{DeviceSlice, DeviceSliceMut};

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_empty_read_chunk_table_has_zero_lengths() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let chunks: [&dyn DeviceSlice<u8>; 0] = [];
        let table = DeviceChunkTable::from_device_slices(allocation_context, &chunks)?;

        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        assert_eq!(table.max_chunk_bytes(), 0);
        assert_eq!(table.total_bytes(), 0);
        assert_eq!(table.view().len(), 0);
        assert!(table.view().device_chunk_ptrs().is_null());
        assert!(table.view().device_chunk_bytes().is_null());

        Ok(())
    }

    #[test]
    fn test_empty_write_chunk_table_has_zero_lengths() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut chunks: [&mut dyn DeviceSliceMut<u8>; 0] = [];
        let table = DeviceChunkTableMut::from_device_slices(allocation_context, &mut chunks)?;

        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
        assert_eq!(table.min_buffer_bytes(), 0);
        assert_eq!(table.max_buffer_bytes(), 0);
        assert_eq!(table.total_buffer_bytes(), 0);
        assert_eq!(table.buffers().len(), 0);
        assert_eq!(table.written_chunks().len(), 0);
        assert!(table.buffers().device_chunk_ptrs().is_null());
        assert!(table.buffers().device_chunk_bytes().is_null());
        assert!(table.written_chunks().device_chunk_ptrs().is_null());
        assert!(table.written_chunks().device_chunk_bytes().is_null());
        assert!(table.written_bytes_device_ptr().is_null());

        Ok(())
    }

    #[test]
    fn test_contiguous_read_chunk_table_splits_prefix() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 16)?;
        let table =
            DeviceChunkTable::from_contiguous_chunk_sizes(allocation_context, &memory, &[4, 4, 2])?;

        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());
        assert_eq!(table.max_chunk_bytes(), 4);
        assert_eq!(table.total_bytes(), 10);
        assert!(table.is_aligned_to(4));

        Ok(())
    }

    #[test]
    fn test_strided_read_chunk_table_uses_actual_sizes() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 24)?;
        let table = DeviceChunkTable::from_contiguous_strided_chunks(
            allocation_context,
            &memory,
            8,
            &[3, 5, 1],
        )?;

        assert_eq!(table.len(), 3);
        assert_eq!(table.max_chunk_bytes(), 5);
        assert_eq!(table.total_bytes(), 9);
        assert!(table.is_aligned_to(8));

        Ok(())
    }

    #[test]
    fn test_contiguous_read_chunk_table_uses_variable_sizes() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 16)?;
        let table =
            DeviceChunkTable::from_contiguous_chunk_sizes(allocation_context, &memory, &[3, 5, 1])?;

        assert_eq!(table.len(), 3);
        assert_eq!(table.max_chunk_bytes(), 5);
        assert_eq!(table.total_bytes(), 9);

        Ok(())
    }

    #[test]
    fn test_contiguous_read_chunk_table_rejects_oversized_variable_sizes() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 8)?;
        let error =
            DeviceChunkTable::from_contiguous_chunk_sizes(allocation_context, &memory, &[3, 6])
                .unwrap_err();

        assert!(matches!(
            error,
            Error::BufferTooSmall {
                name,
                required: 9,
                actual: 8,
            } if name == "chunk bytes"
        ));

        Ok(())
    }

    #[test]
    fn test_strided_read_chunk_table_rejects_oversized_chunk() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 16)?;
        let error =
            DeviceChunkTable::from_contiguous_strided_chunks(allocation_context, &memory, 8, &[9])
                .unwrap_err();

        assert!(matches!(
            error,
            Error::BufferTooSmall {
                name,
                required: 9,
                actual: 8,
            } if name == "stride bytes"
        ));

        Ok(())
    }

    #[test]
    fn test_strided_read_chunk_table_rejects_undersized_buffer() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 12)?;
        let error = DeviceChunkTable::from_contiguous_strided_chunks(
            allocation_context,
            &memory,
            8,
            &[3, 5],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::BufferTooSmall {
                name,
                required: 13,
                actual: 12,
            } if name == "strided buffer bytes"
        ));

        Ok(())
    }

    #[test]
    fn test_contiguous_write_chunk_table_splits_buffer() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut memory = DeviceMemory::<u8>::create_uninit(allocation_context, 10)?;
        let table =
            DeviceChunkTableMut::from_contiguous_chunks(allocation_context, &mut memory, 4)?;

        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());
        assert_eq!(table.min_buffer_bytes(), 2);
        assert_eq!(table.max_buffer_bytes(), 4);
        assert_eq!(table.total_buffer_bytes(), 10);
        assert!(table.is_aligned_to(2));

        Ok(())
    }

    #[test]
    fn test_contiguous_write_chunk_table_uses_variable_sizes() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut memory = DeviceMemory::<u8>::create_uninit(allocation_context, 16)?;
        let table = DeviceChunkTableMut::from_contiguous_chunk_sizes(
            allocation_context,
            &mut memory,
            &[3, 5, 1],
        )?;

        assert_eq!(table.len(), 3);
        assert_eq!(table.min_buffer_bytes(), 1);
        assert_eq!(table.max_buffer_bytes(), 5);
        assert_eq!(table.total_buffer_bytes(), 9);

        Ok(())
    }

    #[test]
    fn test_contiguous_write_chunk_table_rejects_oversized_variable_sizes() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut memory = DeviceMemory::<u8>::create_uninit(allocation_context, 8)?;
        let error = DeviceChunkTableMut::from_contiguous_chunk_sizes(
            allocation_context,
            &mut memory,
            &[3, 6],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            Error::BufferTooSmall {
                name,
                required: 9,
                actual: 8,
            } if name == "chunk bytes"
        ));

        Ok(())
    }

    #[test]
    fn test_contiguous_chunk_tables_reject_zero_chunk_size() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut memory = DeviceMemory::<u8>::zeroed(allocation_context, 10)?;

        assert!(matches!(
            DeviceChunkTable::from_contiguous_chunks(allocation_context, &memory, 0).unwrap_err(),
            Error::NotPositive { name } if name == "chunk bytes"
        ));
        assert!(matches!(
            DeviceChunkTableMut::from_contiguous_chunks(allocation_context, &mut memory, 0).unwrap_err(),
            Error::NotPositive { name } if name == "chunk bytes"
        ));

        Ok(())
    }

    #[test]
    fn test_read_chunk_table_tracks_pointer_alignment() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let memory = DeviceMemory::<u8>::zeroed(allocation_context, 16)?;
        let misaligned = memory.view().slice(1..)?;
        let table = DeviceChunkTable::from_contiguous(allocation_context, &misaligned)?;

        assert!(table.is_aligned_to(1));
        assert!(!table.is_aligned_to(2));
        assert!(table.view().is_aligned_to(1));
        assert!(!table.view().is_aligned_to(2));

        Ok(())
    }

    #[test]
    fn test_write_chunk_table_tracks_pointer_alignment() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut memory = DeviceMemory::<u8>::create_uninit(allocation_context, 16)?;
        let mut view = memory.view_mut();
        let mut misaligned = view.slice_mut(1..)?;
        let table = DeviceChunkTableMut::from_contiguous(allocation_context, &mut misaligned)?;

        assert!(table.is_aligned_to(1));
        assert!(!table.is_aligned_to(2));
        assert!(table.buffers().is_aligned_to(1));
        assert!(!table.buffers().is_aligned_to(2));
        assert!(table.written_chunks().is_aligned_to(1));
        assert!(!table.written_chunks().is_aligned_to(2));

        Ok(())
    }

    #[test]
    fn test_device_statuses_copy_and_check_returns_all_successes() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let statuses = DeviceStatuses::from_memory(
            DeviceMemory::from_slice(
                allocation_context,
                &[
                    DeviceStatus::from_raw(Status::Success.raw()),
                    DeviceStatus::from_raw(Status::Success.raw()),
                ],
            )?
            .into_uninit(),
        );

        assert_eq!(statuses.len(), 2);
        assert!(!statuses.is_empty());
        statuses.require(2)?;
        assert_eq!(
            // SAFETY: the statuses were copied from host values, so no producer is outstanding.
            unsafe { statuses.copy_and_check_completed()? },
            vec![Status::Success, Status::Success]
        );

        Ok(())
    }

    #[test]
    fn test_empty_device_statuses_use_null_device_pointer() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let mut statuses = DeviceStatuses::create(allocation_context, 0)?;

        assert!(statuses.is_empty());
        assert!(statuses.as_device_ptr().is_null());
        statuses.require(0)?;

        Ok(())
    }

    #[test]
    fn test_device_statuses_copy_and_check_reports_first_failure() -> Result<()> {
        let test = setup_context()?;
        let allocation_context = test.cuda_context();
        let statuses = DeviceStatuses::from_memory(
            DeviceMemory::from_slice(
                allocation_context,
                &[
                    DeviceStatus::from_raw(Status::Success.raw()),
                    DeviceStatus::from_raw(Status::CannotCompress.raw()),
                    DeviceStatus::from_raw(Status::OutputBufferTooSmall.raw()),
                ],
            )?
            .into_uninit(),
        );

        // SAFETY: the statuses were copied from host values, so no producer is outstanding.
        let error = unsafe { statuses.copy_and_check_completed() }.unwrap_err();
        assert!(matches!(
            error,
            Error::OperationFailed {
                operation,
                status: Status::CannotCompress,
            } if operation == "chunk 1"
        ));

        Ok(())
    }
}
