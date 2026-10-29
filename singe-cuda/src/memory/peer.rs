//! Direct context-to-context device copies with explicit readiness and ownership.

use std::{mem::size_of, sync::Arc};

use singe_cuda_sys::driver;

use crate::{
    context::{Context, PeerAccessFlags},
    device::DeviceAttribute,
    error::{Error, Result, Status},
    stream::{Stream, StreamCaptureStatus},
    submission::{Access, AccessMode, lock_access},
    try_ffi,
    view::{
        DeviceInput, DeviceOutput, DeviceRegion, DeviceRepr, DeviceSlice, DeviceViewMut,
        ranges_overlap,
    },
};

/// A directed copy path retaining its source and destination contexts.
/// Different devices must support direct peer access, which is enabled at creation.
/// Permission remains enabled after this value drops, until explicitly disabled
/// or either context is destroyed. This type does not stage copies through the host.
#[derive(Debug, Clone)]
pub struct PeerCopy {
    source: Arc<Context>,
    destination: Arc<Context>,
}

impl PeerCopy {
    /// Prepares a direct copy path. Unsupported peer access is returned to the caller.
    /// Contexts on the same device can copy without inter-device peer permission.
    /// Both endpoints must support unified addressing. Pool and VMM allocations
    /// retain their own access-descriptor requirements.
    ///
    /// # Errors
    ///
    /// Returns the native not-supported status if either endpoint lacks unified addressing. Also propagates device-attribute query errors and failures to enable directed peer access from the destination context to the source context.
    pub fn create(source: &Arc<Context>, destination: &Arc<Context>) -> Result<Self> {
        if source
            .device()
            .attribute(DeviceAttribute::UnifiedAddressing)?
            == 0
            || destination
                .device()
                .attribute(DeviceAttribute::UnifiedAddressing)?
                == 0
        {
            return Err(Status::NotSupported.into());
        }
        if source.device() != destination.device() {
            destination.enable_peer_access(source, PeerAccessFlags::DEFAULT)?;
        }
        Ok(Self {
            source: Arc::clone(source),
            destination: Arc::clone(destination),
        })
    }

    /// Returns the retained context owning source allocations.
    #[must_use]
    pub fn source_context(&self) -> &Arc<Context> {
        &self.source
    }

    /// Returns the retained context owning destination allocations and the copy stream.
    #[must_use]
    pub fn destination_context(&self) -> &Arc<Context> {
        &self.destination
    }

    /// Completes copying an initialized source into an equally sized writable range.
    /// Both readiness streams are ordered before the copy. Borrowed storage and
    /// contexts remain live through completion on success, error, and unwind.
    ///
    /// # Errors
    ///
    /// Returns an error for source or destination context mismatches, a stream from another destination context, unequal element counts, overlapping ranges in one allocation, or an unrepresentable byte count. Also rejects active or invalidated capture on the copy or readiness streams and propagates access-tracking, synchronization, stream-ordering, native copy, and output-view construction errors.
    pub fn copy<'destination, T, S, D>(
        &self,
        destination: &'destination mut D,
        source: &S,
        stream: &Stream,
    ) -> Result<DeviceViewMut<'destination, T>>
    where
        T: DeviceRepr,
        S: DeviceSlice<T> + ?Sized,
        D: DeviceOutput<T> + ?Sized,
    {
        self.validate(destination, source, stream)?;
        let source_region = source.region();
        let destination_region = destination.region();
        require_eager(stream)?;
        for region in [source_region, destination_region] {
            if let Some(producer) = region.stream() {
                require_eager(producer)?;
            }
        }
        let accesses = [
            Access::new(source_region, AccessMode::Read),
            Access::new(destination_region, AccessMode::Write),
        ];
        let mut access = lock_access(&self.destination.access_tracker, &accesses)?;
        access.synchronize(source_region)?;
        access.synchronize(destination_region)?;
        stream.sync_scope(|_| {
            if let Some(producer) = source_region.stream()
                && producer != stream
            {
                stream.join(producer)?;
            }
            stream.join_regions(&[destination.region()])?;
            // SAFETY: initialized, exclusive disjoint ranges and both actual
            // contexts stay borrowed until scope completion; readiness is joined.
            unsafe { self.copy_async_unchecked(destination, source, stream) }
        })?;
        // SAFETY: completed copying preserves the source's valid representations.
        unsafe { DeviceViewMut::from_raw_parts(destination.region(), destination.len()) }
    }

    /// Enqueues a direct native copy using both retained actual contexts.
    ///
    /// # Safety
    /// Source values must be valid when read and both allocation readiness
    /// dependencies must precede the copy. Keep the source, exclusive destination,
    /// both contexts, and stream live until completion, including after errors.
    /// During capture retain all dependencies through every graph replay.
    /// Do not revoke peer or pool access while this path or its work can use it.
    ///
    /// # Errors
    ///
    /// Returns `Error::MemoryContextMismatch` for an allocation from the wrong endpoint context, `Error::StreamContextMismatch` for a stream outside the destination context, or `Error::InvalidMemoryAccess` for unequal counts, overlapping ranges, or an unrepresentable byte count. Also propagates destination context binding and native copy errors. An empty, otherwise valid copy succeeds without binding or enqueueing native work.
    pub unsafe fn copy_async_unchecked<T, S, D>(
        &self,
        destination: &mut D,
        source: &S,
        stream: &Stream,
    ) -> Result<()>
    where
        T: DeviceRepr,
        S: DeviceInput<T> + ?Sized,
        D: DeviceOutput<T> + ?Sized,
    {
        self.validate(destination, source, stream)?;
        let bytes = source
            .len()
            .checked_mul(size_of::<T>())
            .ok_or(Error::InvalidMemoryAccess)?;
        let native_bytes =
            driver::size_t::try_from(bytes).map_err(|_| Error::InvalidMemoryAccess)?;
        if bytes == 0 {
            return Ok(());
        }
        self.destination.bind()?;
        // Explicit context validation and peer setup precede this UVA copy.
        // Unlike the context-parameter peer primitives, cuMemcpyAsync supports
        // capture and has no 3D pitch restriction for a linear byte range.
        // SAFETY: `cuMemcpyAsync` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(driver::cuMemcpyAsync(
                destination.as_mut_ptr() as driver::CUdeviceptr,
                source.as_device_ptr() as driver::CUdeviceptr,
                native_bytes,
                stream.as_raw(),
            ))
        }
    }

    fn validate<T, S, D>(&self, destination: &D, source: &S, stream: &Stream) -> Result<()>
    where
        T: DeviceRepr,
        S: DeviceInput<T> + ?Sized,
        D: DeviceOutput<T> + ?Sized,
    {
        let () = T::LAYOUT;
        if source.context() != &self.source || destination.context() != &self.destination {
            return Err(Error::MemoryContextMismatch);
        }
        if stream.context() != &self.destination {
            return Err(Error::StreamContextMismatch);
        }
        if source.len() != destination.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        require_disjoint(source.region(), destination.region())
    }
}

fn require_disjoint(source: DeviceRegion<'_>, destination: DeviceRegion<'_>) -> Result<()> {
    let source_range = source.byte_range();
    let destination_range = destination.byte_range();
    if source.allocation_id() == destination.allocation_id()
        && ranges_overlap(&source_range, &destination_range)
    {
        return Err(Error::InvalidMemoryAccess);
    }
    Ok(())
}

fn require_eager(stream: &Stream) -> Result<()> {
    match stream.capture_status()? {
        StreamCaptureStatus::None => Ok(()),
        StreamCaptureStatus::Invalidated => Err(Status::StreamCaptureInvalidated.into()),
        _ => Err(Status::StreamCaptureUnsupported.into()),
    }
}
