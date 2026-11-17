use std::sync::Arc;

use super::DeviceMemory;
use crate::{
    error::{Error, Result},
    memory::{MemoryCopyKind, PinnedHostSlice, PinnedHostSliceMut, transfer},
    stream::{Stream, StreamScope},
    submission::{Access, AccessMode},
    utility::checked_bytes,
    view::{AnyBitPatternDeviceRepr, DeviceRegionProvider, DeviceRepr, DeviceSlice, HostSlice},
};

impl<T> DeviceMemory<T> {
    /// Copies host values and waits until the device range contains the uploaded values.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow.
    /// Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_from_host(&mut self, host_slice: &[T]) -> Result<()>
    where
        T: DeviceRepr,
    {
        transfer::copy_from_host(self, host_slice)
    }

    /// Copies from page-locked host memory into this device allocation.
    ///
    /// The source length must match this allocation's length.
    /// This is the pinned-memory-specific counterpart to [`DeviceMemory::copy_from_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows, or CUDA rejects the host-to-device copy.
    pub fn copy_from_pinned<B>(&mut self, src: &B) -> Result<()>
    where
        B: PinnedHostSlice<T> + ?Sized,
        T: DeviceRepr,
    {
        let context = Arc::clone(self.context());
        self.bind()?;
        let () = T::LAYOUT;
        if HostSlice::len(src) != self.length {
            return Err(Error::length_mismatch(
                "host source",
                self.length,
                HostSlice::len(src),
            ));
        }

        if self.length == 0 {
            return Ok(());
        }

        checked_bytes::<T>(self.length, "byte length")?;
        let _access = context.acquire_accesses(&[
            Access::new(self.region(), AccessMode::Write),
            Access::host(src.host_region(), AccessMode::Read),
        ])?;
        // SAFETY: the guards acquired above cover both regions, and the lengths were checked to match.
        unsafe {
            transfer::copy_synchronous(
                self.ptr,
                src.as_host_ptr(),
                self.length,
                MemoryCopyKind::HostToDevice,
                None,
            )
        }
    }

    /// Enqueues a copy from page-locked host memory into this device allocation.
    ///
    /// The [`StreamScope`] ties the source and destination lifetimes to the queued copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows, or CUDA cannot enqueue the host-to-device copy.
    pub fn copy_from_pinned_scoped<'env, B>(
        &'env mut self,
        src: &'env B,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        B: PinnedHostSlice<T> + ?Sized,
        T: DeviceRepr,
    {
        self.bind_stream(stream.stream())?;
        let () = T::LAYOUT;
        if HostSlice::len(src) != self.length {
            return Err(Error::length_mismatch(
                "host source",
                self.length,
                HostSlice::len(src),
            ));
        }

        checked_bytes::<T>(self.length, "byte length")?;
        if self.length == 0 {
            return Ok(());
        }

        let accesses = [
            Access::new(self.region(), AccessMode::Write),
            Access::host(src.host_region(), AccessMode::Read),
        ];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    self.ptr,
                    src.as_host_ptr(),
                    self.length,
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a host-to-device copy of the entire allocation in a stream scope.
    ///
    /// The scope retains both borrows until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the element counts differ, the byte count overflows, or the scope’s stream belongs to another context.
    /// Also propagates context-binding, scope submission, and CUDA copy errors.
    pub fn copy_from_host_scoped<'env>(
        &'env mut self,
        host_slice: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        if host_slice.len() != self.len() {
            return Err(Error::length_mismatch(
                "host slice",
                self.len(),
                host_slice.len(),
            ));
        }

        checked_bytes::<T>(self.len(), "byte length")?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }

        let accesses = [Access::new(self.region(), AccessMode::Write)];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    self.ptr,
                    host_slice.as_ptr(),
                    self.len(),
                    MemoryCopyKind::HostToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a copy of `host_slice` into this allocation on `stream`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until `stream` has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the host and device element counts differ, the stream belongs to another context, or the byte count overflows.
    /// Also propagates context-binding, producer-ordering, and CUDA asynchronous copy errors.
    pub unsafe fn copy_from_host_async(&mut self, host_slice: &[T], stream: &Stream) -> Result<()> {
        self.bind_stream(stream)?;
        if host_slice.len() != self.len() {
            return Err(Error::length_mismatch(
                "host slice",
                self.len(),
                host_slice.len(),
            ));
        }

        if self.is_empty() {
            return Ok(());
        }

        stream.join_regions(&[self.region()])?;
        // SAFETY: the lengths match, `stream` is ordered after the allocation's producer, and the caller keeps both ranges valid until `stream` completes the copy.
        unsafe {
            Self::copy_async(
                self.as_mut_ptr(),
                host_slice.as_ptr(),
                self.len(),
                MemoryCopyKind::HostToDevice,
                stream,
            )
        }
    }

    /// Copies all device elements into an equally sized host slice and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow.
    /// Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host(&self, host_slice: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        transfer::copy_to_host(self, host_slice)
    }

    /// Copies device values to host storage under a kernel-specific validity contract.
    ///
    /// # Safety
    ///
    /// Every source element must be initialized and contain a valid value of `T`.
    /// `T` must be transferable without duplicating ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if host and device element counts differ or the byte count overflows.
    /// Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn copy_to_host_unchecked(&self, host_slice: &mut [T]) -> Result<()>
    where
        T: DeviceRepr,
    {
        // SAFETY: the caller vouches that every source element is a valid, transferable `T`.
        unsafe { transfer::copy_to_host_unchecked(self, host_slice) }
    }

    /// Copies this device allocation into page-locked host memory.
    ///
    /// The destination length must match this allocation's length.
    /// This is the pinned-memory-specific counterpart to [`DeviceMemory::copy_to_host`].
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows, or CUDA rejects the device-to-host copy.
    pub fn copy_to_pinned<B>(&self, dst: &mut B) -> Result<()>
    where
        B: PinnedHostSliceMut<T> + ?Sized,
        T: AnyBitPatternDeviceRepr,
    {
        let context = Arc::clone(self.context());
        self.bind()?;
        let () = T::LAYOUT;
        if HostSlice::len(dst) != self.length {
            return Err(Error::length_mismatch(
                "host destination",
                self.length,
                HostSlice::len(dst),
            ));
        }

        if self.length == 0 {
            return Ok(());
        }

        checked_bytes::<T>(self.length, "byte length")?;
        let _access = context.acquire_accesses(&[
            Access::new(self.region(), AccessMode::Read),
            Access::host(dst.host_region(), AccessMode::Write),
        ])?;
        // SAFETY: the guards acquired above cover both regions, and the lengths were checked to match.
        unsafe {
            transfer::copy_synchronous(
                dst.as_host_mut_ptr(),
                self.ptr,
                self.length,
                MemoryCopyKind::DeviceToHost,
                None,
            )
        }
    }

    /// Enqueues a copy from this device allocation into page-locked host memory.
    ///
    /// The [`StreamScope`] ties the source and destination lifetimes to the queued copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths differ, byte-size arithmetic overflows, or CUDA cannot enqueue the device-to-host copy.
    pub fn copy_to_pinned_scoped<'env, B>(
        &'env self,
        dst: &'env mut B,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        B: PinnedHostSliceMut<T> + ?Sized,
        T: AnyBitPatternDeviceRepr,
    {
        self.bind_stream(stream.stream())?;
        let () = T::LAYOUT;
        if HostSlice::len(dst) != self.length {
            return Err(Error::length_mismatch(
                "host destination",
                self.length,
                HostSlice::len(dst),
            ));
        }

        checked_bytes::<T>(self.length, "byte length")?;
        if self.length == 0 {
            return Ok(());
        }

        let accesses = [
            Access::new(self.region(), AccessMode::Read),
            Access::host(dst.host_region(), AccessMode::Write),
        ];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    dst.as_host_mut_ptr(),
                    self.ptr,
                    self.length,
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a device-to-host copy of the entire allocation in a stream scope.
    ///
    /// The scope retains the source and exclusive destination borrows until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the element counts differ, the byte count overflows, or the scope’s stream belongs to another context.
    /// Also propagates context-binding, scope submission, and CUDA copy errors.
    pub fn copy_to_host_scoped<'env>(
        &'env self,
        host_slice: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        if host_slice.len() != self.len() {
            return Err(Error::length_mismatch(
                "host slice",
                self.len(),
                host_slice.len(),
            ));
        }

        checked_bytes::<T>(self.len(), "byte length")?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }

        let accesses = [Access::new(self.region(), AccessMode::Read)];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    host_slice.as_mut_ptr(),
                    self.ptr,
                    self.len(),
                    MemoryCopyKind::DeviceToHost,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a copy of this allocation into `host_slice` on `stream`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until `stream` has completed the transfer.
    /// Every source element must be initialized and valid for `T`, without Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if the host and device element counts differ, the stream belongs to another context, or the byte count overflows.
    /// Also propagates context-binding, producer-ordering, and CUDA asynchronous copy errors.
    pub unsafe fn copy_to_host_async(&self, host_slice: &mut [T], stream: &Stream) -> Result<()> {
        self.bind_stream(stream)?;
        if host_slice.len() != self.len() {
            return Err(Error::length_mismatch(
                "host slice",
                self.len(),
                host_slice.len(),
            ));
        }

        if self.is_empty() {
            return Ok(());
        }

        stream.join_regions(&[self.region()])?;
        // SAFETY: the lengths match, `stream` is ordered after the allocation's producer, and the caller keeps both ranges valid until `stream` completes and vouches for the source values.
        unsafe {
            Self::copy_async(
                host_slice.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
                stream,
            )
        }
    }

    /// Copies all device elements into a new host vector and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates byte-size overflow, access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        transfer::copy_to_host_vec(self)
    }

    /// Copies initialized device values into a host vector under a kernel-specific validity contract.
    ///
    /// # Safety
    ///
    /// Every source element must contain a valid value of `T`.
    /// `T` must be transferable without duplicating ownership or host references.
    ///
    /// # Errors
    ///
    /// Propagates byte-size, access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn copy_to_host_vec_unchecked(&self) -> Result<Vec<T>> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.bind()?;
        // SAFETY: the context is bound, the access guard orders tracked producers, `ptr` holds `length` elements, and the caller vouches that each is a valid `T`.
        unsafe {
            transfer::copy_to_host_vec_unchecked(self.ptr, self.length, self.region().stream())
        }
    }

    /// Copies a disjoint device range into this allocation and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if contexts or element counts differ, the ranges overlap within one allocation, or the byte count overflows.
    /// Also propagates access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_from_device<S>(&mut self, source: &S) -> Result<()>
    where
        T: DeviceRepr,
        S: DeviceSlice<T> + ?Sized,
    {
        transfer::copy_device(self, source)
    }

    /// Enqueues a device-to-device copy of the entire allocation in a stream scope.
    ///
    /// The scope retains both borrows until the transfer completes.
    ///
    /// # Errors
    ///
    /// Returns an error if source and destination contexts or element counts differ, the byte count overflows, or the scope’s stream belongs to another context.
    /// Also propagates context-binding, scope submission, and CUDA copy errors.
    pub fn copy_from_device_scoped<'env>(
        &'env mut self,
        src: &'env Self,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        if self.context() != src.context() {
            return Err(Error::MemoryContextMismatch);
        }

        if src.len() != self.len() {
            return Err(Error::length_mismatch("source", self.len(), src.len()));
        }

        checked_bytes::<T>(self.len(), "byte length")?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }

        let accesses = [
            Access::new(self.region(), AccessMode::Write),
            Access::new(src.region(), AccessMode::Read),
        ];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::copy_async(
                    self.ptr,
                    src.as_ptr(),
                    self.len(),
                    MemoryCopyKind::DeviceToDevice,
                    stream.stream(),
                )
            })
        }
    }

    /// Enqueues a copy of `src` into this allocation on `stream`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `self` and `src` both remain valid until `stream` has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if source and destination contexts or element counts differ, the stream belongs to another context, or the byte count overflows.
    /// Also propagates context-binding, producer-ordering, and CUDA asynchronous copy errors.
    pub unsafe fn copy_from_device_async(&mut self, src: &Self, stream: &Stream) -> Result<()> {
        if self.context() != src.context() {
            return Err(Error::MemoryContextMismatch);
        }

        self.bind_stream(stream)?;
        if src.len() != self.len() {
            return Err(Error::length_mismatch("source", self.len(), src.len()));
        }

        if self.is_empty() {
            return Ok(());
        }

        stream.join_regions(&[self.region(), src.region()])?;
        // SAFETY: contexts and lengths match, `&mut self` keeps the ranges disjoint, `stream` is ordered after both producers, and the caller keeps both valid until `stream` completes.
        unsafe {
            Self::copy_async(
                self.as_mut_ptr(),
                src.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToDevice,
                stream,
            )
        }
    }
}
