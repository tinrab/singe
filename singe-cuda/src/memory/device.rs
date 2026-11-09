use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use super::{
    DeviceAllocation, DeviceDeallocation, DeviceMemory, MemoryAttachFlags, MemoryCopyKind,
    MemoryCopyOperation, MemoryLocation, MemorySetOperation, PinnedHostSlice, PinnedHostSliceMut,
    PointerAttributes, checked_bytes, checked_len, checked_volume, transfer,
};
use crate::{
    context::Context,
    error::{Error, Result},
    ipc::IpcMemoryHandle,
    stream::{Stream, StreamBinding, StreamScope},
    submission::{Access, AccessMode},
    try_ffi,
    types::{DevicePtr, MemoryAdvise},
    utility::to_usize,
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceRegionProvider, DeviceRepr, DeviceSlice,
        HostSlice, NoUninitDeviceRepr, ZeroableDeviceRepr, cast_length,
    },
};

/// Associated utility functions.
impl<T> DeviceMemory<T> {
    /// Allocates size bytes of linear memory on the device and returns a pointer to the allocated memory.
    /// The allocated memory is suitably aligned for any kind of variable.
    /// The memory is not cleared.
    /// [`DeviceMemory::alloc`] returns [`crate::error::Status::OutOfMemory`] on allocation failure.
    ///
    /// The device version of [`DeviceMemory::free`] cannot be used with a pointer allocated using the host API, and vice versa.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot allocate device memory, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`].
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory.
    /// The caller must use it only for `count` elements of `T` and eventually free it with a compatible CUDA free function.
    pub unsafe fn alloc(count: usize) -> Result<*mut T> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }

        let mut p = ptr::null_mut();
        // SAFETY: `cudaMalloc` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMalloc(&raw mut p, bytes as _))?;
        }
        Ok(p.cast())
    }

    /// Allocates CUDA managed memory for `count` elements.
    ///
    /// `flags` controls the initial stream attachment behavior for the managed allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot allocate managed memory, or `count` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer is managed CUDA memory.
    /// The caller must eventually free it with a CUDA-compatible device free and must synchronize host and device access according to CUDA managed-memory rules.
    pub unsafe fn alloc_managed(count: usize, flags: MemoryAttachFlags) -> Result<*mut T> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }

        let mut p = ptr::null_mut();
        // SAFETY: `cudaMallocManaged` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMallocManaged(
                &raw mut p,
                bytes as _,
                flags.bits(),
            ))?;
        }
        Ok(p.cast::<T>())
    }

    /// Frees the memory space pointed to by `ptr`, which must have been returned by a previous call to one of these allocation functions: [`DeviceMemory::alloc`], [`sys::cudaMallocPitch`](singe_cuda_sys::runtime::cudaMallocPitch), [`DeviceMemory::alloc_managed`], [`DeviceMemory::alloc_async`], or [`sys::cudaMallocFromPoolAsync`](singe_cuda_sys::runtime::cudaMallocFromPoolAsync).
    ///
    /// This does not perform implicit synchronization when the pointer was allocated with [`DeviceMemory::alloc_async`] or [`sys::cudaMallocFromPoolAsync`](singe_cuda_sys::runtime::cudaMallocFromPoolAsync).
    /// Callers must ensure that all accesses to this pointer have completed before invoking [`DeviceMemory::free`].
    /// For best performance and memory reuse, use [`DeviceMemory::free_async`] to free memory allocated via the stream ordered memory allocator.
    /// For all other pointers, this call may perform implicit synchronization.
    ///
    /// If [`DeviceMemory::free`] has already been called before, an error is returned.
    /// If `ptr` is null, no operation is performed.
    /// [`DeviceMemory::free`] returns an error on failure.
    ///
    /// The device version of [`DeviceMemory::free`] cannot be used with a pointer allocated using the host API, and vice versa.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot free `ptr`, `ptr` has already been freed, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or a live allocation returned by a compatible CUDA device allocation function, and no work may access it after it is freed.
    /// No live allocation owner may subsequently free the pointer again.
    pub unsafe fn free(ptr: *mut T) -> Result<()> {
        // SAFETY: `cudaFree` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaFree(ptr.cast()))?;
        }
        Ok(())
    }

    /// Allocates 2D pitched device memory and returns the base pointer and row pitch in bytes.
    ///
    /// `width` is measured in elements of `T`, while the returned pitch is measured in bytes.
    /// CUDA may pad each row so that row-to-row addressing satisfies device alignment requirements.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, `T` is a non-empty zero-sized allocation, CUDA cannot allocate pitched memory, or CUDA returns a null allocation for a non-empty request.
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory.
    /// The caller must use it only as a `width` by `height` pitched allocation with the returned pitch and eventually free it with a CUDA-compatible device free.
    pub unsafe fn alloc_pitched(width: usize, height: usize) -> Result<(*mut T, usize)> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok((ptr::null_mut(), row_bytes));
        }

        let mut ptr = ptr::null_mut();
        let mut pitch = 0;
        // SAFETY: `cudaMallocPitch` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMallocPitch(
                &raw mut ptr,
                &raw mut pitch,
                row_bytes as _,
                height as _,
            ))?;
        }

        if ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok((ptr.cast(), to_usize(pitch, "allocation pitch")?))
    }

    /// Allocates 3D pitched device memory and returns CUDA's pitched pointer descriptor.
    ///
    /// `width` is measured in elements of `T`.
    /// CUDA's `cudaPitchedPtr::pitch` and `xsize` fields are measured in bytes for linear memory, and `ysize` records the logical height.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, `T` is a non-empty zero-sized allocation, CUDA cannot allocate pitched memory, or CUDA returns a null allocation for a non-empty request.
    ///
    /// # Safety
    ///
    /// The returned allocation is uninitialized device memory.
    /// The caller must treat it as a `width` by `height` by `depth` pitched allocation and free the base pointer with a CUDA-compatible device free.
    pub unsafe fn alloc_pitched_3d(
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<runtime::cudaPitchedPtr> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(runtime::cudaPitchedPtr {
                ptr: ptr::null_mut(),
                pitch: row_bytes as _,
                xsize: row_bytes as _,
                ysize: height as _,
            });
        }

        let mut pitched = runtime::cudaPitchedPtr::default();
        // SAFETY: `cudaMalloc3D` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMalloc3D(
                &raw mut pitched,
                runtime::cudaExtent {
                    width: row_bytes as _,
                    height: height as _,
                    depth: depth as _,
                },
            ))?;
        }

        if pitched.ptr.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(pitched)
    }

    /// Copies `count` elements from `src` to `dst`.
    /// The transfer direction is specified by [`MemoryCopyKind`].
    /// [`MemoryCopyKind::Default`] is recommended when unified virtual addressing is available, in which case the transfer direction is inferred from the pointer values.
    /// Calling [`DeviceMemory::copy`] with `dst` and `src` pointers that do not match the direction of the copy results in undefined behavior.
    ///
    /// * Exhibits `synchronous` behavior for most use cases.
    /// * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all.
    ///   Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and return [`crate::error::Status::InvalidValue`].
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA rejects the pointer combination or copy kind, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must be valid for `count` elements of `T` according to `kind`, and the source and destination regions must not overlap unless CUDA permits that transfer.
    pub unsafe fn copy(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }

        // SAFETY: `cudaMemcpy` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemcpy(
                dst.cast(),
                src.cast(),
                bytes as _,
                kind.into(),
            ))?;
        }
        Ok(())
    }

    /// Copies a typed 2D rectangle between pitched host or device memory regions.
    ///
    /// `width` and `height` are measured in elements and rows.
    /// `dst_pitch` and `src_pitch` are byte strides between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, either pitch is smaller than the copied row width, CUDA rejects the pointer combination or copy kind, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid for the described rectangle according to `kind`.
    /// The source and destination regions must not overlap unless CUDA permits that transfer.
    pub unsafe fn copy_2d(
        dst: *mut T,
        dst_pitch: usize,
        src: *const T,
        src_pitch: usize,
        width: usize,
        height: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > dst_pitch || row_bytes > src_pitch {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: `cudaMemcpy2D` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemcpy2D(
                dst.cast(),
                dst_pitch as _,
                src.cast(),
                src_pitch as _,
                row_bytes as _,
                height as _,
                kind.into(),
            ))?;
        }
        Ok(())
    }

    /// Enqueues a typed 2D rectangle copy on `stream`.
    ///
    /// `width` and `height` are measured in elements and rows.
    /// Pitches are byte strides between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, either pitch is smaller than the copied row width, CUDA cannot enqueue the copy, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for the described rectangle until `stream` reaches the enqueued copy.
    /// The pointer combination must match `kind`.
    pub unsafe fn copy_2d_async(
        dst: *mut T,
        dst_pitch: usize,
        src: *const T,
        src_pitch: usize,
        width: usize,
        height: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > dst_pitch || row_bytes > src_pitch {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: `cudaMemcpy2DAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemcpy2DAsync(
                dst.cast(),
                dst_pitch as _,
                src.cast(),
                src_pitch as _,
                row_bytes as _,
                height as _,
                kind.into(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Copies a typed 3D extent between CUDA pitched pointer operands.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and slices.
    /// CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, either pitch is smaller than the copied row width, CUDA rejects the copy parameters, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must describe live host or device memory large enough for the requested extent according to `kind`.
    /// The source and destination regions must not overlap unless CUDA permits that transfer.
    pub unsafe fn copy_3d(
        dst: runtime::cudaPitchedPtr,
        src: runtime::cudaPitchedPtr,
        width: usize,
        height: usize,
        depth: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        if row_bytes > to_usize(dst.pitch, "destination pitch")?
            || row_bytes > to_usize(src.pitch, "source pitch")?
        {
            return Err(Error::InvalidMemoryAccess);
        }

        let params = runtime::cudaMemcpy3DParms {
            srcArray: ptr::null_mut(),
            srcPos: runtime::cudaPos::default(),
            srcPtr: src,
            dstArray: ptr::null_mut(),
            dstPos: runtime::cudaPos::default(),
            dstPtr: dst,
            extent: runtime::cudaExtent {
                width: row_bytes as _,
                height: height as _,
                depth: depth as _,
            },
            kind: kind.into(),
        };
        // SAFETY: `cudaMemcpy3D` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemcpy3D(&raw const params))?;
        }
        Ok(())
    }

    /// Enqueues a typed 3D extent copy on `stream`.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and slices.
    /// CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, either pitch is smaller than the copied row width, CUDA cannot enqueue the copy, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for the described extent until `stream` reaches the enqueued copy.
    /// The pointer combination must match `kind`.
    pub unsafe fn copy_3d_async(
        dst: runtime::cudaPitchedPtr,
        src: runtime::cudaPitchedPtr,
        width: usize,
        height: usize,
        depth: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        if row_bytes > to_usize(dst.pitch, "destination pitch")?
            || row_bytes > to_usize(src.pitch, "source pitch")?
        {
            return Err(Error::InvalidMemoryAccess);
        }

        let params = runtime::cudaMemcpy3DParms {
            srcArray: ptr::null_mut(),
            srcPos: runtime::cudaPos::default(),
            srcPtr: src,
            dstArray: ptr::null_mut(),
            dstPos: runtime::cudaPos::default(),
            dstPtr: dst,
            extent: runtime::cudaExtent {
                width: row_bytes as _,
                height: height as _,
                depth: depth as _,
            },
            kind: kind.into(),
        };
        // SAFETY: `cudaMemcpy3DAsync` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemcpy3DAsync(
                &raw const params,
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Fills `count` elements at `dst` with the constant byte pattern `value`.
    ///
    /// This call is asynchronous with respect to the host unless `ptr` refers to pinned host memory.
    ///
    /// See the CUDA memset synchronization rules for when this operation blocks the host.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA rejects the pointer or size, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count * size_of::<T>()` bytes and must refer to memory that CUDA can memset.
    pub unsafe fn set(dst: *mut T, value: u8, count: usize) -> Result<()> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }

        // SAFETY: `cudaMemset` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemset(dst.cast(), value.into(), bytes as _))?;
        }
        Ok(())
    }

    /// Fills a typed 2D pitched memory region with a byte value.
    ///
    /// `width` and `height` are measured in elements and rows.
    /// `pitch` is the byte stride between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, `pitch` is smaller than the row width, CUDA rejects the memory region, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes over the described pitched rectangle and must refer to memory CUDA can memset.
    pub unsafe fn set_2d(
        dst: *mut T,
        pitch: usize,
        value: u8,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > pitch {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: `cudaMemset2D` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemset2D(
                dst.cast(),
                pitch as _,
                value.into(),
                row_bytes as _,
                height as _,
            ))?;
        }
        Ok(())
    }

    /// Fills a typed 3D pitched memory region with a byte value.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and slices.
    /// CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, `dst.pitch` is smaller than the row width, CUDA rejects the memory region, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must describe live memory large enough for the requested extent and must refer to memory CUDA can memset.
    pub unsafe fn set_3d(
        dst: runtime::cudaPitchedPtr,
        value: u8,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        if row_bytes > to_usize(dst.pitch, "destination pitch")? {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: `cudaMemset3D` receives only plain values.
        unsafe {
            try_ffi!(runtime::cudaMemset3D(
                dst,
                value.into(),
                runtime::cudaExtent {
                    width: row_bytes as _,
                    height: height as _,
                    depth: depth as _,
                },
            ))?;
        }
        Ok(())
    }

    /// Returns the total amount of memory available to the current context and the amount of memory free on the device.
    /// CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free.
    /// In a multi-tenant situation, the free-memory estimate is prone to a race condition: an allocation or free by another process or thread between estimation and reporting can make the reported free value differ from actual free memory.
    ///
    /// The integrated GPU on Tegra shares memory with CPU and other component of the `SoC`.
    /// The free and total values returned by this call exclude the SWAP memory space maintained by the OS on some platforms.
    /// The OS may move some of the memory pages into swap area as the GPU or CPU allocate or access memory.
    /// See Tegra app note on how to calculate total and free memory on Tegra.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query memory information, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn memory_info() -> Result<(usize, usize)> {
        let mut free: runtime::size_t = 0;
        let mut total: runtime::size_t = 0;
        // SAFETY: `cudaMemGetInfo` receives valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMemGetInfo(&raw mut free, &raw mut total))?;
        }
        Ok((
            to_usize(free, "free device memory")?,
            to_usize(total, "total device memory")?,
        ))
    }

    /// Returns the attributes of `ptr`.
    /// If `ptr` was not allocated in, mapped by, or registered with a context that supports unified addressing, [`crate::error::Status::InvalidValue`] is returned.
    ///
    /// In CUDA 11.0 and later, passing a host pointer reports [`MemoryType::Unregistered`] in [`PointerAttributes::memory_type`].
    ///
    /// * [`PointerAttributes::memory_type`] identifies the type of memory.
    ///   It can be [`MemoryType::Unregistered`] for unregistered host memory, [`MemoryType::Host`] for registered host memory, [`MemoryType::Device`] for device memory, or [`MemoryType::Managed`] for managed memory.
    ///
    /// * [`PointerAttributes::device`] is the device against which `ptr` was allocated.
    ///   If `ptr` has memory type [`MemoryType::Device`], this identifies the device on which the memory physically resides.
    ///   If `ptr` has memory type [`MemoryType::Host`], this identifies the device that was current when the allocation was made, and if that device is deinitialized then this allocation will vanish with that device's state.
    ///
    /// * [`PointerAttributes::device_pointer`] is the device pointer alias through which the memory referred to by `ptr` may be accessed on the current device.
    ///   If the memory referred to by `ptr` cannot be accessed directly by the current device then this is null.
    ///
    /// * [`PointerAttributes::host_pointer`] is the host pointer alias through which the memory referred to by `ptr` may be accessed on the host.
    ///   If the memory referred to by `ptr` cannot be accessed directly by the host then this is null.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query attributes for `ptr`, `ptr` is not known to a unified-addressing context, or CUDA reports runtime initialization diagnostics.
    pub fn pointer_attributes(ptr: *const T) -> Result<PointerAttributes> {
        PointerAttributes::query(ptr)
    }

    /// Enqueues a stream-ordered device allocation on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte size overflows, CUDA cannot enqueue `cudaMallocAsync`, or `count` is non-zero for a zero-sized `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory whose lifetime is ordered by `stream`.
    /// The caller must ensure all accesses are stream ordered correctly and free it with [`DeviceMemory::free_async`] or a compatible CUDA free after all accesses complete.
    pub unsafe fn alloc_async(count: usize, stream: &Stream) -> Result<*mut T> {
        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }

        let mut p = ptr::null_mut();
        // SAFETY: `cudaMallocAsync` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(runtime::cudaMallocAsync(
                &raw mut p,
                bytes as _,
                stream.as_raw()
            ))?;
        }
        Ok(p.cast::<T>())
    }

    /// Inserts a free operation into `stream`.
    /// The allocation must not be accessed after stream execution reaches the free.
    /// After this call returns, accessing the memory from any subsequent work launched on the GPU or querying its pointer attributes results in undefined behavior.
    ///
    /// During stream capture, this creates a free node and must therefore be passed the address of a graph allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the free on `stream`, `ptr` is invalid for asynchronous freeing, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or a live stream-ordered CUDA allocation.
    /// No work may access it after `stream` reaches the enqueued free.
    pub unsafe fn free_async(ptr: *mut T, stream: &Stream) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // SAFETY: `cudaFreeAsync` receives live handles owned by the wrappers.
        unsafe { try_ffi!(runtime::cudaFreeAsync(ptr.cast(), stream.as_raw())) }
    }

    /// Enqueues a typed one-dimensional memory copy on `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA cannot enqueue the copy, the pointer combination is invalid for `kind`, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for `count` elements until `stream` reaches the enqueued copy.
    /// The pointer combination must match `kind`.
    pub unsafe fn copy_async(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: the borrowed stream shares the handle and context; pointer contracts are unchanged.
        unsafe {
            Self::copy_async_on(
                dst,
                src,
                count,
                kind,
                &StreamBinding::Borrowed(stream.to_borrowed()),
            )
        }
    }

    /// Enqueues a typed copy on an explicit, externally borrowed or default stream.
    ///
    /// # Safety
    /// `dst` and `src` must remain valid for `count` elements until the stream reaches the copy.
    /// The pointer combination must match `kind` and the stream's context.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized copy is requested, context binding fails, or CUDA rejects the asynchronous copy.
    pub unsafe fn copy_async_on(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
        stream: &StreamBinding,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }

        stream.context().bind()?;
        // SAFETY: `cudaMemcpyAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemcpyAsync(
                dst.cast(),
                src.cast(),
                bytes as _,
                kind.into(),
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Fills the first `count` bytes of the memory area pointed to by `ptr` with the constant byte `value`.
    ///
    /// [`DeviceMemory::set_async`] is asynchronous with respect to the host, so the call may return before the memset is complete.
    /// The operation can optionally be associated with a stream by passing a non-zero stream argument.
    /// If `stream` is non-zero, the operation may overlap with operations in other streams.
    ///
    /// The device version only handles device-to-device copies and cannot be given local or shared pointers.
    ///
    /// See the CUDA memset synchronization rules for when this operation blocks the host.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, CUDA cannot enqueue the memset on `stream`, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `count * size_of::<T>()` bytes until `stream` reaches the enqueued memset.
    pub unsafe fn set_async(dst: *mut T, value: u8, count: usize, stream: &Stream) -> Result<()> {
        // SAFETY: the borrowed binding retains this stream's context and handle; pointer contracts are unchanged.
        unsafe {
            Self::set_async_on(
                dst,
                value,
                count,
                &StreamBinding::Borrowed(stream.to_borrowed()),
            )
        }
    }

    /// Fills `count` elements with a repeated byte on an explicit, borrowed or default stream.
    /// Supports ordinary enqueue and stream capture without allocating a host source.
    ///
    /// # Errors
    /// Returns an error for byte-count overflow, context binding failure or rejected native enqueue.
    ///
    /// # Safety
    /// `dst` must remain valid and exclusive for `count * size_of::<T>()` writable bytes until completion and during every replay of a captured fill.
    pub unsafe fn set_async_on(
        dst: *mut T,
        value: u8,
        count: usize,
        stream: &StreamBinding,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let bytes = checked_bytes::<T>(count)?;
        if bytes == 0 {
            return Ok(());
        }

        stream.context().bind()?;
        // SAFETY: `cudaMemsetAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemsetAsync(
                dst.cast(),
                value.into(),
                bytes as _,
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Enqueues a 2D pitched memset on `stream`.
    ///
    /// `width` and `height` are measured in elements and rows.
    /// `pitch` is the byte stride between adjacent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte rectangle overflows, `pitch` is smaller than the row width, CUDA cannot enqueue the memset, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must remain valid for writes over the described pitched rectangle until `stream` reaches the enqueued memset.
    pub unsafe fn set_2d_async(
        dst: *mut T,
        pitch: usize,
        value: u8,
        width: usize,
        height: usize,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_len(row_bytes, height)?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > pitch {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: `cudaMemset2DAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemset2DAsync(
                dst.cast(),
                pitch as _,
                value.into(),
                row_bytes as _,
                height as _,
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Enqueues a 3D pitched memset on `stream`.
    ///
    /// `width`, `height`, and `depth` are measured in elements, rows, and slices.
    /// CUDA receives an extent whose width is converted to bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte volume overflows, `dst.pitch` is smaller than the row width, CUDA cannot enqueue the memset, or a previous asynchronous launch reports an error.
    ///
    /// # Safety
    ///
    /// `dst` must remain valid for the described extent until `stream` reaches the enqueued memset.
    pub unsafe fn set_3d_async(
        dst: runtime::cudaPitchedPtr,
        value: u8,
        width: usize,
        height: usize,
        depth: usize,
        stream: &Stream,
    ) -> Result<()> {
        let row_bytes = checked_bytes::<T>(width)?;
        checked_volume(row_bytes, height, depth)?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        if row_bytes > to_usize(dst.pitch, "destination pitch")? {
            return Err(Error::InvalidMemoryAccess);
        }

        // SAFETY: `cudaMemset3DAsync` receives live handles owned by the wrappers.
        unsafe {
            try_ffi!(runtime::cudaMemset3DAsync(
                dst,
                value.into(),
                runtime::cudaExtent {
                    width: row_bytes as _,
                    height: height as _,
                    depth: depth as _,
                },
                stream.as_raw(),
            ))?;
        }
        Ok(())
    }

    /// Prefetches memory to the specified destination location.
    /// `ptr` is the base device pointer of the memory to be prefetched, `location` specifies the destination location, `count` specifies the number of bytes to copy, and `stream` is the stream in which the operation is enqueued.
    /// The memory range must refer to managed memory allocated via [`DeviceMemory::alloc_managed`] or declared via `__managed__` variables.
    /// It may also refer to memory allocated from a managed memory pool, or to system-allocated memory on systems where [`DeviceProperties::pageable_memory_access`](crate::device::DeviceProperties::pageable_memory_access) is enabled.
    ///
    /// Using [`MemoryLocation::Device`] prefetches memory to the GPU identified by its device ID.
    /// That device, and the device associated with `stream`, must support concurrent managed access.
    /// Using [`MemoryLocation::Host`] prefetches data to host memory.
    /// Applications can request prefetching memory to a specific host NUMA node by using [`MemoryLocation::Numa`] with a valid NUMA node identifier, or to the NUMA node closest to the current thread's CPU by using [`MemoryLocation::NumaCurrent`].
    ///
    /// The start and end addresses of the memory range are rounded down and up, respectively, to CPU page-size alignment before the prefetch operation is enqueued in the stream.
    ///
    /// If no physical memory has been allocated for this region, CUDA populates and maps it on the destination device.
    /// If there is insufficient memory to prefetch the desired region, the Unified Memory driver may evict pages from other [`DeviceMemory::alloc_managed`] allocations to host memory to make room.
    /// Device memory allocated using [`DeviceMemory::alloc`] or [`sys::cudaMallocArray`](singe_cuda_sys::runtime::cudaMallocArray) is not evicted.
    ///
    /// By default, mappings to the previous location of the migrated pages are removed and mappings for the new location are only set up at the destination.
    /// The exact behavior also depends on the settings applied to this memory range via `cuMemAdvise` as described below:
    ///
    /// If read-mostly advice was set on any subset of this memory range, then that subset will create a read-only copy of the pages at the destination location.
    /// If the destination location is a host NUMA node, any pages of that subset that are already in another host NUMA node are transferred to the destination.
    ///
    /// If preferred-location advice was set on any subset of this memory range, then the pages will migrate to `location` even if it is not the preferred location of every page in the range.
    ///
    /// If accessed-by advice was set on any subset of this memory range, then mappings to those pages from all appropriate processors are updated to refer to the new location if establishing such a mapping is possible.
    /// Otherwise, those mappings are cleared.
    ///
    /// This is not required for correctness; it improves performance by allowing the application to migrate data to a suitable location before access.
    /// Memory accesses to this range are always coherent and are allowed even when the data is actively being migrated.
    ///
    /// This call is asynchronous with respect to the host and all work on other devices.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the prefetch on `stream`, the memory range or destination location is invalid, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn prefetch_async(
        ptr: DevicePtr,
        count: usize,
        location: MemoryLocation,
        stream: &Stream,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        // SAFETY: `cudaMemPrefetchAsync` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemPrefetchAsync(
                ptr.as_ptr() as _,
                count as _,
                location.into(),
                0, // flags
                stream.as_raw()
            ))?;
        }
        Ok(())
    }

    /// Applies a managed-memory advice policy to a byte range.
    ///
    /// The memory range must refer to managed memory, a managed memory-pool allocation, or pageable memory on systems that support pageable memory access.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA rejects the advice, memory range, or location, or if a previous asynchronous launch reports an error.
    pub fn advise(
        ptr: DevicePtr,
        count: usize,
        advice: MemoryAdvise,
        location: MemoryLocation,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        // SAFETY: `cudaMemAdvise` receives pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(runtime::cudaMemAdvise(
                ptr.as_ptr().cast_const().cast(),
                count as _,
                advice.into(),
                location.into(),
            ))?;
        }
        Ok(())
    }
}

impl<T> DeviceMemory<T> {
    /// Takes ownership of an existing device allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be null for an empty allocation or point to `length` live elements allocated by `cudaMallocManaged` or another CUDA allocation function compatible with `cudaFree`.
    /// `length * size_of::<T>()` must fit in `usize`.
    /// No other owner may free the pointer while the returned value is alive.
    /// The allocation must belong to `context`, and all elements must be initialized before they are read.
    ///
    /// # Panics
    ///
    /// Panics if the byte count overflows or a nonempty zero-sized allocation is described.
    /// Also panics if allocation identities are exhausted; see [`AllocationId::create`].
    pub unsafe fn from_raw_parts(context: &Arc<Context>, ptr: *mut T, length: usize) -> Self {
        Self {
            allocation: Arc::new(DeviceAllocation {
                id: AllocationId::create(),
                context: Arc::clone(context),
                ptr: ptr.cast(),
                bytes: checked_bytes::<T>(length).expect("device memory byte length overflow"),
                deallocation: DeviceDeallocation::Immediate,
            }),
            ptr,
            length,
            _phantom: PhantomData,
        }
    }

    /// Transfers ownership to the caller when no other allocation owners remain.
    ///
    /// Returns the original buffer if submissions, graphs, or foreign users still retain its allocation.
    /// On success the caller owns the pointer and must retain the returned context until the allocation is freed.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw_parts(self) -> StdResult<(*mut T, usize, Arc<Context>), Self> {
        let Self {
            allocation,
            ptr: device_ptr,
            length,
            ..
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(allocation) => {
                let allocation = ManuallyDrop::new(allocation);
                // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
                let context = unsafe { ptr::read(&raw const allocation.context) };
                Ok((device_ptr, length, context))
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr: device_ptr,
                length,
                _phantom: PhantomData,
            }),
        }
    }

    /// Erases initialization guarantees while preserving the allocation and its context.
    #[must_use]
    pub fn into_uninit(self) -> DeviceMemory<MaybeUninit<T>> {
        let Self {
            allocation,
            ptr,
            length,
            ..
        } = self;
        DeviceMemory {
            allocation,
            ptr: ptr.cast(),
            length,
            _phantom: PhantomData,
        }
    }

    /// Allocates uninitialized device storage while retaining its context.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, context binding fails, or CUDA cannot allocate device memory.
    pub fn create_uninit(
        context: &Arc<Context>,
        length: usize,
    ) -> Result<DeviceMemory<MaybeUninit<T>>>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `MaybeUninit<T>` elements need no initialization.
        unsafe { DeviceMemory::<MaybeUninit<T>>::create_unchecked(context, length) }
    }

    /// Allocates typed device storage for a raw producer to initialize.
    ///
    /// Prefer [`Self::create_uninit`] when initialization is represented in the buffer's element type.
    ///
    /// # Safety
    ///
    /// Every element must be initialized to a valid `T` before it is read.
    /// Raw producers and consumers must use the device ABI for `T` and must not transfer Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is requested, context binding fails, or CUDA cannot allocate device memory.
    pub unsafe fn create_unchecked(context: &Arc<Context>, length: usize) -> Result<Self> {
        context.bind()?;
        let bytes = checked_bytes::<T>(length)?;
        let device_ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe { Self::alloc(length)? }
        };
        Ok(Self {
            allocation: Arc::new(DeviceAllocation {
                id: AllocationId::create(),
                context: Arc::clone(context),
                ptr: device_ptr.cast(),
                bytes,
                deallocation: DeviceDeallocation::Immediate,
            }),
            ptr: device_ptr,
            length,
            _phantom: PhantomData,
        })
    }

    /// Allocates device memory and completes zero initialization before returning.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, access-tracking, context-binding, CUDA fill, and default-stream synchronization errors.
    pub fn zeroed(context: &Arc<Context>, length: usize) -> Result<Self>
    where
        T: ZeroableDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let mut mem = unsafe { Self::create_unchecked(context, length)? };
        mem.set_zeroes()?;
        Ok(mem)
    }

    /// Allocates device storage and completes its repeated byte fill before returning.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, access-tracking, context-binding, CUDA fill, and default-stream synchronization errors.
    pub fn filled(context: &Arc<Context>, length: usize, value: u8) -> Result<Self>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: the memory is fully initialized below before it is returned.
        let mut memory = unsafe { Self::create_unchecked(context, length)? };
        memory.set_value(value)?;
        Ok(memory)
    }

    /// Allocates device memory and completes the upload before returning.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub fn from_slice(context: &Arc<Context>, v: &[T]) -> Result<Self>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: DeviceRepr`, so copying the initialized host values yields valid device values.
        unsafe { Self::from_slice_unchecked(context, v) }
    }

    /// Uploads values with an externally established device ABI.
    ///
    /// This supports vendor ABI descriptors whose padding or embedded device addresses are not modeled by `DeviceRepr`.
    ///
    /// # Safety
    ///
    /// `T` must match the device ABI and must not transfer Rust ownership or host references.
    /// Embedded addresses must remain valid for every operation that consumes the uploaded descriptors.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn from_slice_unchecked(context: &Arc<Context>, v: &[T]) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        let mem = unsafe { Self::create_unchecked(context, v.len())? };
        if !v.is_empty() {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                transfer::copy_synchronous(
                    mem.ptr,
                    v.as_ptr(),
                    v.len(),
                    MemoryCopyKind::HostToDevice,
                    None,
                )?;
            }
        }
        Ok(mem)
    }

    /// # Safety
    ///
    /// The caller must ensure `v` remains valid and unmodified until `stream` has completed the transfer.
    /// The uploaded values must match the device ABI without transferring Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate device memory or enqueue the host-to-device copy.
    pub unsafe fn from_slice_async(v: &[T], stream: &Stream) -> Result<Self> {
        // SAFETY: the caller upholds this function's safety contract.
        let mut mem = unsafe { Self::create_unchecked(stream.context(), v.len())? };
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            mem.copy_from_host_async_unchecked(v, stream)?;
        }
        Ok(mem)
    }

    /// Returns the number of elements in the logical allocation range.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the logical allocation range contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the size of the logical allocation range in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the element count multiplied by the element size overflows `usize`.
    /// Allocation construction validates this size, so this condition cannot occur for a valid allocation.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        checked_bytes::<T>(self.length).expect("device memory byte length overflow")
    }

    /// Returns the allocation’s CUDA-accessible element pointer without synchronizing pending work.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the allocation’s CUDA-accessible mutable element pointer.
    ///
    /// The pointer does not establish execution ordering or permit conflicting accesses.
    #[must_use]
    pub const fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Returns the allocation’s address as an untyped CUDA device pointer.
    #[must_use]
    pub const fn as_device_ptr(&self) -> DevicePtr {
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        unsafe { DevicePtr::from_raw(self.ptr.cast()) }
    }

    /// Returns the context that owns this allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the shared owner retained by asynchronous submissions and interop users.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        &self.allocation
    }

    /// Preserves allocation ownership while exposing initialized representation bytes.
    #[must_use]
    pub fn into_bytes(self) -> DeviceMemory<u8>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let length = self.byte_len();
        DeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            _phantom: PhantomData,
        }
    }

    /// Reinterprets initialized storage without reallocating it.
    ///
    /// Returns the original buffer if alignment or byte length is incompatible.
    /// The original allocation owner and deallocation metadata are preserved.
    ///
    /// # Errors
    ///
    /// Returns the original allocation if `U` is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<DeviceMemory<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let Ok(length) = cast_length::<U>(self.region()) else {
            return Err(self);
        };
        Ok(DeviceMemory {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length,
            _phantom: PhantomData,
        })
    }

    fn bind(&self) -> Result<()> {
        self.context().bind()
    }

    fn bind_stream(&self, stream: &Stream) -> Result<()> {
        if self.context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.bind()
    }

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
            return Err(Error::InvalidMemoryAccess);
        }

        if self.length == 0 {
            return Ok(());
        }

        checked_bytes::<T>(self.length)?;
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
    pub fn copy_from_pinned_async<'env, B>(
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
            return Err(Error::InvalidMemoryAccess);
        }

        checked_bytes::<T>(self.length)?;
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
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        host_slice: &'env [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: DeviceRepr,
    {
        let () = T::LAYOUT;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }

        checked_bytes::<T>(self.len())?;
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

    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until `stream` has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if the host and device element counts differ, the stream belongs to another context, or the byte count overflows.
    /// Also propagates context-binding and CUDA asynchronous copy errors.
    pub unsafe fn copy_from_host_async_unchecked(
        &mut self,
        host_slice: &[T],
        stream: &Stream,
    ) -> Result<()> {
        self.bind_stream(stream)?;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }

        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
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

    /// Returns a capture operation that copies from host memory into this device allocation.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores the host and device pointer addresses in the resulting CUDA graph.
    /// The caller must ensure `self` and `host_slice` remain valid whenever a captured graph using this operation is launched.
    /// The destination allocation must remain exclusive for the work ordered by those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if the source and destination element counts differ.
    pub unsafe fn copy_from_host_operation<'a>(
        &'a mut self,
        host_slice: &'a [T],
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe {
            MemoryCopyOperation::new(
                self.as_mut_ptr(),
                host_slice.as_ptr(),
                self.len(),
                MemoryCopyKind::HostToDevice,
            )
        })
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
    /// Also propagates context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn copy_to_host_unchecked(&self, host_slice: &mut [T]) -> Result<()> {
        self.bind()?;
        if host_slice.len() != self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller establishes source value validity.
        unsafe {
            transfer::copy_synchronous(
                host_slice.as_mut_ptr(),
                self.ptr,
                self.length,
                MemoryCopyKind::DeviceToHost,
                None,
            )
        }
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
            return Err(Error::InvalidMemoryAccess);
        }

        if self.length == 0 {
            return Ok(());
        }

        checked_bytes::<T>(self.length)?;
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
    pub fn copy_to_pinned_async<'env, B>(
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
            return Err(Error::InvalidMemoryAccess);
        }

        checked_bytes::<T>(self.length)?;
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
    pub fn copy_to_host_async<'env>(
        &'env self,
        host_slice: &'env mut [T],
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }

        checked_bytes::<T>(self.len())?;
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

    /// # Safety
    ///
    /// The caller must ensure `self` and `host_slice` both remain valid until `stream` has completed the transfer.
    /// Every source element must be initialized and valid for `T`, without Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if the host and device element counts differ, the stream belongs to another context, or the byte count overflows.
    /// Also propagates context-binding and CUDA asynchronous copy errors.
    pub unsafe fn copy_to_host_async_unchecked(
        &self,
        host_slice: &mut [T],
        stream: &Stream,
    ) -> Result<()> {
        self.bind_stream(stream)?;
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }

        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
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

    /// Returns a capture operation that copies this allocation into host memory.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores the device and host pointer addresses in the resulting CUDA graph.
    /// The caller must ensure `self` and `host_slice` remain valid whenever a captured graph using this operation is launched.
    /// The host destination must remain exclusive for the work ordered by those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if the source and destination element counts differ.
    pub unsafe fn copy_to_host_operation<'a>(
        &'a self,
        host_slice: &'a mut [T],
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if host_slice.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe {
            MemoryCopyOperation::new(
                host_slice.as_mut_ptr(),
                self.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToHost,
            )
        })
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
    /// Propagates byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// A nonempty synchronous transfer is rejected during stream capture.
    pub unsafe fn copy_to_host_vec_unchecked(&self) -> Result<Vec<T>> {
        self.bind()?;
        // SAFETY: the caller establishes source value validity.
        unsafe { transfer::copy_to_host_vec_unchecked(self.ptr, self.length, None) }
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
    pub fn copy_from_device_async<'env>(
        &'env mut self,
        src: &'env Self,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        if self.context() != src.context() {
            return Err(Error::MemoryContextMismatch);
        }

        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }

        checked_bytes::<T>(self.len())?;
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

    /// # Safety
    ///
    /// The caller must ensure `self` and `src` both remain valid until `stream` has completed the transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if source and destination contexts or element counts differ, the stream belongs to another context, or the byte count overflows.
    /// Also propagates context-binding and CUDA asynchronous copy errors.
    pub unsafe fn copy_from_device_async_unchecked(
        &mut self,
        src: &Self,
        stream: &Stream,
    ) -> Result<()> {
        if self.context() != src.context() {
            return Err(Error::MemoryContextMismatch);
        }

        self.bind_stream(stream)?;
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }

        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
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

    /// Returns a capture operation that copies from another device allocation into this allocation.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores both device pointer addresses in the resulting CUDA graph.
    /// The caller must ensure `self` and `src` remain valid whenever a captured graph using this operation is launched.
    /// The destination allocation must remain exclusive for the work ordered by those launches.
    ///
    /// # Errors
    ///
    /// Returns an error if the source and destination element counts differ.
    pub unsafe fn copy_from_device_operation<'a>(
        &'a mut self,
        src: &'a Self,
    ) -> Result<MemoryCopyOperation<'a, T>> {
        if src.len() != self.len() {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe {
            MemoryCopyOperation::new(
                self.as_mut_ptr(),
                src.as_ptr(),
                self.len(),
                MemoryCopyKind::DeviceToDevice,
            )
        })
    }

    /// Fills every element with zero bytes and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub fn set_zeroes(&mut self) -> Result<()>
    where
        T: ZeroableDeviceRepr,
    {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.bind()?;
        let () = T::LAYOUT;
        if self.length == 0 {
            return Ok(());
        }

        // SAFETY: the allocation covers `length` elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            Self::set(self.ptr, 0, self.length)?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Fills every byte of the allocation with `value` and waits for completion.
    ///
    /// The byte pattern repeats across each element; `value` is not converted to `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub fn set_value(&mut self, value: u8) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so repeating `value` produces valid `T` values.
        unsafe { self.set_value_unchecked(value) }
    }

    /// Fills storage with a byte pattern known to be valid for `T`.
    ///
    /// # Safety
    ///
    /// The repeated byte pattern must represent a valid value of `T`.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking, context-binding, byte-size, CUDA fill, and default-stream synchronization errors.
    pub unsafe fn set_value_unchecked(&mut self, value: u8) -> Result<()> {
        let context = Arc::clone(self.context());
        let _access = context.acquire_host_access(&[self.region()])?;
        self.bind()?;
        if self.length == 0 {
            return Ok(());
        }

        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            Self::set(self.ptr, value, self.length)?;
        }
        StreamBinding::Default(Arc::clone(self.context())).synchronize()
    }

    /// Enqueues a repeated-byte fill of the entire allocation in a stream scope.
    ///
    /// The scope retains the exclusive allocation borrow until the fill completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows or the scope’s stream belongs to another context.
    /// Also propagates context-binding, scope submission, and CUDA fill errors.
    pub fn set_value_async<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;

        checked_bytes::<T>(self.len())?;
        self.bind_stream(stream.stream())?;
        if self.is_empty() {
            return Ok(());
        }

        let accesses = [Access::new(self.region(), AccessMode::Write)];
        // SAFETY: the declared accesses retain both regions, and the stream orders every access to them.
        unsafe {
            stream.enqueue(&accesses, || {
                Self::set_async(self.ptr, value, self.len(), stream.stream())
            })
        }
    }

    /// # Safety
    ///
    /// The caller must ensure `self` remains valid until `stream` has completed the memset.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot enqueue the memset on `stream`.
    pub unsafe fn set_value_async_unchecked(&mut self, value: u8, stream: &Stream) -> Result<()> {
        self.bind_stream(stream)?;
        if self.is_empty() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::set_async(self.as_mut_ptr(), value, self.len(), stream) }
    }

    /// Returns a capture operation that fills this device allocation with `value`.
    ///
    /// # Safety
    ///
    /// Capturing this operation stores this allocation's pointer address in the resulting CUDA graph.
    /// The caller must ensure `self` remains valid and exclusive whenever a captured graph using this operation is launched.
    pub unsafe fn set_value_operation(&mut self, value: u8) -> MemorySetOperation<'_, T> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { MemorySetOperation::new(self.as_mut_ptr(), value, self.len()) }
    }

    /// Takes a pointer to the base of an existing device memory allocation created with [`DeviceMemory::alloc`] and exports it for use in another process.
    /// This is a lightweight operation and may be called multiple times on an allocation without adverse effects.
    ///
    /// If a region of memory is freed with [`DeviceMemory::free`] and a subsequent call to [`DeviceMemory::alloc`] returns memory with the same device address, [`DeviceMemory::ipc_handle`] returns a unique handle for the new memory.
    ///
    /// IPC is restricted to devices with unified-addressing support on Linux and Windows.
    /// IPC on Windows is supported for compatibility but is not recommended because of its performance cost.
    /// Check device IPC support through the device properties exposed by this crate, for example [`DeviceProperties::ipc_event_supported`](crate::device::DeviceProperties::ipc_event_supported).
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation is empty, CUDA cannot export an IPC handle for the allocation, or CUDA reports runtime initialization diagnostics.
    pub fn ipc_handle(&self) -> Result<IpcMemoryHandle> {
        self.bind()?;
        if self.is_empty() {
            // Cannot get handle for null pointer / zero size? Check docs.
            return Err(Error::InvalidMemoryAccess);
        }

        let mut handle = MaybeUninit::uninit();
        // SAFETY: `cudaIpcGetMemHandle` writes only the local handle for this live allocation.
        unsafe {
            try_ffi!(runtime::cudaIpcGetMemHandle(
                handle.as_mut_ptr(),
                self.as_ptr().cast_mut().cast(),
            ))?;
            Ok(IpcMemoryHandle::from_raw(handle.assume_init()))
        }
    }

    /// Allocates independent device storage and copies all elements into it.
    ///
    /// Returns after the device-to-device copy completes.
    ///
    /// # Errors
    ///
    /// Propagates allocation, byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    pub fn try_clone(&self) -> Result<Self> {
        // SAFETY: the memory is fully initialized below before it is returned.
        let new_mem = unsafe { Self::create_unchecked(self.context(), self.length)? };
        if self.length == 0 || size_of::<T>() == 0 {
            return Ok(new_mem);
        }

        // SAFETY: the guards acquired above cover both regions, and the lengths were checked to match.
        unsafe {
            transfer::copy_synchronous(
                new_mem.as_mut_ptr(),
                self.as_ptr(),
                self.length,
                MemoryCopyKind::DeviceToDevice,
                None,
            )?;
        }

        Ok(new_mem)
    }
}

impl<T> DeviceMemory<MaybeUninit<T>> {
    /// Exposes an initialized prefix while retaining the complete allocation.
    ///
    /// # Safety
    ///
    /// The first `length` elements must contain valid initialized values of `T`.
    /// Producing work must be ordered before every consumer and host read.
    /// The remainder stays allocated but is no longer exposed as typed elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<DeviceMemory<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }

        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        Ok(unsafe { self.assume_init() })
    }

    /// Converts storage initialized by a device producer into initialized device memory.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid, initialized `T` before any consumer reads it.
    /// Pending initialization must be ordered before every subsequent consumer and host readback.
    #[must_use]
    pub unsafe fn assume_init(self) -> DeviceMemory<T> {
        let Self {
            allocation,
            ptr,
            length,
            ..
        } = self;
        DeviceMemory {
            allocation,
            ptr: ptr.cast(),
            length,
            _phantom: PhantomData,
        }
    }
}
