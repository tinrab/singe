use std::ptr;

use singe_cuda_sys::runtime;

use super::DeviceMemory;
use crate::{
    context::Context,
    error::{Error, Result},
    memory::{MemoryAttachFlags, MemoryCopyKind, MemoryLocation, PointerAttributes},
    stream::{AsStream, Stream},
    try_ffi,
    types::{DevicePtr, MemoryAdvise},
    utility::{checked_bytes, checked_element_count, to_usize},
};

/// Associated utility functions.
impl<T> DeviceMemory<T> {
    /// Allocates uninitialized linear device memory for `count` elements and returns its pointer.
    ///
    /// The memory is aligned for any element type and is not cleared.
    /// An allocation failure returns [`crate::error::Status::OutOfMemory`].
    /// A zero-byte request returns a null pointer without calling CUDA.
    ///
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory.
    /// The caller must use it only for `count` elements of `T` and eventually free it with a compatible CUDA free function.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte size overflows, CUDA cannot allocate device memory, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics such as [`crate::error::Status::NotInitialized`], [`crate::error::Status::CallRequiresNewerDriver`], or [`crate::error::Status::NoDevice`].
    pub unsafe fn alloc(context: &Context, count: usize) -> Result<*mut T> {
        context.bind()?;
        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }

        let mut p = ptr::null_mut();
        // SAFETY: `p` is a local output, and `bytes` is a checked nonzero size.
        unsafe {
            try_ffi!(runtime::cudaMalloc(&raw mut p, bytes as _))?;
        }
        Ok(p.cast())
    }

    /// Allocates CUDA managed memory for `count` elements.
    ///
    /// `flags` controls the initial stream attachment behavior for the managed allocation.
    ///
    /// # Safety
    ///
    /// The returned pointer is managed CUDA memory.
    /// The caller must eventually free it with a CUDA-compatible device free and must synchronize host and device access according to CUDA managed-memory rules.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte size overflows, CUDA cannot allocate managed memory, or `count` is non-zero for a zero-sized `T`.
    pub unsafe fn alloc_managed(
        context: &Context,
        count: usize,
        flags: MemoryAttachFlags,
    ) -> Result<*mut T> {
        context.bind()?;
        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }

        let mut p = ptr::null_mut();
        // SAFETY: `p` is a local output, and `bytes` is a checked nonzero size.
        unsafe {
            try_ffi!(runtime::cudaMallocManaged(
                &raw mut p,
                bytes as _,
                flags.bits(),
            ))?;
        }
        Ok(p.cast::<T>())
    }

    /// Frees the device allocation at `ptr`.
    ///
    /// This does not synchronize when the pointer came from the stream-ordered allocator, such as [`DeviceMemory::alloc_async`].
    /// For best performance and memory reuse, free such memory with [`DeviceMemory::free_async`] instead.
    /// For all other pointers, this call may synchronize implicitly.
    ///
    /// If `ptr` is null, no operation is performed.
    ///
    /// # Safety
    ///
    /// `ptr` must be null or a live allocation of `context` returned by a CUDA device allocation function, and not freed before.
    /// All work that accesses it must have completed, no work may access it afterwards, and no allocation owner may free it again.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, CUDA cannot free `ptr`, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub unsafe fn free(context: &Context, ptr: *mut T) -> Result<()> {
        context.bind()?;
        // SAFETY: the caller passes null or a live CUDA allocation that no work uses any more.
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
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory.
    /// The caller must use it only as a `width` by `height` pitched allocation with the returned pitch and eventually free it with a CUDA-compatible device free.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte size overflows, `T` is a non-empty zero-sized allocation, CUDA cannot allocate pitched memory, or CUDA returns a null allocation for a non-empty request.
    pub unsafe fn alloc_pitched(
        context: &Context,
        width: usize,
        height: usize,
    ) -> Result<(*mut T, usize)> {
        context.bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height], "rectangle bytes")?;
        if row_bytes == 0 || height == 0 {
            return Ok((ptr::null_mut(), row_bytes));
        }

        let mut ptr = ptr::null_mut();
        let mut pitch = 0;
        // SAFETY: `ptr` and `pitch` are local outputs, and the rectangle size was checked for overflow.
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
    /// # Safety
    ///
    /// The returned allocation is uninitialized device memory.
    /// The caller must treat it as a `width` by `height` by `depth` pitched allocation and free the base pointer with a CUDA-compatible device free.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte volume overflows, `T` is a non-empty zero-sized allocation, CUDA cannot allocate pitched memory, or CUDA returns a null allocation for a non-empty request.
    pub unsafe fn alloc_pitched_3d(
        context: &Context,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<runtime::cudaPitchedPtr> {
        context.bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height, depth], "volume bytes")?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(runtime::cudaPitchedPtr {
                ptr: ptr::null_mut(),
                pitch: row_bytes as _,
                xsize: row_bytes as _,
                ysize: height as _,
            });
        }

        let mut pitched = runtime::cudaPitchedPtr::default();
        // SAFETY: `pitched` is a local output, and the volume size was checked for overflow.
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
    /// The copy is synchronous with respect to the host for most pointer combinations.
    /// Each range must be entirely registered with CUDA or, for pageable host transfers, not registered at all.
    /// A range spanning registered and unregistered allocations returns [`crate::error::Status::InvalidValue`].
    ///
    /// # Safety
    ///
    /// Device operands must belong to `context` or be accessible from it.
    /// `src` and `dst` must be valid for `count` elements of `T` according to `kind`, and the source and destination regions must not overlap unless CUDA permits that transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte count overflows, CUDA rejects the pointer combination or copy kind, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub unsafe fn copy(
        context: &Context,
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        context.bind()?;
        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(());
        }

        // SAFETY: the caller guarantees `count` valid elements at both pointers for `kind`, and `bytes` was checked.
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
    /// # Safety
    ///
    /// Device operands must belong to `context` or be accessible from it.
    /// `dst` and `src` must be valid for the described rectangle according to `kind`.
    /// The source and destination regions must not overlap unless CUDA permits that transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte rectangle overflows, either pitch is smaller than the copied row width, CUDA rejects the pointer combination or copy kind, or a previous asynchronous launch reports an error.
    pub unsafe fn copy_2d(
        context: &Context,
        dst: *mut T,
        dst_pitch: usize,
        src: *const T,
        src_pitch: usize,
        width: usize,
        height: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        context.bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height], "rectangle bytes")?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > dst_pitch {
            return Err(Error::buffer_too_small(
                "destination pitch",
                row_bytes,
                dst_pitch,
            ));
        }

        if row_bytes > src_pitch {
            return Err(Error::buffer_too_small(
                "source pitch",
                row_bytes,
                src_pitch,
            ));
        }

        // SAFETY: the caller guarantees both rectangles for `kind`, and both pitches were checked against the row width.
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
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for the described rectangle until `stream` reaches the enqueued copy.
    /// The pointer combination must match `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, the requested byte rectangle overflows, either pitch is smaller than the copied row width, CUDA cannot enqueue the copy, or a previous asynchronous launch reports an error.
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
        stream.context().bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height], "rectangle bytes")?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > dst_pitch {
            return Err(Error::buffer_too_small(
                "destination pitch",
                row_bytes,
                dst_pitch,
            ));
        }

        if row_bytes > src_pitch {
            return Err(Error::buffer_too_small(
                "source pitch",
                row_bytes,
                src_pitch,
            ));
        }

        // SAFETY: the caller keeps both rectangles valid until `stream` reaches the copy, and both pitches were checked.
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
    /// # Safety
    ///
    /// Device operands must belong to `context` or be accessible from it.
    /// `dst` and `src` must describe live host or device memory large enough for the requested extent according to `kind`.
    /// The source and destination regions must not overlap unless CUDA permits that transfer.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte volume overflows, either pitch is smaller than the copied row width, CUDA rejects the copy parameters, or a previous asynchronous launch reports an error.
    pub unsafe fn copy_3d(
        context: &Context,
        dst: runtime::cudaPitchedPtr,
        src: runtime::cudaPitchedPtr,
        width: usize,
        height: usize,
        depth: usize,
        kind: MemoryCopyKind,
    ) -> Result<()> {
        context.bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height, depth], "volume bytes")?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        let dst_pitch = to_usize(dst.pitch, "destination pitch")?;
        if row_bytes > dst_pitch {
            return Err(Error::buffer_too_small(
                "destination pitch",
                row_bytes,
                dst_pitch,
            ));
        }

        let src_pitch = to_usize(src.pitch, "source pitch")?;
        if row_bytes > src_pitch {
            return Err(Error::buffer_too_small(
                "source pitch",
                row_bytes,
                src_pitch,
            ));
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
        // SAFETY: `params` is a local, the caller guarantees both extents for `kind`, and both pitches were checked.
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
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for the described extent until `stream` reaches the enqueued copy.
    /// The pointer combination must match `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, the requested byte volume overflows, either pitch is smaller than the copied row width, CUDA cannot enqueue the copy, or a previous asynchronous launch reports an error.
    pub unsafe fn copy_3d_async(
        dst: runtime::cudaPitchedPtr,
        src: runtime::cudaPitchedPtr,
        width: usize,
        height: usize,
        depth: usize,
        kind: MemoryCopyKind,
        stream: &Stream,
    ) -> Result<()> {
        stream.context().bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height, depth], "volume bytes")?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        let dst_pitch = to_usize(dst.pitch, "destination pitch")?;
        if row_bytes > dst_pitch {
            return Err(Error::buffer_too_small(
                "destination pitch",
                row_bytes,
                dst_pitch,
            ));
        }

        let src_pitch = to_usize(src.pitch, "source pitch")?;
        if row_bytes > src_pitch {
            return Err(Error::buffer_too_small(
                "source pitch",
                row_bytes,
                src_pitch,
            ));
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
        // SAFETY: `params` is a local read during the call, the caller keeps both extents valid until `stream` reaches the copy, and both pitches were checked.
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
    /// # Safety
    ///
    /// Device operands must belong to `context` or be accessible from it.
    /// `dst` must be valid for writes of `count * size_of::<T>()` bytes and must refer to memory that CUDA can memset.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte count overflows, CUDA rejects the pointer or size, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub unsafe fn set(context: &Context, dst: *mut T, value: u8, count: usize) -> Result<()> {
        context.bind()?;
        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(());
        }

        // SAFETY: the caller guarantees `bytes` writable bytes at `dst`.
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
    /// # Safety
    ///
    /// Device operands must belong to `context` or be accessible from it.
    /// `dst` must be valid for writes over the described pitched rectangle and must refer to memory CUDA can memset.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte rectangle overflows, `pitch` is smaller than the row width, CUDA rejects the memory region, or a previous asynchronous launch reports an error.
    pub unsafe fn set_2d(
        context: &Context,
        dst: *mut T,
        pitch: usize,
        value: u8,
        width: usize,
        height: usize,
    ) -> Result<()> {
        context.bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height], "rectangle bytes")?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > pitch {
            return Err(Error::buffer_too_small("pitch", row_bytes, pitch));
        }

        // SAFETY: the caller guarantees the writable rectangle, and `pitch` was checked against the row width.
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
    /// # Safety
    ///
    /// Device operands must belong to `context` or be accessible from it.
    /// `dst` must describe live memory large enough for the requested extent and must refer to memory CUDA can memset.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound, the requested byte volume overflows, `dst.pitch` is smaller than the row width, CUDA rejects the memory region, or a previous asynchronous launch reports an error.
    pub unsafe fn set_3d(
        context: &Context,
        dst: runtime::cudaPitchedPtr,
        value: u8,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<()> {
        context.bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height, depth], "volume bytes")?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        let dst_pitch = to_usize(dst.pitch, "destination pitch")?;
        if row_bytes > dst_pitch {
            return Err(Error::buffer_too_small(
                "destination pitch",
                row_bytes,
                dst_pitch,
            ));
        }

        // SAFETY: the caller guarantees the writable extent, and `dst.pitch` was checked against the row width.
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

    /// Returns the free and total device memory, in bytes, seen from `context`.
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
    /// Returns an error if `context` cannot be bound, CUDA cannot query memory information, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn memory_info(context: &Context) -> Result<(usize, usize)> {
        context.bind()?;
        let mut free: runtime::size_t = 0;
        let mut total: runtime::size_t = 0;
        // SAFETY: `free` and `total` are local outputs.
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
    /// In CUDA 11.0 and later, passing a host pointer reports [`MemoryType::Unregistered`](crate::memory::MemoryType::Unregistered) in [`PointerAttributes::memory_type`].
    ///
    /// * [`PointerAttributes::memory_type`] identifies the type of memory.
    ///   It can be [`MemoryType::Unregistered`](crate::memory::MemoryType::Unregistered) for unregistered host memory, [`MemoryType::Host`](crate::memory::MemoryType::Host) for registered host memory, [`MemoryType::Device`](crate::memory::MemoryType::Device) for device memory, or [`MemoryType::Managed`](crate::memory::MemoryType::Managed) for managed memory.
    ///
    /// * [`PointerAttributes::device`] is the device against which `ptr` was allocated.
    ///   If `ptr` has memory type [`MemoryType::Device`](crate::memory::MemoryType::Device), this identifies the device on which the memory physically resides.
    ///   If `ptr` has memory type [`MemoryType::Host`](crate::memory::MemoryType::Host), this identifies the device that was current when the allocation was made, and if that device is deinitialized then this allocation will vanish with that device's state.
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
    /// # Safety
    ///
    /// The returned pointer is uninitialized device memory whose lifetime is ordered by `stream`.
    /// The caller must ensure all accesses are stream ordered correctly and free it with [`DeviceMemory::free_async`] or a compatible CUDA free after all accesses complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, the requested byte size overflows, CUDA cannot enqueue the allocation, or `count` is non-zero for a zero-sized `T`.
    pub unsafe fn alloc_async(count: usize, stream: &Stream) -> Result<*mut T> {
        stream.context().bind()?;
        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }

        let mut p = ptr::null_mut();
        // SAFETY: `p` is a local output, `bytes` is a checked nonzero size, and `stream` is borrowed live.
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
    /// # Safety
    ///
    /// `ptr` must be null or a live stream-ordered CUDA allocation.
    /// No work may access it after `stream` reaches the enqueued free.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, CUDA cannot enqueue the free on `stream`, `ptr` is invalid for asynchronous freeing, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub unsafe fn free_async(ptr: *mut T, stream: &Stream) -> Result<()> {
        stream.context().bind()?;
        if ptr.is_null() {
            return Ok(());
        }
        // SAFETY: the caller passes a live stream-ordered allocation that no work uses after `stream` reaches this free.
        unsafe { try_ffi!(runtime::cudaFreeAsync(ptr.cast(), stream.as_raw())) }
    }

    /// Enqueues a typed one-dimensional memory copy on `stream`.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must remain valid for `count` elements until `stream` reaches the enqueued copy, and during every replay of a captured copy.
    /// The pointer combination must match `kind` and the stream's context.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, the context cannot be bound, CUDA cannot enqueue the copy, the pointer combination is invalid for `kind`, or a previous asynchronous launch reports an error.
    pub unsafe fn copy_async(
        dst: *mut T,
        src: *const T,
        count: usize,
        kind: MemoryCopyKind,
        stream: &(impl AsStream + ?Sized),
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(());
        }

        stream.context().bind()?;
        // SAFETY: the caller keeps `count` elements at both pointers valid until the stream reaches the copy, and the stream's context is bound.
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

    /// Enqueues a fill of `count` elements at `dst` with the byte `value` on `stream`.
    ///
    /// The call may return before the fill completes, and the fill may overlap work on other streams.
    /// See the CUDA memset synchronization rules for when this operation blocks the host.
    ///
    /// # Safety
    ///
    /// `dst` must remain valid and exclusive for writes of `count * size_of::<T>()` bytes until `stream` reaches the enqueued memset, and during every replay of a captured fill.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested byte count overflows, the context cannot be bound, CUDA cannot enqueue the memset on `stream`, or a previous asynchronous launch reports an error.
    pub unsafe fn set_async(
        dst: *mut T,
        value: u8,
        count: usize,
        stream: &(impl AsStream + ?Sized),
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let bytes = checked_bytes::<T>(count, "byte count")?;
        if bytes == 0 {
            return Ok(());
        }

        stream.context().bind()?;
        // SAFETY: the caller keeps `bytes` bytes at `dst` writable until the stream reaches the fill, and the stream's context is bound.
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
    /// # Safety
    ///
    /// `dst` must remain valid for writes over the described pitched rectangle until `stream` reaches the enqueued memset.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, the requested byte rectangle overflows, `pitch` is smaller than the row width, CUDA cannot enqueue the memset, or a previous asynchronous launch reports an error.
    pub unsafe fn set_2d_async(
        dst: *mut T,
        pitch: usize,
        value: u8,
        width: usize,
        height: usize,
        stream: &Stream,
    ) -> Result<()> {
        stream.context().bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height], "rectangle bytes")?;
        if row_bytes == 0 || height == 0 {
            return Ok(());
        }

        if row_bytes > pitch {
            return Err(Error::buffer_too_small("pitch", row_bytes, pitch));
        }

        // SAFETY: the caller keeps the rectangle writable until `stream` reaches the fill, and `pitch` was checked against the row width.
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
    /// # Safety
    ///
    /// `dst` must remain valid for the described extent until `stream` reaches the enqueued memset.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream context cannot be bound, the requested byte volume overflows, `dst.pitch` is smaller than the row width, CUDA cannot enqueue the memset, or a previous asynchronous launch reports an error.
    pub unsafe fn set_3d_async(
        dst: runtime::cudaPitchedPtr,
        value: u8,
        width: usize,
        height: usize,
        depth: usize,
        stream: &Stream,
    ) -> Result<()> {
        stream.context().bind()?;
        let row_bytes = checked_bytes::<T>(width, "row bytes")?;
        checked_element_count(&[row_bytes, height, depth], "volume bytes")?;
        if row_bytes == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        let dst_pitch = to_usize(dst.pitch, "destination pitch")?;
        if row_bytes > dst_pitch {
            return Err(Error::buffer_too_small(
                "destination pitch",
                row_bytes,
                dst_pitch,
            ));
        }

        // SAFETY: the caller keeps the extent writable until `stream` reaches the fill, and `dst.pitch` was checked against the row width.
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

    /// Enqueues a prefetch of `count` bytes at `ptr` to `location` on `stream`.
    ///
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
    /// The exact behavior also depends on the advice applied to this memory range with [`DeviceMemory::advise`]:
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
    /// Returns an error if the stream context cannot be bound, CUDA cannot enqueue the prefetch on `stream`, the memory range or destination location is invalid, a previous asynchronous launch reports an error, or CUDA reports runtime initialization diagnostics.
    pub fn prefetch(
        ptr: DevicePtr,
        count: usize,
        location: MemoryLocation,
        stream: &Stream,
    ) -> Result<()> {
        stream.context().bind()?;
        if count == 0 {
            return Ok(());
        }

        // SAFETY: prefetching migrates pages without changing their contents, CUDA rejects ranges it does not manage, and `stream` is borrowed live.
        unsafe {
            try_ffi!(runtime::cudaMemPrefetchAsync(
                ptr.as_const_ptr().cast(),
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

        // SAFETY: advice changes only migration policy, not contents, and CUDA rejects ranges it does not manage.
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
