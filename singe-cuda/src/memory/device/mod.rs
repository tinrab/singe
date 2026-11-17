mod copy;
mod raw;

use std::{
    marker::PhantomData,
    mem::{MaybeUninit, replace},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use singe_cuda_sys::runtime;

use crate::{
    context::Context,
    error::{Error, Result},
    ipc::IpcMemoryHandle,
    memory::{DeviceAllocation, DeviceDeallocation, MemoryCopyKind, StreamOrderedFree, transfer},
    stream::{AsStream, Stream, StreamBinding, StreamScope},
    submission::{Access, AccessMode},
    try_ffi,
    types::DevicePtr,
    utility::checked_bytes,
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceOutput, DeviceRange, DeviceRegion,
        DeviceRegionProvider, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr, cast_length,
    },
};

impl<T> DeviceMemory<T> {
    /// Takes ownership of an existing device allocation.
    ///
    /// # Safety
    ///
    /// `raw.ptr` must be null for an empty allocation or point to `raw.len` live elements of `raw.context`.
    /// Without `raw.stream_ordered`, the allocation must be valid to free with `cudaFree`; with it, it must come from `cudaMallocAsync` or `cudaMallocFromPoolAsync` and be valid to free on its stream.
    /// No other owner may free the pointer while the returned value is alive, and all elements must be initialized before they are read.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count overflows, a nonempty zero-sized allocation is described, or a nonempty allocation has a null pointer.
    /// Returns `Error::StreamContextMismatch` if the freeing stream belongs to another context.
    pub unsafe fn from_raw(raw: RawDeviceMemory<T>) -> Result<Self> {
        let RawDeviceMemory {
            ptr,
            len,
            context,
            stream_ordered,
        } = raw;
        let bytes = checked_bytes::<T>(len, "byte length")?;
        if bytes != 0 && ptr.is_null() {
            return Err(Error::null_pointer("device memory"));
        }

        let deallocation = match stream_ordered {
            None => DeviceDeallocation::Immediate,
            Some(StreamOrderedFree { stream, pool }) => {
                if stream.context() != &context {
                    return Err(Error::StreamContextMismatch);
                }

                DeviceDeallocation::StreamOrdered { stream, pool }
            }
        };
        Ok(Self {
            allocation: DeviceAllocation::from_parts(&context, ptr.cast(), bytes, deallocation),
            ptr,
            length: len,
            _phantom: PhantomData,
        })
    }

    /// Transfers ownership to the caller when no other allocation owners remain.
    ///
    /// On success the caller owns the pointer and must retain the returned context until the allocation is freed.
    ///
    /// # Errors
    ///
    /// Returns the original wrapper if another owner, submission, graph, or foreign handle retains the allocation.
    pub fn into_raw(self) -> StdResult<RawDeviceMemory<T>, Self> {
        let Self {
            allocation,
            ptr,
            length,
            ..
        } = self;
        match Arc::try_unwrap(allocation) {
            Ok(mut allocation) => {
                let context = Arc::clone(allocation.context());
                allocation.ptr = ptr::null_mut();
                let stream_ordered =
                    match replace(&mut allocation.deallocation, DeviceDeallocation::Immediate) {
                        DeviceDeallocation::Immediate => None,
                        DeviceDeallocation::StreamOrdered { stream, pool } => {
                            Some(StreamOrderedFree { stream, pool })
                        }
                    };
                Ok(RawDeviceMemory {
                    ptr,
                    len: length,
                    context,
                    stream_ordered,
                })
            }
            Err(allocation) => Err(Self {
                allocation,
                ptr,
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
    /// Write every element, then call [`DeviceMemory::assume_init`] to expose them as `T`.
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
        DeviceMemory::<MaybeUninit<T>>::allocate(context, length)
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
        let mut memory = Self::create_uninit(context, length)?;
        memory.set_zeroes()?;
        // SAFETY: `set_zeroes` completed a zero fill of every element, and an all-zero `T` is valid.
        Ok(unsafe { memory.assume_init() })
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
        let mut memory = Self::create_uninit(context, length)?;
        memory.set_value(value)?;
        // SAFETY: `set_value` completed a fill of every element, and every bit pattern is a valid `T`.
        Ok(unsafe { memory.assume_init() })
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
        let memory = DeviceMemory::<MaybeUninit<T>>::allocate(context, v.len())?;
        if !v.is_empty() {
            // SAFETY: `memory` holds `v.len()` fresh elements, `v` is a live host slice of that length, and the caller vouches for `T`'s device ABI.
            unsafe {
                transfer::copy_synchronous(
                    memory.ptr.cast::<T>(),
                    v.as_ptr(),
                    v.len(),
                    MemoryCopyKind::HostToDevice,
                    None,
                )?;
            }
        }
        // SAFETY: the completed copy wrote every element from `v`.
        Ok(unsafe { memory.assume_init() })
    }

    /// Allocates device memory and enqueues an upload of `v` on `stream`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `v` remains valid and unmodified until `stream` has completed the transfer.
    /// The uploaded values must match the device ABI without transferring Rust ownership or host references.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate device memory or enqueue the host-to-device copy.
    pub unsafe fn from_slice_async(v: &[T], stream: &Stream) -> Result<Self> {
        let memory = DeviceMemory::<MaybeUninit<T>>::allocate(stream.context(), v.len())?;
        // SAFETY: the copy enqueued below initializes every element before `stream` runs any later consumer, and a failed enqueue drops the memory unread.
        let mut mem = unsafe { memory.assume_init() };
        // SAFETY: the caller keeps `v` live and unmodified until `stream` completes, and `mem` lives in the returned owner.
        unsafe {
            mem.copy_from_host_async(v, stream)?;
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
        checked_bytes::<T>(self.length, "byte length").expect("device memory byte length overflow")
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
        // SAFETY: `ptr` is this live allocation's base or null, and `DevicePtr` only records the address.
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

        // SAFETY: the host access guard excludes other work on the `length` owned elements, and `T: ZeroableDeviceRepr` makes the zero fill valid.
        unsafe {
            Self::set(self.context(), self.ptr, 0, self.length)?;
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

        // SAFETY: the host access guard excludes other work on the `length` owned elements, and the caller vouches for the byte pattern.
        unsafe {
            Self::set(self.context(), self.ptr, value, self.length)?;
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
    pub fn set_value_scoped<'env>(
        &'env mut self,
        value: u8,
        stream: &StreamScope<'_, 'env>,
    ) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;

        checked_bytes::<T>(self.len(), "byte length")?;
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

    /// Enqueues a repeated-byte fill of this allocation on `stream`, ordered after the allocation's producer.
    ///
    /// # Safety
    ///
    /// The caller must ensure `self` remains valid until `stream` has completed the memset.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another context, producer ordering fails, or CUDA cannot enqueue the memset on `stream`.
    pub unsafe fn set_value_async(&mut self, value: u8, stream: &Stream) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;

        self.bind_stream(stream)?;
        if self.is_empty() {
            return Ok(());
        }

        stream.join_regions(&[self.region()])?;
        // SAFETY: the stream shares this allocation's context and is ordered after its producer, every byte pattern is a valid `T`, and the caller keeps the allocation valid until `stream` completes the fill.
        unsafe { Self::set_async(self.as_mut_ptr(), value, self.len(), stream) }
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
            // An empty allocation has no base address for CUDA to export.
            return Err(Error::empty("device allocation"));
        }

        let mut handle = MaybeUninit::uninit();
        // SAFETY: `handle` is a local output, the allocation is live and nonempty, and success initializes the handle.
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
        let new_mem = DeviceMemory::<MaybeUninit<T>>::allocate(self.context(), self.length)?;
        if self.length == 0 || size_of::<T>() == 0 {
            // SAFETY: an empty or zero-sized range has no bytes to initialize.
            return Ok(unsafe { new_mem.assume_init() });
        }

        let source = self.region();
        // Waits for tracked stream work that still writes or reads the source.
        let _access = self.context().acquire_host_access(&[source])?;
        self.context().bind()?;
        // SAFETY: `new_mem` is a fresh allocation of `self.length` elements disjoint from `self`, the context is bound, and the access guard ordered the source's tracked producers before this copy.
        unsafe {
            transfer::copy_synchronous(
                new_mem.as_mut_ptr().cast::<T>(),
                self.as_ptr(),
                self.length,
                MemoryCopyKind::DeviceToDevice,
                source.stream(),
            )?;
        }

        // SAFETY: the completed copy wrote every element from `self`.
        Ok(unsafe { new_mem.assume_init() })
    }
}

impl<T> DeviceMemory<MaybeUninit<T>> {
    // Allocates `length` elements whose storage stays uninitialized, which `MaybeUninit<T>` permits.
    fn allocate(context: &Arc<Context>, length: usize) -> Result<Self> {
        context.bind()?;
        let bytes = checked_bytes::<T>(length, "byte length")?;
        let device_ptr = if bytes == 0 {
            ptr::null_mut()
        } else {
            // SAFETY: the allocation is owned below and freed with `cudaFree`, and its `MaybeUninit` elements need no initialization.
            unsafe { Self::alloc(context, length)? }
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
            return Err(Error::buffer_too_small("allocation", length, self.length));
        }

        self.length = length;
        // SAFETY: the caller vouches that the first `length` elements, now the whole exposed range, are valid `T` values.
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

// SAFETY: the typed range lies inside the retained allocation, so its region is a slice of the allocation's region.
unsafe impl<T> DeviceRegionProvider for DeviceMemory<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let offset = self.ptr.addr() - self.allocation.ptr.addr();
        self.allocation
            .region()
            .slice(offset..offset + self.byte_len())
            .expect("device buffer exceeds its allocation")
    }
}

/// Raw ownership of a device allocation, its context, and how it is freed.
///
/// Free `ptr` with `cudaFree` while `context` is alive, or with `cudaFreeAsync` on the stream in `stream_ordered`, or re-adopt the value with [`DeviceMemory::from_raw`].
#[derive(Debug)]
pub struct RawDeviceMemory<T> {
    /// Base of the allocation, or null for an empty one.
    pub ptr: *mut T,
    /// Number of elements of `T`.
    pub len: usize,
    /// Context the allocation belongs to.
    pub context: Arc<Context>,
    /// The stream and pool of a stream-ordered allocation, or `None` for one freed with `cudaFree`.
    pub stream_ordered: Option<StreamOrderedFree>,
}

/// Represents a region of owned CUDA device memory for elements of type `T`.
///
/// The last owner of the allocation frees it with `cudaFree`, or, for a stream-ordered allocation, enqueues `cudaFreeAsync` on its stream.
#[derive(Debug)]
pub struct DeviceMemory<T> {
    pub(crate) allocation: Arc<DeviceAllocation>,
    /// Raw pointer to the allocated device memory.
    pub(crate) ptr: *mut T,
    /// Number of elements of type `T` allocated.
    pub(crate) length: usize,
    /// Marker for the type `T`.
    pub(crate) _phantom: PhantomData<T>,
}

impl<T> PartialEq for DeviceMemory<T> {
    fn eq(&self, other: &Self) -> bool {
        self.context() == other.context() && self.ptr == other.ptr && self.length == other.length
    }
}

impl<T> Eq for DeviceMemory<T> {}

// SAFETY: the host never dereferences `ptr`, the owner holds `T` values like a `Vec<T>`, and the shared `DeviceAllocation` is `Send`.
unsafe impl<T: Send> Send for DeviceMemory<T> {}

// SAFETY: `&DeviceMemory<T>` exposes only device pointers and metadata, and device writes need `&mut` or an unsafe raw API.
unsafe impl<T: Sync> Sync for DeviceMemory<T> {}

// SAFETY: `&mut self` is exclusive over the `length` retained elements at `ptr`.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceMemory<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr
    }
}

// SAFETY: `MaybeUninit<T>` has `T`'s layout, so the retained range holds `length` slots for `T`, and no initialized reads are claimed.
unsafe impl<T: DeviceRepr> DeviceRange<T> for DeviceMemory<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: `&mut self` is exclusive over `length` retained slots with `T`'s layout.
unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceMemory<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        let () = T::LAYOUT;
        self.ptr.cast()
    }
}
