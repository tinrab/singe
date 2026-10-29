//! Safe ownership wrappers for CUDA external memory imports.

use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    mem::{ManuallyDrop, MaybeUninit, align_of, size_of},
    os::fd::RawFd,
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::driver;

use crate::{
    context::Context,
    error::{Error, Result},
    module::{KernelParameters, PushKernelArg},
    try_ffi,
    view::{
        AllocationId, AnyBitPatternDeviceRepr, DeviceInput, DeviceOutput, DeviceOutputView,
        DeviceRegion, DeviceRegionProvider, DeviceRepr, DeviceSlice, DeviceSliceMut, DeviceView,
        DeviceViewMut, NoUninitDeviceRepr,
    },
};

/// External memory handle types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ExternalMemoryHandleType {
    /// Handle is an opaque file descriptor.
    OpaqueFileDescriptor =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD as _,
    /// Handle is an opaque shared NT handle.
    OpaqueWin32 =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 as _,
    /// Handle is an opaque, globally shared handle.
    OpaqueWin32Kmt =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
            as _,
    /// Handle is a `dma_buf` file descriptor.
    DmaBufferFileDescriptor =
        driver::CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD as _,
}

impl_enum_conversion!(driver::CUexternalMemoryHandleType, ExternalMemoryHandleType);

bitflags::bitflags! {
    /// Flags describing the external allocation imported into CUDA.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ExternalMemoryFlags: u32 {
        /// The external resource has a dedicated memory allocation.
        const DEDICATED = driver::CUDA_EXTERNAL_MEMORY_DEDICATED;
    }
}

impl Display for ExternalMemoryFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Owns an imported external-memory object in an explicit CUDA context.
#[derive(Debug)]
pub struct ExternalMemory {
    handle: driver::CUexternalMemory,
    size: usize,
    context: Arc<Context>,
}

/// Retains a native mapping, its imported memory object, and the actual CUDA context.
///
/// Raw addresses and regions grant neither initialized access nor permission to enqueue conflicting work.
#[derive(Debug)]
pub struct MappedAllocation {
    memory: ManuallyDrop<ExternalMemory>,
    ptr: *mut u8,
    bytes: usize,
    offset: usize,
    id: AllocationId,
}

/// Typed storage in one native external-memory mapping.
///
/// Casts and initialized prefixes retain the original mapping and import independently of typed access.
#[derive(Debug)]
pub struct MappedBuffer<T: DeviceRepr> {
    allocation: Arc<MappedAllocation>,
    ptr: *mut T,
    length: usize,
}

impl ExternalMemory {
    /// Imports an opaque file descriptor in `context`.
    ///
    /// # Safety
    /// `fd` must reference a compatible memory object of `size` bytes on this context's device.
    /// CUDA takes descriptor ownership on success. Do not use it afterward.
    /// Keep external backing resources alive and coordinate foreign accesses with all CUDA accesses.
    /// Do not create independently mutable imports of overlapping external storage.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAllocationRequest` for a zero byte size. Also propagates context binding and native import errors, or `Error::NullHandle` for a null imported CUDA handle. Descriptor ownership remains with the caller if the native import fails.
    #[cfg(unix)]
    pub unsafe fn import_opaque_file_descriptor(
        context: &Arc<Context>,
        fd: RawFd,
        size: usize,
        flags: ExternalMemoryFlags,
    ) -> Result<Self> {
        let mut desc = handle_desc(ExternalMemoryHandleType::OpaqueFileDescriptor, size, flags)?;
        desc.handle.fd = fd;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::import(context, &desc, size) }
    }

    /// Imports an opaque Win32 shared handle in `context`.
    ///
    /// # Safety
    /// The handle must reference a compatible live external allocation of the stated size on this device.
    /// Ownership of the Win32 handle remains with the caller.
    /// Retain foreign backing resources, coordinate every foreign access, and prevent independently mutable aliases.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `handle` is null. Returns `Error::InvalidMemoryAllocationRequest` for a zero byte size. Also propagates context binding and native import errors, or `Error::NullHandle` for a null imported CUDA handle.
    pub unsafe fn import_opaque_win32_handle(
        context: &Arc<Context>,
        handle: *mut c_void,
        size: usize,
        flags: ExternalMemoryFlags,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        let mut desc = handle_desc(ExternalMemoryHandleType::OpaqueWin32, size, flags)?;
        desc.handle.win32.handle = handle;
        desc.handle.win32.name = ptr::null();
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::import(context, &desc, size) }
    }

    /// Imports a named Win32 external memory object in `context`.
    ///
    /// # Safety
    /// `name` must reference a valid null-terminated UTF-16 object name throughout import.
    /// The object must be compatible with this device and have the stated size.
    /// Retain foreign backing resources, coordinate every foreign access, and prevent independently mutable aliases.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` if `name` is null. Returns `Error::InvalidMemoryAllocationRequest` for a zero byte size. Also propagates context binding and native import errors, or `Error::NullHandle` for a null imported CUDA handle.
    pub unsafe fn import_opaque_win32_name(
        context: &Arc<Context>,
        name: *const c_void,
        size: usize,
        flags: ExternalMemoryFlags,
    ) -> Result<Self> {
        if name.is_null() {
            return Err(Error::NullHandle);
        }
        let mut desc = handle_desc(ExternalMemoryHandleType::OpaqueWin32, size, flags)?;
        desc.handle.win32.handle = ptr::null_mut();
        desc.handle.win32.name = name;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::import(context, &desc, size) }
    }

    unsafe fn import(
        context: &Arc<Context>,
        desc: &driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
        size: usize,
    ) -> Result<Self> {
        context.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: the caller establishes external-handle validity and ownership.
        unsafe {
            try_ffi!(driver::cuImportExternalMemory(&raw mut handle, desc))?;
        }
        // SAFETY: CUDA returned ownership in the context just bound.
        unsafe { Self::from_raw(context, handle, size) }
    }

    /// Consumes this import and maps uninitialized typed storage.
    ///
    /// One import owns one native mapping. Use borrowed views and splits for subranges.
    /// This avoids overlapping native mappings with distinct virtual aliases and coherence requirements.
    /// Mapping establishes no claim about externally produced values or completion.
    /// Offset and size must also satisfy the originating API's alignment requirements.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed byte count or range end overflows, the range exceeds the imported object, the context cannot be bound, or CUDA rejects the mapping. Also rejects a null nonempty mapping, a mapping address not aligned for `T`, or an invalid mapped address range. The consumed import is released on failure unless a native mapping cannot be safely freed.
    ///
    /// # Panics
    ///
    /// Panics if `T`'s alignment cannot be represented by a 64-bit device address.
    pub fn map_buffer<T: DeviceRepr>(
        self,
        offset_bytes: usize,
        length: usize,
    ) -> Result<MappedBuffer<MaybeUninit<T>>> {
        let () = T::LAYOUT;
        let bytes = checked_bytes::<T>(length)?;
        let end = offset_bytes
            .checked_add(bytes)
            .ok_or(Error::InvalidMemoryAllocationRequest)?;
        if end > self.size {
            return Err(Error::InvalidMemoryAccess);
        }
        self.context.bind()?;
        let mut pointer = 0;
        if bytes != 0 {
            let desc = driver::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
                offset: offset_bytes as _,
                size: bytes as _,
                flags: 0,
                reserved: [0; 16],
            };
            // SAFETY: the import owns this compatible handle and the checked range fits its backing object.
            unsafe {
                try_ffi!(driver::cuExternalMemoryGetMappedBuffer(
                    &raw mut pointer,
                    self.handle,
                    &raw const desc
                ))?;
            }
            if pointer == 0 {
                return Err(Error::NullHandle);
            }
        }
        let allocation = Arc::new(MappedAllocation {
            memory: ManuallyDrop::new(self),
            ptr: pointer as *mut u8,
            bytes,
            offset: offset_bytes,
            id: AllocationId::create(),
        });
        // SAFETY: CUDA returned this live mapping; validate its address range before exposing a provider.
        unsafe {
            DeviceRegion::from_raw_parts(
                allocation.context(),
                allocation.id,
                allocation.ptr,
                bytes,
            )?
        };
        if bytes != 0
            && !pointer.is_multiple_of(
                u64::try_from(align_of::<T>()).expect("alignment fits a device address"),
            )
        {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(MappedBuffer {
            allocation,
            ptr: pointer as *mut MaybeUninit<T>,
            length,
        })
    }

    /// Returns the byte size declared for the imported external-memory object.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.size
    }

    /// Returns the borrowed native import handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(&self) -> driver::CUexternalMemory {
        self.handle
    }

    /// Returns the actual CUDA import context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Takes ownership of an unmapped native external-memory import.
    ///
    /// # Safety
    /// The handle must be live in `context`, have the stated size, and have no other destroying owner or mapped buffers.
    /// Retain external backing resources, coordinate foreign accesses, and prevent independently mutable aliases.
    ///
    /// # Errors
    ///
    /// Returns `Error::NullHandle` for a null handle or `Error::InvalidMemoryAllocationRequest` for a zero byte size. The caller must satisfy the remaining handle validity and ownership requirements.
    pub unsafe fn from_raw(
        context: &Arc<Context>,
        handle: driver::CUexternalMemory,
        size: usize,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        if size == 0 {
            return Err(Error::InvalidMemoryAllocationRequest);
        }
        Ok(Self {
            handle,
            size,
            context: Arc::clone(context),
        })
    }

    /// Transfers the unmapped native handle, byte size, and retained context to its next owner.
    ///
    /// Bind the returned context before use and retain it through native destruction.
    #[must_use]
    pub fn into_raw(self) -> (driver::CUexternalMemory, usize, Arc<Context>) {
        let memory = ManuallyDrop::new(self);
        // SAFETY: ManuallyDrop prevents destruction, and the context owner is transferred exactly once.
        (memory.handle, memory.size, unsafe {
            ptr::read(&raw const memory.context)
        })
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind().and_then(|()| {
            // SAFETY: this wrapper owns the live unmapped import in the bound context.
            unsafe { try_ffi!(driver::cuDestroyExternalMemory(self.handle)) }
        }) {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy cuda external memory: {error}");
        }
    }
}

// SAFETY: imports expose only raw handles and immutable metadata and bind their retained context for native operations.
unsafe impl Send for ExternalMemory {}

unsafe impl Sync for ExternalMemory {}

// The owner exposes raw addresses and metadata; typed borrows or unsafe contracts govern access.
unsafe impl Send for MappedAllocation {}

unsafe impl Sync for MappedAllocation {}

impl MappedAllocation {
    /// Returns the stable allocation identity shared by views and casts of this mapping.
    #[must_use]
    pub const fn id(&self) -> AllocationId {
        self.id
    }

    /// Returns the CUDA context retained by the imported-memory owner.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.memory.context()
    }

    /// Returns the device mapping's raw byte address without granting initialized access or host accessibility.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Returns the byte size of this native mapping.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }

    /// Returns the offset of this native mapping in the original external object.
    #[must_use]
    pub const fn external_offset_bytes(&self) -> usize {
        self.offset
    }

    /// Borrows the imported-memory owner retained until after the native mapping is freed.
    #[must_use]
    pub fn external_memory(&self) -> &ExternalMemory {
        &self.memory
    }
}

// SAFETY: this owner retains the complete native mapping, actual context, and stable allocation identity.
unsafe impl DeviceRegionProvider for MappedAllocation {
    fn region(&self) -> DeviceRegion<'_> {
        // SAFETY: all bytes belong to this live native mapping and its retained import context.
        unsafe { DeviceRegion::from_raw_parts(self.context(), self.id, self.ptr, self.bytes) }
            .expect("native mapping bounds validated during construction")
    }
}

impl Drop for MappedAllocation {
    fn drop(&mut self) {
        if !self.ptr.is_null()
            && let Err(error) = self.context().bind().and_then(|()| {
                // SAFETY: this is the last owner of this mapping, allocated by CUDA in the retained context.
                unsafe { try_ffi!(driver::cuMemFree_v2(self.ptr as driver::CUdeviceptr)) }
            })
        {
            // Preserve the imported memory and its context if the native mapping could still be live.
            #[cfg(debug_assertions)]
            eprintln!("failed to free cuda mapped external memory buffer: {error}");
            return;
        }
        // SAFETY: native mapping destruction completed before destroying its sole imported-memory owner.
        unsafe { ManuallyDrop::drop(&mut self.memory) };
    }
}

impl<T: DeviceRepr> MappedBuffer<T> {
    /// Returns a device pointer to the buffer's typed storage without granting host accessibility.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable device pointer without extending the allocation lifetime or granting access after this borrow.
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the number of typed storage elements exposed by this buffer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this buffer exposes zero typed elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the byte size of the exposed typed range, which can be shorter than the retained native mapping.
    ///
    /// # Panics
    ///
    /// Panics if the stored length violates the byte-size invariant checked during construction and casts.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        checked_bytes::<T>(self.length).expect("mapped byte length validated during construction")
    }

    /// Returns the CUDA context owning the retained external-memory mapping.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows the complete native mapping owner retained by this typed buffer.
    #[must_use]
    pub fn allocation(&self) -> &Arc<MappedAllocation> {
        &self.allocation
    }

    /// Borrows this buffer as a typed device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_buffer(self)
    }

    /// Exclusively borrows this buffer as a mutable typed device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_buffer(self)
    }

    /// Reinterprets initialized bytes while preserving the complete native mapping and imported owner.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the target type has zero size, the byte count is not a multiple of its size, or a nonempty mapping address is not aligned for it. The returned original buffer retains its mapping unchanged.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<MappedBuffer<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        let () = U::LAYOUT;
        let bytes = self.byte_len();
        if size_of::<U>() == 0
            || !bytes.is_multiple_of(size_of::<U>())
            || (bytes != 0 && !(self.ptr as usize).is_multiple_of(align_of::<U>()))
        {
            return Err(self);
        }
        Ok(MappedBuffer {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: bytes / size_of::<U>(),
        })
    }

    /// Preserves the mapping and element count while discarding the initialized-value claim.
    #[must_use]
    pub fn into_uninit(self) -> MappedBuffer<MaybeUninit<T>> {
        MappedBuffer {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }
}

impl<T: DeviceRepr> MappedBuffer<MaybeUninit<T>> {
    /// Exclusively borrows uninitialized storage as an output view without exposing initialized `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Reinterprets uninitialized storage without claiming initialized bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if the target type has zero size, the byte count is not a multiple of its size, or a nonempty mapping address is not aligned for it. The returned original buffer retains its mapping unchanged.
    pub fn cast_uninit<U: DeviceRepr>(self) -> StdResult<MappedBuffer<MaybeUninit<U>>, Self> {
        let () = U::LAYOUT;
        let bytes = self.byte_len();
        if size_of::<U>() == 0
            || !bytes.is_multiple_of(size_of::<U>())
            || (bytes != 0 && !(self.ptr as usize).is_multiple_of(align_of::<U>()))
        {
            return Err(self);
        }
        Ok(MappedBuffer {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: bytes / size_of::<U>(),
        })
    }

    /// Marks every element initialized after external or CUDA production.
    ///
    /// # Safety
    /// All elements must contain valid T values and all producers must complete or be ordered before every consumer.
    /// Synchronize foreign runtime accesses and retain exclusive access until production completes.
    #[must_use]
    pub unsafe fn assume_init(self) -> MappedBuffer<T> {
        let () = T::LAYOUT;
        MappedBuffer {
            allocation: self.allocation,
            ptr: self.ptr.cast(),
            length: self.length,
        }
    }

    /// Exposes only a produced prefix, retaining the complete native mapping for deallocation.
    ///
    /// # Safety
    /// Prefix values must be valid and producer completion and foreign access must satisfy `assume_init`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` if `length` exceeds the buffer's element count. Value validity and producer ordering remain the caller's responsibility.
    pub unsafe fn assume_init_prefix(mut self, length: usize) -> Result<MappedBuffer<T>> {
        if length > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        self.length = length;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Ok(self.assume_init()) }
    }
}

// SAFETY: typed access follows T's thread-safety requirements; native owners bind their actual context on destruction.
unsafe impl<T: DeviceRepr + Send> Send for MappedBuffer<T> {}

unsafe impl<T: DeviceRepr + Sync> Sync for MappedBuffer<T> {}

// SAFETY: typed handles preserve the owner's identity and expose only their validated leading byte range.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for MappedBuffer<T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.allocation
            .region()
            .slice(..self.byte_len())
            .expect("mapped range validated during construction")
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for MappedBuffer<T> {
    fn as_device_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for MappedBuffer<T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for MappedBuffer<T> {}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for MappedBuffer<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for MappedBuffer<MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for MappedBuffer<MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.cast()
    }
}

impl<T: DeviceRepr> PushKernelArg for &MappedBuffer<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_input(self);
    }
}

impl<T: DeviceRepr> PushKernelArg for &mut MappedBuffer<T> {
    fn push_to(self, params: &mut KernelParameters<'_>) {
        params.device_slice_mut(self);
    }
}

fn handle_desc(
    handle_type: ExternalMemoryHandleType,
    size: usize,
    flags: ExternalMemoryFlags,
) -> Result<driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC> {
    if size == 0 {
        return Err(Error::InvalidMemoryAllocationRequest);
    }

    Ok(driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        type_: handle_type.into(),
        handle: driver::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1::default(),
        size: size as _,
        flags: flags.bits(),
        reserved: [0; 16],
    })
}

fn checked_bytes<T>(length: usize) -> Result<usize> {
    if length != 0 && size_of::<T>() == 0 {
        return Err(Error::InvalidMemoryAllocationRequest);
    }
    length
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAllocationRequest)
}

impl_enum_display!(ExternalMemoryHandleType, {
    ExternalMemoryHandleType::OpaqueFileDescriptor => "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD",
    ExternalMemoryHandleType::OpaqueWin32 => "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32",
    ExternalMemoryHandleType::OpaqueWin32Kmt => "CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT",
    ExternalMemoryHandleType::DmaBufferFileDescriptor => "CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD",
});
