//! Borrowed typed views over CUDA-accessible memory.
use std::{
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
    num::Wrapping,
    ops::{Bound, Range, RangeBounds},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    context::Context,
    error::{Error, Result},
    memory::{
        DeviceAllocation, DeviceMemory,
        transfer::{self, copy_device},
    },
    stream::Stream,
    types::{Complex, bf16, f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue5m3, f8ue8m0, f16},
};

/// A Rust type that can be represented as plain CUDA device memory.
///
/// # Safety
///
/// Implementors must have a stable representation without implicit padding, Rust ownership, or host references.
/// Copying valid values must not depend on host-only pointer validity.
/// Memory APIs must evaluate [`Self::LAYOUT`] before transferring values or exposing initialized storage.
/// A generic implementation may reject unsupported layouts through that constant.
pub unsafe trait DeviceRepr: Copy + 'static {
    /// Checks that this concrete type has a transferable layout.
    ///
    /// Derives evaluate field layouts and reject implicit padding at compile time.
    /// Manual implementations must uphold the layout contract whenever this constant evaluates successfully.
    const LAYOUT: () = ();
}

/// A [`DeviceRepr`] whose all-zero byte pattern is a valid value.
///
/// # Safety
///
/// Implementors must accept an all-zero byte pattern as a valid instance.
pub unsafe trait ZeroableDeviceRepr: DeviceRepr {}

/// A device representation whose valid values contain no uninitialized bytes.
///
/// Safe byte reinterpretation requires this guarantee on the source type.
/// `MaybeUninit<T>` does not satisfy it, even when `T` does.
///
/// # Safety
///
/// Every byte of every valid value must be initialized whenever `Self::LAYOUT`
/// evaluates successfully. Consumers must evaluate that layout before reading
/// or reinterpreting the representation's bytes.
pub unsafe trait NoUninitDeviceRepr: DeviceRepr {}

/// A [`DeviceRepr`] for which every initialized byte pattern is a valid value.
///
/// This permits safe readback from device storage without a kernel-specific validity check.
/// It does not permit reading uninitialized memory.
///
/// # Safety
///
/// Every initialized bit pattern of the representation must be valid for `Self`.
/// Types with restricted values, including `bool`, do not satisfy this contract.
pub unsafe trait AnyBitPatternDeviceRepr: ZeroableDeviceRepr {}

/// Contiguous device storage that a producer may initialize as values of `T`.
///
/// Both initialized and uninitialized buffers can provide output storage.
/// This trait does not grant permission to read existing elements or establish
/// that a producer initializes the entire range.
///
/// # Safety
///
/// The pointer must identify writable storage for `len()` elements with the
/// device ABI and alignment of `T`, owned by `context()` and live for the
/// borrow. Mutable access must be exclusive. The byte length must fit `usize`.
/// `region()` must describe that same pointer and the exact byte range of those elements.
pub unsafe trait DeviceOutput<T: DeviceRepr>: DeviceInput<T> {
    /// Returns a pointer to the writable device output range.
    ///
    /// Writing valid `T` values requires the caller to establish CUDA execution ordering.
    fn as_mut_ptr(&mut self) -> *mut T;
}

#[cfg(feature = "macros")]
pub use singe_cuda_macros::{
    AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr,
};

macro_rules! impl_device_repr {
    ($($ty:ty),* $(,)?) => {
        $(
            unsafe impl DeviceRepr for $ty {}
            unsafe impl NoUninitDeviceRepr for $ty {}
            unsafe impl ZeroableDeviceRepr for $ty {}
            unsafe impl AnyBitPatternDeviceRepr for $ty {}
        )*
    };
}

impl_device_repr!(
    (),
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    f16,
    bf16,
    f8e4m3,
    f8e5m2,
    f8ue8m0,
    f8ue5m3,
    f6e2m3,
    f6e3m2,
    f4e2m1,
);

// Boolean values can be transferred and zero-initialized, but device bytes other than 0 and 1 are invalid.
unsafe impl DeviceRepr for bool {}

unsafe impl NoUninitDeviceRepr for bool {}

unsafe impl ZeroableDeviceRepr for bool {}

unsafe impl<T: DeviceRepr, const N: usize> DeviceRepr for [T; N] {
    const LAYOUT: () = T::LAYOUT;
}

unsafe impl<T: ZeroableDeviceRepr, const N: usize> ZeroableDeviceRepr for [T; N] {}

unsafe impl<T: NoUninitDeviceRepr, const N: usize> NoUninitDeviceRepr for [T; N] {}

unsafe impl<T: AnyBitPatternDeviceRepr, const N: usize> AnyBitPatternDeviceRepr for [T; N] {}

unsafe impl<T: DeviceRepr> DeviceRepr for Wrapping<T> {
    const LAYOUT: () = T::LAYOUT;
}

unsafe impl<T: ZeroableDeviceRepr> ZeroableDeviceRepr for Wrapping<T> {}

unsafe impl<T: NoUninitDeviceRepr> NoUninitDeviceRepr for Wrapping<T> {}

unsafe impl<T: AnyBitPatternDeviceRepr> AnyBitPatternDeviceRepr for Wrapping<T> {}

// Complex is repr(C) with two consecutive fields of the same element type.
unsafe impl<T: DeviceRepr> DeviceRepr for Complex<T> {
    const LAYOUT: () = T::LAYOUT;
}

unsafe impl<T: ZeroableDeviceRepr> ZeroableDeviceRepr for Complex<T> {}

unsafe impl<T: NoUninitDeviceRepr> NoUninitDeviceRepr for Complex<T> {}

unsafe impl<T: AnyBitPatternDeviceRepr> AnyBitPatternDeviceRepr for Complex<T> {}

// MaybeUninit accepts arbitrary storage, while DeviceRepr on T rules out padding and host ownership.
unsafe impl<T: DeviceRepr> DeviceRepr for MaybeUninit<T> {
    const LAYOUT: () = T::LAYOUT;
}

unsafe impl<T: DeviceRepr> ZeroableDeviceRepr for MaybeUninit<T> {}

unsafe impl<T: DeviceRepr> AnyBitPatternDeviceRepr for MaybeUninit<T> {}

// PhantomData has no stored bytes and owns no resource regardless of its marker type.
unsafe impl<T: ?Sized + 'static> DeviceRepr for PhantomData<T> {}

unsafe impl<T: ?Sized + 'static> ZeroableDeviceRepr for PhantomData<T> {}

unsafe impl<T: ?Sized + 'static> NoUninitDeviceRepr for PhantomData<T> {}

unsafe impl<T: ?Sized + 'static> AnyBitPatternDeviceRepr for PhantomData<T> {}

/// Reports byte-range overlap for allocation hazard tracking.
#[must_use]
pub(crate) fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

/// Identifies one allocation across its typed handles, mappings, and subranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(u64);

impl AllocationId {
    /// Creates an identity for a new allocation or an externally owned resource.
    /// Reuse this identity for every alias of the same live allocation.
    ///
    /// # Panics
    ///
    /// Panics if the process exhausts the nonzero `u64` allocation identity space.
    pub fn create() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(
            NEXT_ID
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("allocation identity exhausted"),
        )
    }
}

/// A byte range in pinned or registered host storage, independent of device mappings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostRegion<'a> {
    context: &'a Arc<Context>,
    allocation_id: AllocationId,
    allocation_bytes: usize,
    offset: usize,
    bytes: usize,
    portable: bool,
}

impl<'a> HostRegion<'a> {
    /// Describes a complete live host allocation or registration.
    ///
    /// # Safety
    /// The range must remain registered for `'a` in `context`.
    /// All host and device aliases must share `allocation_id` and byte offsets.
    /// `portable` must reflect CUDA's registration or allocation flags.
    #[must_use]
    pub unsafe fn from_raw_parts(
        context: &'a Arc<Context>,
        allocation_id: AllocationId,
        bytes: usize,
        portable: bool,
    ) -> Self {
        Self {
            context,
            allocation_id,
            allocation_bytes: bytes,
            offset: 0,
            bytes,
            portable,
        }
    }

    /// Returns the CUDA context associated with this allocation.
    #[must_use]
    pub fn context(self) -> &'a Arc<Context> {
        self.context
    }

    /// Returns the identity shared by all aliases of the backing allocation.
    #[must_use]
    pub const fn allocation_id(self) -> AllocationId {
        self.allocation_id
    }

    /// Returns the byte length of the complete allocation before slicing.
    #[must_use]
    pub const fn allocation_byte_len(self) -> usize {
        self.allocation_bytes
    }

    /// Returns the byte length of this region.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        self.bytes
    }

    /// Returns whether CUDA recognizes this host allocation or registration in other contexts.
    #[must_use]
    pub const fn is_portable(self) -> bool {
        self.portable
    }

    /// Returns this region’s byte offsets relative to the start of the backing allocation.
    #[must_use]
    pub fn byte_range(self) -> Range<usize> {
        self.offset..self.offset + self.bytes
    }

    /// Returns metadata for a byte subrange while preserving the host registration identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.bytes)?;
        Ok(Self {
            offset: self.offset + start,
            bytes: end - start,
            ..self
        })
    }
}

/// A borrowed byte range within one context-associated allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceRegion<'a> {
    context: &'a Arc<Context>,
    stream: Option<&'a Stream>,
    allocation_id: AllocationId,
    base: *mut u8,
    allocation_bytes: usize,
    offset: usize,
    bytes: usize,
}

impl<'a> DeviceRegion<'a> {
    /// Describes an externally established allocation and its full byte range.
    ///
    /// # Safety
    ///
    /// `base` must identify `bytes` live CUDA-accessible bytes in `context`
    /// for `'a`. `allocation_id` must be shared by all aliases of this allocation
    /// and must not identify another live allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if a nonempty range has a null base pointer or its ending address overflows `usize`.
    pub unsafe fn from_raw_parts(
        context: &'a Arc<Context>,
        allocation_id: AllocationId,
        base: *mut u8,
        bytes: usize,
    ) -> Result<Self> {
        if bytes != 0 && base.is_null() {
            return Err(Error::InvalidMemoryAccess);
        }
        (base as usize)
            .checked_add(bytes)
            .ok_or(Error::InvalidMemoryAccess)?;
        Ok(Self {
            context,
            stream: None,
            allocation_id,
            base,
            allocation_bytes: bytes,
            offset: 0,
            bytes,
        })
    }

    /// Returns the CUDA context associated with this allocation.
    #[must_use]
    pub fn context(self) -> &'a Arc<Context> {
        self.context
    }

    /// The stream that orders allocation and writes for stream-ordered storage.
    #[must_use]
    pub const fn stream(self) -> Option<&'a Stream> {
        self.stream
    }

    /// Associates this region with the stream that establishes its readiness.
    ///
    /// # Safety
    ///
    /// Allocation and pending producer writes must be ordered on this stream.
    /// Work on other streams must establish dependencies before accessing the region.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to a different context.
    pub unsafe fn with_stream(self, stream: &'a Stream) -> Result<Self> {
        if stream.context() != self.context {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            stream: Some(stream),
            ..self
        })
    }

    /// Orders this range's allocation and producer stream before a consumer stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the consumer stream belongs to a different context or joining the readiness stream fails.
    pub fn order_on(self, stream: &Stream) -> Result<()> {
        if stream.context() != self.context {
            return Err(Error::StreamContextMismatch);
        }
        if let Some(ready) = self.stream {
            stream.join(ready)?;
        }
        Ok(())
    }

    /// Waits for an associated allocation stream before synchronous access.
    ///
    /// # Errors
    ///
    /// Propagates access-tracking and synchronization errors while waiting for pending accesses to this region.
    pub fn synchronize(self) -> Result<()> {
        let _guard = self.context.acquire_host_access(&[self])?;
        Ok(())
    }

    /// Returns the identity shared by all aliases of the backing allocation.
    #[must_use]
    pub const fn allocation_id(self) -> AllocationId {
        self.allocation_id
    }

    /// Returns the byte length of the complete allocation before slicing.
    #[must_use]
    pub const fn allocation_byte_len(self) -> usize {
        self.allocation_bytes
    }

    /// Returns the byte length of this region.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        self.bytes
    }

    /// Returns this region’s byte offsets relative to the start of the backing allocation.
    #[must_use]
    pub fn byte_range(self) -> Range<usize> {
        self.offset..self.offset + self.bytes
    }

    /// Reports conflicts between byte ranges sharing one allocation identity.
    /// Empty subranges inside another range conservatively count as conflicts.
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        self.allocation_id == other.allocation_id
            && ranges_overlap(&self.byte_range(), &other.byte_range())
    }

    /// Returns the CUDA-accessible pointer at this region’s starting byte offset.
    ///
    /// The pointer does not grant host access or establish initialization or exclusive access.
    #[must_use]
    pub fn as_ptr(self) -> *mut u8 {
        self.base.wrapping_add(self.offset)
    }

    /// Borrows a byte subrange while preserving allocation and context identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.bytes)?;
        Ok(Self {
            offset: self.offset + start,
            bytes: end - start,
            ..self
        })
    }
}

/// Provides the allocation identity and byte range used by CUDA operations.
///
/// # Safety
///
/// Reported metadata must describe the same live memory as the provider's
/// pointers. All aliases must preserve allocation identity and byte offsets.
/// The context must stay retained while the provider is alive.
/// Returned regions must retain their identity, bounds, and liveness for the shared borrow.
pub unsafe trait DeviceRegionProvider {
    /// Borrows the exact allocation range described by this provider’s pointers.
    fn region(&self) -> DeviceRegion<'_>;
}

/// Retains an actual CUDA allocation and a checked byte range independently of typed buffers.
///
/// This is ownership and metadata only.
/// It does not establish initialization, exclusive access, or permission to enqueue conflicting work.
#[derive(Debug)]
pub struct OwnedDeviceRegion<O: DeviceRegionProvider + 'static = DeviceAllocation> {
    owner: Arc<O>,
    range: Range<usize>,
    readiness: Option<Stream>,
}

impl<O: DeviceRegionProvider + 'static> OwnedDeviceRegion<O> {
    /// Retains a checked byte range relative to the owner's exposed region.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range is reversed or exceeds the owner’s exposed region.
    pub fn from_owner(owner: Arc<O>, range: Range<usize>) -> Result<Self> {
        owner.region().slice(range.clone())?;
        Ok(Self {
            owner,
            range,
            readiness: None,
        })
    }

    #[must_use]
    pub(crate) fn into_owner(self) -> Arc<O> {
        self.owner
    }

    /// Returns the retained allocation provider.
    #[must_use]
    pub fn owner(&self) -> &Arc<O> {
        &self.owner
    }

    /// Returns the context associated with the retained allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.owner.region().context()
    }

    /// Returns the byte length of the retained subrange.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.range.len()
    }

    /// Narrows this retained range while preserving its backing owner and readiness stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(mut self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.range.len())?;
        self.range = self.range.start + start..self.range.start + end;
        Ok(self)
    }

    /// Associates pending writes with a retained stream without transferring allocation ownership.
    ///
    /// # Safety
    ///
    /// Allocation and producer writes must be ordered before consumers on this stream.
    /// Other streams must establish dependencies before accessing this range.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to a different context.
    pub unsafe fn with_stream(mut self, stream: &Stream) -> Result<Self> {
        if self.context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }
        self.readiness = Some(stream.clone());
        Ok(self)
    }
}

impl<O: DeviceRegionProvider + 'static> Clone for OwnedDeviceRegion<O> {
    /// Clones allocation retention and metadata without granting another access lease.
    fn clone(&self) -> Self {
        Self {
            owner: Arc::clone(&self.owner),
            range: self.range.clone(),
            readiness: self.readiness.clone(),
        }
    }
}

// SAFETY: the retained provider keeps its region live, and slicing preserves its identity and bounds.
unsafe impl<O: DeviceRegionProvider + 'static> DeviceRegionProvider for OwnedDeviceRegion<O> {
    fn region(&self) -> DeviceRegion<'_> {
        let region = self
            .owner
            .region()
            .slice(self.range.clone())
            .expect("retained range exceeds its allocation");
        self.readiness.as_ref().map_or(region, |stream| {
            unsafe { region.with_stream(stream) }
                .expect("retained readiness stream context mismatch")
        })
    }
}

impl OwnedDeviceRegion {
    /// Retains this buffer's exact logical range, including its original allocation offset.
    #[must_use]
    pub fn from_memory<T: DeviceRepr>(memory: &DeviceMemory<T>) -> Self {
        Self {
            owner: Arc::clone(memory.allocation()),
            range: memory.region().byte_range(),
            readiness: None,
        }
    }
}

pub(crate) fn cast_length<T: DeviceRepr>(region: DeviceRegion<'_>) -> Result<usize> {
    let () = T::LAYOUT;
    let size = size_of::<T>();
    if size == 0
        || !region.byte_len().is_multiple_of(size)
        || (region.byte_len() != 0 && !region.as_ptr().addr().is_multiple_of(align_of::<T>()))
    {
        return Err(Error::InvalidMemoryAccess);
    }
    Ok(region.byte_len() / size)
}

/// Typed input metadata for a contiguous CUDA allocation range.
///
/// The range may be initialized by an earlier operation in a captured graph.
/// This trait does not establish initialized values or grant permission to read
/// them. Raw consumers must establish validity and ordering when they execute.
///
/// # Safety
///
/// The pointer must identify live, aligned CUDA-accessible storage for `len()`
/// elements of `T` throughout the borrow. The byte length must fit `usize`.
/// `region()` must describe the exact byte range, actual context, allocation
/// identity, and offset. Nonempty ranges must have the same pointer.
pub unsafe trait DeviceInput<T: DeviceRepr>: DeviceRegionProvider {
    /// Returns the context associated with the input allocation.
    fn context(&self) -> &Arc<Context> {
        self.region().context()
    }

    /// Returns the device input pointer without asserting that its values are initialized.
    ///
    /// An empty range returns an aligned dangling pointer.
    fn as_device_ptr(&self) -> *const T {
        if self.is_empty() {
            NonNull::<T>::dangling().as_ptr()
        } else {
            self.region().as_ptr().cast()
        }
    }

    /// Returns the number of input elements.
    fn len(&self) -> usize;

    /// Returns whether the input range contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the input range length in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the element count multiplied by `size_of::<T>()` overflows `usize`.
    fn byte_len(&self) -> Result<usize> {
        let () = T::LAYOUT;
        self.len()
            .checked_mul(size_of::<T>())
            .ok_or(Error::InvalidMemoryAllocationRequest)
    }
}

/// A typed contiguous range containing initialized device values.
///
/// # Safety
///
/// Every element must contain a valid `T` for the borrow. Safe consumers may
/// read the range using CUDA, subject to stream ordering and shared-borrow rules.
pub unsafe trait DeviceSlice<T: DeviceRepr>: DeviceInput<T> {}

/// A mutable typed contiguous range of CUDA-accessible device memory.
///
/// # Safety
///
/// The mutable pointer must identify the same range as `DeviceSlice` and grant
/// exclusive access for its mutable borrow, including work using that borrow.
/// Its initialized input and writable output capabilities must describe the same range.
pub unsafe trait DeviceSliceMut<T: DeviceRepr>: DeviceSlice<T> + DeviceOutput<T> {}

/// A typed contiguous host-memory range that can be copied to CUDA memory.
///
/// # Safety
///
/// The pointer must identify `len()` live, aligned `T` values readable for the
/// borrow. The byte length must fit `usize`; shared access must not mutate them.
pub unsafe trait HostSlice<T: DeviceRepr> {
    /// Returns a pointer to the readable host elements.
    fn as_host_ptr(&self) -> *const T;

    /// Returns the number of host elements.
    fn len(&self) -> usize;

    /// Returns whether the host range contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A mutable typed contiguous host-memory range that can be copied from CUDA memory.
///
/// # Safety
///
/// The mutable pointer must identify the same range as `HostSlice`, writable
/// exclusively for the mutable borrow.
pub unsafe trait HostSliceMut<T: DeviceRepr>: HostSlice<T> {
    /// Returns a pointer to the exclusively writable host elements.
    fn as_host_mut_ptr(&mut self) -> *mut T;
}

/// Shared abstraction for read-only byte buffers.
///
/// # Safety
///
/// The pointer must identify `byte_len()` live readable bytes for the borrow.
/// Consumers must use the address space appropriate to the concrete buffer.
pub unsafe trait ByteBuffer {
    /// Returns a pointer to the readable bytes in the buffer’s address space.
    fn as_byte_ptr(&self) -> *const u8;

    /// Returns the number of readable bytes.
    fn byte_len(&self) -> usize;

    /// Returns whether the byte range is empty.
    fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }
}

/// Shared abstraction for mutable byte buffers.
///
/// # Safety
///
/// The mutable pointer must identify the same bytes as `ByteBuffer` and grant
/// exclusive writable access for the mutable borrow.
pub unsafe trait ByteBufferMut: ByteBuffer {
    /// Returns a pointer to the exclusively writable bytes in the buffer’s address space.
    fn as_byte_mut_ptr(&mut self) -> *mut u8;
}

unsafe impl<B> ByteBuffer for B
where
    B: DeviceSlice<u8> + ?Sized,
{
    fn as_byte_ptr(&self) -> *const u8 {
        self.as_device_ptr()
    }

    fn byte_len(&self) -> usize {
        self.len()
    }
}

unsafe impl<B> ByteBufferMut for B
where
    B: DeviceSliceMut<u8> + ?Sized,
{
    fn as_byte_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
}

/// Typed CUDA input metadata without an initialization guarantee.
///
/// Construction checks the element layout and preserves the allocation's
/// context and range. Reading values requires an unsafe consumer to establish
/// initialization and order preceding producers before each read.
#[derive(Debug, Clone, Copy)]
pub struct DeviceInputView<'a, T: DeviceRepr> {
    region: DeviceRegion<'a>,
    length: usize,
    _t: PhantomData<&'a MaybeUninit<T>>,
}

impl<'a, T: DeviceRepr> DeviceInputView<'a, T> {
    /// Describes a live allocation range as input elements of `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` is zero-sized, the region’s byte length is not divisible by `size_of::<T>()`, or a nonempty region is not aligned for `T`.
    pub fn create(region: DeviceRegion<'a>) -> Result<Self> {
        let length = cast_length::<T>(region)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Borrows typed input metadata without strengthening its validity contract.
    pub fn from_buffer<B: DeviceInput<T> + ?Sized>(buffer: &'a B) -> Self {
        let () = T::LAYOUT;
        Self {
            region: buffer.region(),
            length: buffer.len(),
            _t: PhantomData,
        }
    }

    /// Returns the CUDA-accessible element pointer.
    ///
    /// An empty view returns an aligned dangling pointer. This pointer does not grant host access.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        if self.length == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            self.region.as_ptr().cast()
        }
    }

    /// Returns the number of elements in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the view contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the context associated with the viewed allocation.
    #[must_use]
    pub fn context(&self) -> &'a Arc<Context> {
        self.region.context()
    }

    /// Returns the view’s exact byte range and allocation identity.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'a> {
        self.region
    }

    /// Narrows input metadata while preserving its allocation identity and offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the element range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.length)?;
        Ok(Self {
            region: self
                .region
                .slice(start * size_of::<T>()..end * size_of::<T>())?,
            length: end - start,
            _t: PhantomData,
        })
    }

    /// Reinterprets input metadata without asserting initialized destination values.
    ///
    /// # Errors
    ///
    /// Returns an error if `U` is zero-sized, the region’s byte length is not divisible by `size_of::<U>()`, or a nonempty region is not aligned for `U`.
    pub fn cast<U: DeviceRepr>(self) -> Result<DeviceInputView<'a, U>> {
        DeviceInputView::create(self.region)
    }
}

// SAFETY: both constructors preserve the provider's exact live allocation range.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for DeviceInputView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

// SAFETY: construction checks the layout; slicing and casting preserve bounds and alignment.
unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceInputView<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// An immutable borrow of typed CUDA storage, including its allocation identity.
#[derive(Debug, Clone, Copy)]
pub struct DeviceView<'a, T: DeviceRepr> {
    region: DeviceRegion<'a>,
    length: usize,
    _t: PhantomData<&'a T>,
}

/// An exclusive borrow of typed CUDA storage.
#[derive(Debug)]
pub struct DeviceViewMut<'a, T: DeviceRepr> {
    region: DeviceRegion<'a>,
    length: usize,
    _t: PhantomData<&'a mut T>,
}

/// An exclusive write-only range for a producer of `T` values.
///
/// The range may be uninitialized and cannot be used as a readable device slice.
/// Its element type describes produced values rather than the storage's current initialization state.
#[derive(Debug)]
pub struct DeviceOutputView<'a, T: DeviceRepr> {
    region: DeviceRegion<'a>,
    length: usize,
    _t: PhantomData<&'a mut MaybeUninit<T>>,
}

impl<'a, T: DeviceRepr> DeviceOutputView<'a, T> {
    /// Consumes the typed borrow into allocation metadata without granting access.
    #[must_use]
    pub fn into_region(self) -> DeviceRegion<'a> {
        self.region
    }

    /// Borrows an externally established writable region without claiming initialization.
    ///
    /// # Safety
    ///
    /// The region must remain live and exclusively writable for the returned
    /// lifetime, including queued CUDA accesses using this output.
    ///
    /// # Errors
    ///
    /// Returns an error if `length * size_of::<T>()` overflows or differs from the region’s byte length, or a nonempty region has a zero-sized element type or insufficient alignment.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        validate_view::<T>(region, length)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Borrows writable storage without exposing its existing contents as `T` values.
    pub fn from_output<B: DeviceOutput<T> + ?Sized>(output: &'a mut B) -> Self {
        let () = T::LAYOUT;
        Self {
            region: output.region(),
            length: output.len(),
            _t: PhantomData,
        }
    }

    /// Borrows this destination again for a shorter exclusive lifetime.
    pub fn reborrow(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Returns the number of output elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the output range is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the pointer a producer may write through.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }

    /// Returns the output allocation's context.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Narrows this exclusive output borrow to an element range.
    ///
    /// # Errors
    ///
    /// Returns an error if the element range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.length)?;
        Ok(Self {
            region: self
                .region
                .slice(start * size_of::<T>()..end * size_of::<T>())?,
            length: end - start,
            _t: PhantomData,
        })
    }

    /// Splits this exclusive output borrow into disjoint writable ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if `mid` exceeds the view’s element count.
    pub fn split_at_mut(self, mid: usize) -> Result<(Self, Self)> {
        if mid > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        let boundary = mid * size_of::<T>();
        let left = Self {
            region: self.region.slice(..boundary)?,
            length: mid,
            _t: PhantomData,
        };
        let right = Self {
            region: self.region.slice(boundary..)?,
            length: self.length - mid,
            _t: PhantomData,
        };
        Ok((left, right))
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for DeviceOutputView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceOutputView<'_, T> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceOutputView<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.region.as_ptr().cast()
    }
}

fn validate_view<T: DeviceRepr>(region: DeviceRegion<'_>, length: usize) -> Result<()> {
    let () = T::LAYOUT;
    if length.checked_mul(size_of::<T>()) != Some(region.byte_len())
        || (length != 0
            && (size_of::<T>() == 0 || !region.as_ptr().addr().is_multiple_of(align_of::<T>())))
    {
        return Err(Error::InvalidMemoryAccess);
    }
    Ok(())
}

impl<'a, T: DeviceRepr> DeviceView<'a, T> {
    /// Consumes the typed borrow into allocation metadata without granting new access.
    #[must_use]
    pub fn into_region(self) -> DeviceRegion<'a> {
        self.region
    }

    /// Borrows a region containing `length` values of `T`.
    ///
    /// # Safety
    ///
    /// The region must contain valid `T` values for the returned lifetime.
    /// Access through other aliases must respect this shared borrow and CUDA ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if `length * size_of::<T>()` overflows or differs from the region’s byte length, or a nonempty region has a zero-sized element type or insufficient alignment.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        validate_view::<T>(region, length)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Borrows every initialized element of a device allocation without copying data.
    #[must_use]
    pub fn from_memory(memory: &'a DeviceMemory<T>) -> Self {
        let () = T::LAYOUT;
        let () = T::LAYOUT;
        Self {
            region: memory.region(),
            length: memory.len(),
            _t: PhantomData,
        }
    }

    /// Borrows any typed provider that also retains allocation metadata.
    pub fn from_buffer<B: DeviceSlice<T> + ?Sized>(buffer: &'a B) -> Self {
        let () = T::LAYOUT;
        let () = T::LAYOUT;
        Self {
            region: buffer.region(),
            length: buffer.len(),
            _t: PhantomData,
        }
    }

    /// Returns the CUDA-accessible element pointer.
    ///
    /// An empty view returns an aligned dangling pointer. This pointer does not grant host access.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        if self.length == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            self.region.as_ptr().cast()
        }
    }

    /// Returns the number of elements in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the view contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the context associated with the viewed allocation.
    #[must_use]
    pub fn context(&self) -> &'a Arc<Context> {
        self.region.context()
    }

    /// Returns the view’s exact byte range and allocation identity.
    #[must_use]
    pub const fn region(&self) -> DeviceRegion<'a> {
        self.region
    }

    /// Copies every viewed element into a host slice and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow. Also propagates access-tracking, readiness synchronization, context binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host(&self, destination: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        transfer::copy_to_host(self, destination)
    }

    /// Copies every viewed element into a new host vector and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates byte-size overflow, access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        transfer::copy_to_host_vec(self)
    }

    /// Returns a view of an element subrange, preserving allocation identity and readiness.
    ///
    /// # Errors
    ///
    /// Returns an error if the element range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(self, range: R) -> Result<Self> {
        let (start, end) = bounds_to_range(range, self.length)?;
        let region = self
            .region
            .slice(start * size_of::<T>()..end * size_of::<T>())?;
        Ok(Self {
            region,
            length: end - start,
            _t: PhantomData,
        })
    }

    /// Reinterprets initialized bytes as values accepting every bit pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if `U` is zero-sized, the region’s byte length is not divisible by `size_of::<U>()`, or a nonempty region is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> Result<DeviceView<'a, U>>
    where
        T: NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: the source contains initialized bytes and every destination pattern is valid.
        unsafe { self.cast_unchecked() }
    }

    /// Reinterprets a region with a caller-established initialization contract.
    ///
    /// # Safety
    ///
    /// Every destination element must be initialized and valid for `U`.
    /// Alignment and byte length are checked by this method.
    ///
    /// # Errors
    ///
    /// Returns an error if `U` is zero-sized, the region’s byte length is not divisible by `size_of::<U>()`, or a nonempty region is not aligned for `U`.
    pub unsafe fn cast_unchecked<U: DeviceRepr>(self) -> Result<DeviceView<'a, U>> {
        let length = cast_length::<U>(self.region)?;
        Ok(DeviceView {
            region: self.region,
            length,
            _t: PhantomData,
        })
    }
}

impl<'a, T: DeviceRepr> DeviceViewMut<'a, T> {
    /// Consumes the typed borrow into allocation metadata without granting access.
    #[must_use]
    pub fn into_region(self) -> DeviceRegion<'a> {
        self.region
    }

    /// Exclusively borrows a region containing `length` values of `T`.
    ///
    /// # Safety
    ///
    /// The region must contain valid `T` values and remain exclusively borrowed
    /// for the returned lifetime, including all GPU work using this view.
    ///
    /// # Errors
    ///
    /// Returns an error if `length * size_of::<T>()` overflows or differs from the region’s byte length, or a nonempty region has a zero-sized element type or insufficient alignment.
    pub unsafe fn from_raw_parts(region: DeviceRegion<'a>, length: usize) -> Result<Self> {
        validate_view::<T>(region, length)?;
        Ok(Self {
            region,
            length,
            _t: PhantomData,
        })
    }

    /// Exclusively borrows every initialized element of a device allocation without copying data.
    pub fn from_memory(memory: &'a mut DeviceMemory<T>) -> Self {
        let () = T::LAYOUT;
        let () = T::LAYOUT;
        Self {
            region: memory.region(),
            length: memory.len(),
            _t: PhantomData,
        }
    }

    /// Exclusively borrows an initialized typed provider without copying data.
    pub fn from_buffer<B: DeviceSliceMut<T> + ?Sized>(buffer: &'a mut B) -> Self {
        let () = T::LAYOUT;
        let () = T::LAYOUT;
        Self {
            region: buffer.region(),
            length: buffer.len(),
            _t: PhantomData,
        }
    }

    /// Borrows this mutable view again for a shorter exclusive lifetime.
    pub fn reborrow(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut {
            region: self.region(),
            length: self.length,
            _t: PhantomData,
        }
    }

    /// Returns the CUDA-accessible element pointer.
    ///
    /// An empty view returns an aligned dangling pointer. This pointer does not grant host access.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        if self.length == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            self.region.as_ptr().cast()
        }
    }

    /// Returns the exclusively borrowed CUDA-accessible element pointer.
    ///
    /// This pointer does not grant host access or synchronize pending work.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_ptr().cast_mut()
    }

    /// Returns the number of elements in the view.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether the view contains no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the context associated with the viewed allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.region.context()
    }

    /// Returns the view’s exact byte range and allocation identity.
    #[must_use]
    pub fn region(&self) -> DeviceRegion<'_> {
        self.region
    }

    /// Copies a host slice into the entire view and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow. Also propagates access-tracking, readiness synchronization, context binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_from_host(&mut self, source: &[T]) -> Result<()> {
        transfer::copy_from_host(self, source)
    }

    /// Copies a disjoint device range into the entire view and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the ranges have different contexts or element counts, overlap in the same allocation, or their byte size overflows. Also propagates access-tracking, readiness synchronization, context binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_from_device<B: DeviceSlice<T> + ?Sized>(&mut self, source: &B) -> Result<()> {
        copy_device(self, source)
    }

    /// Copies every viewed element into a host slice and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow. Also propagates access-tracking, readiness synchronization, context binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host(&self, destination: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        self.as_view().copy_to_host(destination)
    }

    /// Copies every viewed element into a new host vector and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates byte-size overflow, access-tracking, readiness synchronization, context-binding, stream-capture, CUDA copy, and stream synchronization errors. A nonempty synchronous transfer is rejected during stream capture.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        self.as_view().copy_to_host_vec()
    }

    /// Borrows this view for reading without extending the lifetime of the mutable borrow.
    #[must_use]
    pub fn as_view(&self) -> DeviceView<'_, T> {
        DeviceView {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }

    /// Borrows an initialized element subrange for reading.
    ///
    /// # Errors
    ///
    /// Returns an error if the element range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Result<DeviceView<'_, T>> {
        self.as_view().slice(range)
    }

    /// Exclusively borrows an initialized element subrange.
    ///
    /// # Errors
    ///
    /// Returns an error if the element range is reversed, exceeds this range’s bounds, or an inclusive bound cannot be converted without overflow.
    pub fn slice_mut<R: RangeBounds<usize>>(&mut self, range: R) -> Result<DeviceViewMut<'_, T>> {
        let (start, end) = bounds_to_range(range, self.length)?;
        let region = self
            .region
            .slice(start * size_of::<T>()..end * size_of::<T>())?;
        Ok(DeviceViewMut {
            region,
            length: end - start,
            _t: PhantomData,
        })
    }

    /// Borrows two disjoint mutable views, split at element index `mid`.
    ///
    /// # Errors
    ///
    /// Returns an error if `mid` exceeds the view’s element count.
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> Result<(DeviceViewMut<'_, T>, DeviceViewMut<'_, T>)> {
        if mid > self.length {
            return Err(Error::InvalidMemoryAccess);
        }
        let bytes = mid * size_of::<T>();
        Ok((
            DeviceViewMut {
                region: self.region.slice(..bytes)?,
                length: mid,
                _t: PhantomData,
            },
            DeviceViewMut {
                region: self.region.slice(bytes..)?,
                length: self.length - mid,
                _t: PhantomData,
            },
        ))
    }

    /// Reinterprets a mutable borrow while preserving both element types' validity.
    ///
    /// # Errors
    ///
    /// Returns an error if `U` is zero-sized, the region’s byte length is not divisible by `size_of::<U>()`, or a nonempty region is not aligned for `U`.
    pub fn cast<U: AnyBitPatternDeviceRepr + NoUninitDeviceRepr>(
        self,
    ) -> Result<DeviceViewMut<'a, U>>
    where
        T: AnyBitPatternDeviceRepr + NoUninitDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: both representations accept all initialized bits and U cannot introduce uninitialized bytes.
        unsafe { self.cast_unchecked() }
    }

    /// Reinterprets a mutable borrow under a caller-established validity contract.
    ///
    /// # Safety
    ///
    /// Every destination element must be initialized and valid for `U`.
    /// Writes through the returned view must leave valid `T` values when the borrow ends.
    ///
    /// # Errors
    ///
    /// Returns an error if `U` is zero-sized, the region’s byte length is not divisible by `size_of::<U>()`, or a nonempty region is not aligned for `U`.
    pub unsafe fn cast_unchecked<U: DeviceRepr>(self) -> Result<DeviceViewMut<'a, U>> {
        let length = cast_length::<U>(self.region)?;
        Ok(DeviceViewMut {
            region: self.region,
            length,
            _t: PhantomData,
        })
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for DeviceView<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for DeviceViewMut<'_, T> {
    fn region(&self) -> DeviceRegion<'_> {
        self.region
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceViewMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

// SAFETY: output metadata describes the same live typed range without initialized reads.
unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceViewMut<'_, MaybeUninit<T>> {
    fn len(&self) -> usize {
        self.length
    }
}

unsafe impl<T: DeviceRepr> DeviceOutput<T> for DeviceViewMut<'_, MaybeUninit<T>> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr().cast()
    }
}

impl<'a, T: DeviceRepr> DeviceViewMut<'a, MaybeUninit<T>> {
    /// Borrows this uninitialized range as a write-only output of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Exposes elements initialized by a producer through this exclusive view.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid T value. Producing writes must be
    /// ordered before all consumers, and must complete before host readback.
    #[must_use]
    pub unsafe fn assume_init(self) -> DeviceViewMut<'a, T> {
        let () = T::LAYOUT;
        DeviceViewMut {
            region: self.region,
            length: self.length,
            _t: PhantomData,
        }
    }
}

impl<T: DeviceRepr> DeviceMemory<T> {
    /// Borrows all initialized elements as a shared device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_memory(self)
    }

    /// Borrows all initialized elements as an exclusive device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_memory(self)
    }
}

impl<T: DeviceRepr> DeviceMemory<MaybeUninit<T>> {
    /// Borrows this uninitialized allocation as a write-only output of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }
}

unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceMemory<T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for DeviceMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for DeviceMemory<T> {}

unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceView<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for DeviceView<'_, T> {}

unsafe impl<T: DeviceRepr> DeviceInput<T> for DeviceViewMut<'_, T> {
    fn as_device_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.length
    }
}

// SAFETY: the typed owner or view preserves initialized values for its borrow.
unsafe impl<T: DeviceRepr> DeviceSlice<T> for DeviceViewMut<'_, T> {}

unsafe impl<T: DeviceRepr> DeviceSliceMut<T> for DeviceViewMut<'_, T> {}

unsafe impl<T: DeviceRepr> HostSlice<T> for [T] {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSliceMut<T> for [T] {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

unsafe impl<T: DeviceRepr, const N: usize> HostSlice<T> for [T; N] {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        N
    }
}

unsafe impl<T: DeviceRepr, const N: usize> HostSliceMut<T> for [T; N] {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

unsafe impl<T: DeviceRepr> HostSlice<T> for Vec<T> {
    fn as_host_ptr(&self) -> *const T {
        self.as_ptr()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

unsafe impl<T: DeviceRepr> HostSliceMut<T> for Vec<T> {
    fn as_host_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

fn bounds_to_range<R: RangeBounds<usize>>(range: R, length: usize) -> Result<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&value) => value,
        Bound::Excluded(&value) => value.checked_add(1).ok_or(Error::InvalidMemoryAccess)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&value) => value.checked_add(1).ok_or(Error::InvalidMemoryAccess)?,
        Bound::Excluded(&value) => value,
        Bound::Unbounded => length,
    };

    if start > end || end > length {
        return Err(Error::InvalidMemoryAccess);
    }

    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use crate::view::ranges_overlap;

    #[test]
    fn test_byte_range_conflicts_preserve_boundaries_and_aliases() {
        for (left, right, expected) in [
            (0..4, 4..8, false),
            (0..4, 8..12, false),
            (0..8, 4..12, true),
            (0..16, 4..8, true),
            (4..8, 4..8, true),
            (0..0, 0..0, false),
            // Hazard tracking conservatively includes an empty subrange inside another view.
            (0..8, 4..4, true),
        ] {
            assert_eq!(ranges_overlap(&left, &right), expected);
            assert_eq!(ranges_overlap(&right, &left), expected);
        }
    }
}
