//! CUDA arrays, mipmapped arrays and borrowed mip levels.

use std::{
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Range},
    ptr,
    result::Result as StdResult,
    sync::Arc,
};

use bitflags::parser;
use half::f16;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    stream::{StreamBinding, StreamScope},
    submission::{Access, AccessMode},
    try_ffi,
    utility::{driver_enum, to_u64},
    view::{AllocationId, AnyBitPatternDeviceRepr, DeviceOutput, DeviceSlice, DeviceViewMut},
};

driver_enum! {
    /// Array formats
    pub enum ArrayFormat: sys::CUarray_format {
        /// Unsigned 8-bit integers.
        U8 = CU_AD_FORMAT_UNSIGNED_INT8,
        /// Unsigned 16-bit integers.
        U16 = CU_AD_FORMAT_UNSIGNED_INT16,
        /// Unsigned 32-bit integers.
        U32 = CU_AD_FORMAT_UNSIGNED_INT32,
        /// Signed 8-bit integers.
        I8 = CU_AD_FORMAT_SIGNED_INT8,
        /// Signed 16-bit integers.
        I16 = CU_AD_FORMAT_SIGNED_INT16,
        /// Signed 32-bit integers.
        I32 = CU_AD_FORMAT_SIGNED_INT32,
        /// 16-bit floating point.
        F16 = CU_AD_FORMAT_HALF,
        /// 32-bit floating point.
        F32 = CU_AD_FORMAT_FLOAT,
    }
}

mod sealed {
    use crate::memory::array::ArrayFormat;

    pub trait Pixel {
        const FORMAT: ArrayFormat;
        const CHANNELS: u32;
    }
}

/// Native scalar or two/four-channel pixel representation supported by CUDA arrays.
pub trait ArrayElement: AnyBitPatternDeviceRepr + Send + Sync + sealed::Pixel {}

macro_rules! pixels {
    ($($ty:ty => $format:ident),* $(,)?) => {$ (
        impl sealed::Pixel for $ty {
            const FORMAT: ArrayFormat = ArrayFormat::$format;
            const CHANNELS: u32 = 1;
        }
        impl ArrayElement for $ty {}
        impl sealed::Pixel for [$ty; 2] {
            const FORMAT: ArrayFormat = ArrayFormat::$format;
            const CHANNELS: u32 = 2;
        }
        impl ArrayElement for [$ty; 2] {}
        impl sealed::Pixel for [$ty; 4] {
            const FORMAT: ArrayFormat = ArrayFormat::$format;
            const CHANNELS: u32 = 4;
        }
        impl ArrayElement for [$ty; 4] {}
    )*};
}

pixels!(u8 => U8, u16 => U16,
    u32 => U32, i8 => I8,
    i16 => I16, i32 => I32,
    f16 => F16, f32 => F32);

/// Array initialization state in which some texels may not yet contain valid values.
#[derive(Debug)]
pub struct Uninitialized;

/// Array initialization state in which every texel contains a valid element.
#[derive(Debug)]
pub struct Initialized;

/// Shape in texels, with layers and cube faces distinguished from spatial depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArrayShape {
    /// One-dimensional array.
    D1 {
        /// Width in texels.
        width: usize,
    },
    /// Two-dimensional array.
    D2 {
        /// Width in texels.
        width: usize,
        /// Height in texels.
        height: usize,
    },
    /// Three-dimensional array.
    D3 {
        /// Width in texels.
        width: usize,
        /// Height in texels.
        height: usize,
        /// Spatial depth in texels.
        depth: usize,
    },
    /// One-dimensional arrays stored in independently addressed layers.
    Layered1D {
        /// Width in texels.
        width: usize,
        /// Number of array layers.
        layers: usize,
    },
    /// Two-dimensional arrays stored in independently addressed layers.
    Layered2D {
        /// Width in texels.
        width: usize,
        /// Height in texels.
        height: usize,
        /// Number of array layers.
        layers: usize,
    },
    /// Six square faces forming a cube map.
    Cube {
        /// Width and height of each cube face in texels.
        edge: usize,
    },
    /// Layered cube maps, with six square faces per cube.
    CubeArray {
        /// Width and height of each cube face in texels.
        edge: usize,
        /// Number of cube maps, each containing six faces.
        cubes: usize,
    },
}

bitflags::bitflags! {
    /// Usage capabilities requested when allocating an array or mipmap chain.
    ///
    /// Layered and cube-map flags are derived from [`ArrayShape`]. CUDA validates
    /// the supported combinations of shape, format, and usage capabilities.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    pub struct ArrayFlags: u32 {
        /// Allows binding the array for surface loads and stores.
        const SURFACE_LOAD_STORE = sys::CUDA_ARRAY3D_SURFACE_LDST;
        /// Allows texture gather operations on the array.
        const TEXTURE_GATHER = sys::CUDA_ARRAY3D_TEXTURE_GATHER;
        /// Marks the array as a depth texture.
        const DEPTH_TEXTURE = sys::CUDA_ARRAY3D_DEPTH_TEXTURE;
        /// Allows binding the array as a color target in an external graphics API.
        const COLOR_ATTACHMENT = sys::CUDA_ARRAY3D_COLOR_ATTACHMENT;
        /// Enables use by hardware video encoding or decoding operations.
        const VIDEO_ENCODE_DECODE = sys::CUDA_ARRAY3D_VIDEO_ENCODE_DECODE;
    }
}

impl fmt::Display for ArrayFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

impl ArrayShape {
    fn native_dimensions(self) -> Result<([usize; 3], u32)> {
        let (dimensions, flags) = match self {
            Self::D1 { width } => ([width, 0, 0], 0),
            Self::D2 { width, height } => ([width, height, 0], 0),
            Self::D3 {
                width,
                height,
                depth,
            } => ([width, height, depth], 0),
            Self::Layered1D { width, layers } => ([width, 0, layers], sys::CUDA_ARRAY3D_LAYERED),
            Self::Layered2D {
                width,
                height,
                layers,
            } => ([width, height, layers], sys::CUDA_ARRAY3D_LAYERED),
            Self::Cube { edge } => ([edge, edge, 6], sys::CUDA_ARRAY3D_CUBEMAP),
            Self::CubeArray { edge, cubes } => (
                [
                    edge,
                    edge,
                    cubes
                        .checked_mul(6)
                        .ok_or(Error::InvalidMemoryAllocationRequest)?,
                ],
                sys::CUDA_ARRAY3D_CUBEMAP | sys::CUDA_ARRAY3D_LAYERED,
            ),
        };
        let valid = match self {
            Self::D1 { width } => width != 0,
            Self::D2 { width, height } => width != 0 && height != 0,
            Self::D3 {
                width,
                height,
                depth,
            } => width != 0 && height != 0 && depth != 0,
            Self::Layered1D { width, layers } => width != 0 && layers != 0,
            Self::Layered2D {
                width,
                height,
                layers,
            } => width != 0 && height != 0 && layers != 0,
            Self::Cube { edge } => edge != 0,
            Self::CubeArray { edge, cubes } => edge != 0 && cubes != 0,
        };
        if !valid {
            return Err(Error::ZeroValue {
                name: "array dimension".into(),
            });
        }
        Ok((dimensions, flags))
    }

    /// Returns the three-dimensional transfer extent in texels.
    ///
    /// Unused spatial dimensions become one. The last dimension counts layers for
    /// layered arrays and faces for cube maps.
    ///
    /// # Errors
    ///
    /// Returns an error if a required dimension is zero or the number of cube faces overflows `usize`.
    pub fn extent(self) -> Result<[usize; 3]> {
        let (dimensions, _) = self.native_dimensions()?;
        Ok(dimensions.map(|dimension| dimension.max(1)))
    }

    fn at_level(self, level: u32) -> Self {
        let shrink = |value: usize| value.checked_shr(level).unwrap_or(0).max(1);
        match self {
            Self::D1 { width } => Self::D1 {
                width: shrink(width),
            },
            Self::D2 { width, height } => Self::D2 {
                width: shrink(width),
                height: shrink(height),
            },
            Self::D3 {
                width,
                height,
                depth,
            } => Self::D3 {
                width: shrink(width),
                height: shrink(height),
                depth: shrink(depth),
            },
            Self::Layered1D { width, layers } => Self::Layered1D {
                width: shrink(width),
                layers,
            },
            Self::Layered2D {
                width,
                height,
                layers,
            } => Self::Layered2D {
                width: shrink(width),
                height: shrink(height),
                layers,
            },
            Self::Cube { edge } => Self::Cube { edge: shrink(edge) },
            Self::CubeArray { edge, cubes } => Self::CubeArray {
                edge: shrink(edge),
                cubes,
            },
        }
    }

    fn maximum_levels(self) -> u32 {
        let maximum = match self {
            Self::D1 { width } | Self::Layered1D { width, .. } => width,
            Self::D2 { width, height } | Self::Layered2D { width, height, .. } => width.max(height),
            Self::D3 {
                width,
                height,
                depth,
            } => width.max(height).max(depth),
            Self::Cube { edge } | Self::CubeArray { edge, .. } => edge,
        };
        maximum.bit_width()
    }
}

/// A rectangular transfer region in texels; `[0, 0, 0]` is the first texel/layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayRegion {
    /// Starting x, y, and z coordinates in texels; z selects layers or cube faces when present.
    pub origin: [usize; 3],
    /// Width, height, and depth of the transfer in texels, layers, or cube faces.
    pub extent: [usize; 3],
}

impl ArrayRegion {
    /// Creates a transfer region without checking it against an array shape.
    #[must_use]
    pub const fn new(origin: [usize; 3], extent: [usize; 3]) -> Self {
        Self { origin, extent }
    }

    /// Returns a region covering all texels, layers, or cube faces of `shape`.
    ///
    /// # Errors
    ///
    /// Returns an error if a required shape dimension is zero or the number of cube faces overflows `usize`.
    pub fn full(shape: ArrayShape) -> Result<Self> {
        Ok(Self::new([0; 3], shape.extent()?))
    }

    fn len(self) -> Result<usize> {
        volume(self.extent)
    }

    fn validate(self, shape: ArrayShape) -> Result<()> {
        for ((origin, extent), bound) in self
            .origin
            .into_iter()
            .zip(self.extent)
            .zip(shape.extent()?)
        {
            if origin.checked_add(extent).is_none_or(|end| end > bound) {
                return Err(Error::InvalidMemoryAccess);
            }
        }
        self.len()?;
        Ok(())
    }
}

fn volume(extent: [usize; 3]) -> Result<usize> {
    extent.into_iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(dimension)
            .ok_or(Error::InvalidMemoryAllocationRequest)
    })
}

/// Owns an array or retains the mipmapped parent owning one level.
#[derive(Debug)]
pub struct ArrayAllocation {
    context: Arc<Context>,
    handle: sys::CUarray,
    id: AllocationId,
    range: Range<usize>,
    shape: ArrayShape,
    flags: ArrayFlags,
    format: ArrayFormat,
    channels: u32,
    parent: Option<Arc<MipmappedAllocation>>,
}

// SAFETY: CUDA addresses are opaque to host code; typed borrows and raw caller contracts govern access.
unsafe impl Send for ArrayAllocation {}

unsafe impl Sync for ArrayAllocation {}

impl ArrayAllocation {
    /// Returns the context that owns the array.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Borrows the native array handle; ownership remains with this allocation.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUarray {
        self.handle
    }

    /// Returns the allocation identity used to track conflicting accesses.
    #[must_use]
    pub fn allocation_id(&self) -> AllocationId {
        self.id
    }

    /// Returns the logical byte range used for access tracking.
    ///
    /// Mip levels occupy distinct ranges within their parent allocation identity.
    /// This range is not a device address or a measure of physical array storage.
    #[must_use]
    pub fn byte_range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the spatial dimensions and layer or cube organization.
    #[must_use]
    pub fn shape(&self) -> ArrayShape {
        self.shape
    }

    /// Returns the usage flags, excluding flags derived from the shape.
    #[must_use]
    pub fn flags(&self) -> ArrayFlags {
        self.flags
    }

    /// Returns the native format of each channel.
    #[must_use]
    pub fn format(&self) -> ArrayFormat {
        self.format
    }

    /// Returns the number of scalar channels per texel.
    #[must_use]
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Describes an access to the whole array or mip level for dependency tracking.
    #[must_use]
    pub fn access(&self, mode: AccessMode) -> Access {
        Access::allocation(&self.context, self.id, self.range.clone(), mode)
    }
}

impl Drop for ArrayAllocation {
    fn drop(&mut self) {
        if self.handle.is_null() || self.parent.is_some() {
            return;
        }
        if let Err(error) = self
            .context
            .bind()
            // SAFETY: `cuArrayDestroy` receives live handles owned by the wrappers.
            .and_then(|()| unsafe { try_ffi!(sys::cuArrayDestroy(self.handle)) })
        {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy cuda array: {error}");
        }
    }
}

/// Typed array storage with explicit whole-array initialization state.
#[derive(Debug)]
pub struct Array<T: ArrayElement, S = Initialized> {
    allocation: Arc<ArrayAllocation>,
    marker: PhantomData<(T, S)>,
}

impl<T: ArrayElement, S> Array<T, S> {
    /// Returns the shared allocation that keeps the native array alive.
    #[must_use]
    pub fn allocation(&self) -> &Arc<ArrayAllocation> {
        &self.allocation
    }

    /// Returns the context that owns the array.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows the native array handle without transferring destruction responsibility.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUarray {
        self.allocation.as_raw()
    }

    /// Returns the spatial dimensions and layer or cube organization.
    #[must_use]
    pub fn shape(&self) -> ArrayShape {
        self.allocation.shape
    }

    /// Returns the usage flags, excluding flags derived from the shape.
    #[must_use]
    pub fn flags(&self) -> ArrayFlags {
        self.allocation.flags
    }

    /// Returns the number of `T` texels across all layers or cube faces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.allocation.range.len() / size_of::<T>()
    }

    /// Returns `false`; successful array construction requires nonzero dimensions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    fn endpoint(&self, region: ArrayRegion) -> Result<ArrayEndpoint> {
        region.validate(self.shape())?;
        Ok(ArrayEndpoint {
            handle: self.as_raw(),
            origin: [
                to_u64(
                    region.origin[0]
                        .checked_mul(size_of::<T>())
                        .ok_or(Error::InvalidMemoryAccess)?,
                    "array x offset",
                )?,
                to_u64(region.origin[1], "array y offset")?,
                to_u64(region.origin[2], "array z offset")?,
            ],
        })
    }

    fn run_copy(&self, raw: &sys::CUDA_MEMCPY3D, accesses: &[Access]) -> Result<()> {
        let _guard = self.context().acquire_accesses(accesses)?;
        self.context().bind()?;
        let binding = StreamBinding::Default(Arc::clone(self.context()));
        // SAFETY: `cuMemcpy3DAsync_v2` receives live handles owned by the wrappers.
        binding.sync_scope(|| unsafe { try_ffi!(sys::cuMemcpy3DAsync_v2(raw, binding.as_raw())) })
    }

    /// Writes a checked rectangular region without promoting whole-array initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the region is outside the array, the element count does not match the region volume, or transfer size or offset calculations overflow. Also propagates access-tracking, context-binding, and CUDA copy or synchronization errors.
    pub fn copy_from_host(&mut self, region: ArrayRegion, source: &[T]) -> Result<()> {
        if source.len() != region.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Host(source.as_ptr().cast()),
            CopyDestination::Array(self.endpoint(region)?),
        )?;
        self.run_copy(&raw, &[self.allocation.access(AccessMode::Write)])
    }

    /// Enqueues a rectangular host-to-array copy in `scope`.
    ///
    /// The scope retains the source and array borrows until its queued work completes.
    /// A region copy does not change the whole-array initialization state.
    ///
    /// # Errors
    ///
    /// Returns an error if the scope belongs to another context, the source length does not match the region, the region is out of bounds, or size calculations overflow. Also propagates scope submission and CUDA copy errors.
    pub fn copy_from_host_async<'env>(
        &'env mut self,
        region: ArrayRegion,
        source: &'env [T],
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        if self.context() != scope.stream().context() {
            return Err(Error::MemoryContextMismatch);
        }
        if source.len() != region.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Host(source.as_ptr().cast()),
            CopyDestination::Array(self.endpoint(region)?),
        )?;
        let accesses = [self.allocation.access(AccessMode::Write)];
        // SAFETY: `cuMemcpy3DAsync_v2` receives only plain values.
        unsafe {
            scope.enqueue(&accesses, || {
                try_ffi!(sys::cuMemcpy3DAsync_v2(
                    &raw const raw,
                    scope.stream().as_raw()
                ))
            })
        }
    }

    /// Copies initialized contiguous device values into a checked rectangular array region.
    ///
    /// # Errors
    ///
    /// Returns an error if the source belongs to another context, its length does not match the region, the region is out of bounds, or size calculations overflow. Also propagates access-tracking, context-binding, and CUDA copy or synchronization errors.
    pub fn copy_from_device<B: DeviceSlice<T> + ?Sized>(
        &mut self,
        region: ArrayRegion,
        source: &B,
    ) -> Result<()> {
        if source.context() != self.context() {
            return Err(Error::MemoryContextMismatch);
        }
        if source.len() != region.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Device(source.as_device_ptr() as _),
            CopyDestination::Array(self.endpoint(region)?),
        )?;
        self.run_copy(
            &raw,
            &[
                self.allocation.access(AccessMode::Write),
                Access::new(source.region(), AccessMode::Read),
            ],
        )
    }

    /// Copies an initialized array region; overlapping regions in the same array are rejected.
    ///
    /// # Errors
    ///
    /// Returns an error if the arrays belong to different contexts, either region is out of bounds, the regions overlap within the same array, or size calculations overflow. Also propagates access-tracking, context-binding, and CUDA copy or synchronization errors.
    pub fn copy_from_array(
        &mut self,
        destination: [usize; 3],
        source: &Array<T>,
        source_region: ArrayRegion,
    ) -> Result<()> {
        if source.context() != self.context() {
            return Err(Error::MemoryContextMismatch);
        }
        source_region.validate(source.shape())?;
        let region = ArrayRegion::new(destination, source_region.extent);
        region.validate(self.shape())?;
        if self.as_raw() == source.as_raw()
            && (0..3).all(|axis| {
                destination[axis] < source_region.origin[axis] + source_region.extent[axis]
                    && source_region.origin[axis] < destination[axis] + region.extent[axis]
            })
        {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Array(source.endpoint(source_region)?),
            CopyDestination::Array(self.endpoint(region)?),
        )?;
        self.run_copy(
            &raw,
            &[
                self.allocation.access(AccessMode::Write),
                source.allocation.access(AccessMode::Read),
            ],
        )
    }

    /// Transfers ownership of a standalone array handle when no retained aliases remain.
    ///
    /// # Errors
    ///
    /// Returns the original array if it is a borrowed mip level or another owner retains its allocation.
    pub fn into_raw(self) -> StdResult<(sys::CUarray, Arc<Context>), Self> {
        if self.allocation.parent.is_some() || Arc::strong_count(&self.allocation) != 1 {
            return Err(self);
        }
        match Arc::try_unwrap(self.allocation) {
            Ok(mut allocation) => Ok((
                mem::replace(&mut allocation.handle, ptr::null_mut()),
                Arc::clone(&allocation.context),
            )),
            Err(allocation) => Err(Self {
                allocation,
                marker: PhantomData,
            }),
        }
    }
}

struct ArrayEndpoint {
    handle: sys::CUarray,
    origin: [u64; 3],
}

enum CopySource {
    Host(*const c_void),
    Device(sys::CUdeviceptr),
    Array(ArrayEndpoint),
}

enum CopyDestination {
    Host(*mut c_void),
    Device(sys::CUdeviceptr),
    Array(ArrayEndpoint),
}

fn transfer<T: ArrayElement>(
    extent: [usize; 3],
    source: CopySource,
    destination: CopyDestination,
) -> Result<sys::CUDA_MEMCPY3D> {
    let row_bytes = extent[0]
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAllocationRequest)?;
    volume(extent)?
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAllocationRequest)?;
    let mut raw = MaybeUninit::<sys::CUDA_MEMCPY3D>::zeroed();
    // SAFETY: all fields except the two memory-type enums accept zero.
    // Write valid discriminants before constructing a Rust value.
    let mut raw = unsafe {
        (*raw.as_mut_ptr()).srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
        (*raw.as_mut_ptr()).dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
        raw.assume_init()
    };
    raw.WidthInBytes = to_u64(row_bytes, "array row bytes")?;
    raw.Height = to_u64(extent[1], "array transfer height")?;
    raw.Depth = to_u64(extent[2], "array transfer depth")?;
    match source {
        CopySource::Host(pointer) => {
            raw.srcHost = pointer;
            raw.srcPitch = raw.WidthInBytes;
            raw.srcHeight = raw.Height;
        }
        CopySource::Device(pointer) => {
            raw.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
            raw.srcDevice = pointer;
            raw.srcPitch = raw.WidthInBytes;
            raw.srcHeight = raw.Height;
        }
        CopySource::Array(array) => {
            raw.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
            raw.srcArray = array.handle;
            [raw.srcXInBytes, raw.srcY, raw.srcZ] = array.origin;
        }
    }
    match destination {
        CopyDestination::Host(pointer) => {
            raw.dstHost = pointer;
            raw.dstPitch = raw.WidthInBytes;
            raw.dstHeight = raw.Height;
        }
        CopyDestination::Device(pointer) => {
            raw.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
            raw.dstDevice = pointer;
            raw.dstPitch = raw.WidthInBytes;
            raw.dstHeight = raw.Height;
        }
        CopyDestination::Array(array) => {
            raw.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_ARRAY;
            raw.dstArray = array.handle;
            [raw.dstXInBytes, raw.dstY, raw.dstZ] = array.origin;
        }
    }
    Ok(raw)
}

fn descriptor<T: ArrayElement>(
    shape: ArrayShape,
    flags: ArrayFlags,
) -> Result<(sys::CUDA_ARRAY3D_DESCRIPTOR, usize)> {
    let (dimensions, shape_flags) = shape.native_dimensions()?;
    let bytes = volume(shape.extent()?)?
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidMemoryAllocationRequest)?;
    let descriptor = sys::CUDA_ARRAY3D_DESCRIPTOR {
        Width: to_u64(dimensions[0], "array width")?,
        Height: to_u64(dimensions[1], "array height")?,
        Depth: to_u64(dimensions[2], "array depth")?,
        Format: T::FORMAT.into(),
        NumChannels: T::CHANNELS,
        Flags: flags.bits() | shape_flags,
    };
    Ok((descriptor, bytes))
}

impl<T: ArrayElement> Array<T, Uninitialized> {
    /// Allocates an array without initializing its texels.
    ///
    /// `T` determines the scalar format and number of channels. The shape determines
    /// layered and cube-map flags in addition to the supplied usage flags.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero, size calculations overflow, flags contain unknown bits, or CUDA rejects the requested format, dimensions, flags, or allocation. Also propagates context-binding errors and rejects a null returned handle.
    pub fn create(context: &Arc<Context>, shape: ArrayShape, flags: ArrayFlags) -> Result<Self> {
        if flags.bits() & !ArrayFlags::all().bits() != 0 {
            return Err(Error::InvalidValue);
        }
        let (descriptor, bytes) = descriptor::<T>(shape, flags)?;
        context.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cuArray3DCreate_v2` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuArray3DCreate_v2(
                &raw mut handle,
                &raw const descriptor
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            allocation: Arc::new(ArrayAllocation {
                context: Arc::clone(context),
                handle,
                id: AllocationId::create(),
                range: 0..bytes,
                shape,
                flags,
                format: T::FORMAT,
                channels: T::CHANNELS,
                parent: None,
            }),
            marker: PhantomData,
        })
    }

    /// Copies all host texels into the array and returns it as initialized.
    ///
    /// The source is contiguous in x, then y, then z, layer, or cube-face order.
    ///
    /// # Errors
    ///
    /// Returns an error if the source length differs from the full array volume or the full-array host copy fails. See [`Array::copy_from_host`].
    pub fn initialize(mut self, source: &[T]) -> Result<Array<T>> {
        self.copy_from_host(ArrayRegion::full(self.shape())?, source)?;
        // SAFETY: the completed full-array copy wrote valid values to every texel.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks the complete array initialized after external producers complete.
    ///
    /// # Safety
    /// All texels must contain valid values of `T`, with producer and subsequent consumer order established.
    #[must_use]
    pub unsafe fn assume_init(self) -> Array<T> {
        Array {
            allocation: self.allocation,
            marker: PhantomData,
        }
    }

    /// Adopts an independently owned array with an explicit format and shape contract.
    ///
    /// # Safety
    /// The handle must belong to `context`, match this exact typed descriptor, and transfer destruction responsibility.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null, the expected shape or size is invalid, or the native descriptor differs from the expected typed descriptor. Also propagates context-binding and CUDA descriptor-query errors.
    pub unsafe fn from_raw(
        handle: sys::CUarray,
        context: Arc<Context>,
        shape: ArrayShape,
        flags: ArrayFlags,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        let (expected, bytes) = descriptor::<T>(shape, flags)?;
        context.bind()?;
        let mut actual = MaybeUninit::<sys::CUDA_ARRAY3D_DESCRIPTOR>::uninit();
        // SAFETY: `cuArray3DGetDescriptor_v2` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuArray3DGetDescriptor_v2(actual.as_mut_ptr(), handle))?;
        }
        // SAFETY: `cuArray3DGetDescriptor_v2` succeeded, so it wrote the whole descriptor.
        let actual = unsafe { actual.assume_init() };
        if expected != actual {
            return Err(Error::InvalidMemoryAccess);
        }
        Ok(Self {
            allocation: Arc::new(ArrayAllocation {
                context,
                handle,
                id: AllocationId::create(),
                range: 0..bytes,
                shape,
                flags,
                format: T::FORMAT,
                channels: T::CHANNELS,
                parent: None,
            }),
            marker: PhantomData,
        })
    }
}

impl<T: ArrayElement> Array<T> {
    /// Allocates an array and initializes every texel from contiguous host data.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or `source` does not match the full array volume. See [`Array::create`] and [`Array::initialize`].
    pub fn from_slice(
        context: &Arc<Context>,
        shape: ArrayShape,
        flags: ArrayFlags,
        source: &[T],
    ) -> Result<Self> {
        Array::<T, Uninitialized>::create(context, shape, flags)?.initialize(source)
    }

    /// Copies an initialized array region into a contiguous host slice and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the region is outside the array, the element count does not match the region volume, or transfer size or offset calculations overflow. Also propagates access-tracking, context-binding, and CUDA copy or synchronization errors.
    pub fn copy_to_host(&self, region: ArrayRegion, destination: &mut [T]) -> Result<()> {
        if destination.len() != region.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Array(self.endpoint(region)?),
            CopyDestination::Host(destination.as_mut_ptr().cast()),
        )?;
        self.run_copy(&raw, &[self.allocation.access(AccessMode::Read)])
    }

    /// Copies an initialized array region into a new contiguous host vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the region is out of bounds or size calculations overflow. Also propagates access-tracking, context-binding, and CUDA copy or synchronization errors.
    pub fn to_vec(&self, region: ArrayRegion) -> Result<Vec<T>> {
        region.validate(self.shape())?;
        let mut values = vec![MaybeUninit::<T>::uninit(); region.len()?];
        let raw = transfer::<T>(
            region.extent,
            CopySource::Array(self.endpoint(region)?),
            CopyDestination::Host(values.as_mut_ptr().cast()),
        )?;
        self.run_copy(&raw, &[self.allocation.access(AccessMode::Read)])?;
        let mut values = ManuallyDrop::new(values);
        // SAFETY: every element was copied from valid initialized texels; layout and allocation are unchanged.
        Ok(unsafe {
            Vec::from_raw_parts(values.as_mut_ptr().cast(), values.len(), values.capacity())
        })
    }

    /// Copies an array region into a contiguous device destination and waits for completion.
    ///
    /// Returns an initialized mutable view covering the destination.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination belongs to another context, its length does not match the region, the region is out of bounds, or size calculations overflow. Also propagates access-tracking, context-binding, CUDA copy or synchronization, and destination-view validation errors.
    pub fn copy_to_device<'a, B: DeviceOutput<T> + ?Sized>(
        &self,
        region: ArrayRegion,
        destination: &'a mut B,
    ) -> Result<DeviceViewMut<'a, T>> {
        if destination.context() != self.context() {
            return Err(Error::MemoryContextMismatch);
        }
        if destination.len() != region.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Array(self.endpoint(region)?),
            CopyDestination::Device(destination.as_mut_ptr() as _),
        )?;
        self.run_copy(
            &raw,
            &[
                self.allocation.access(AccessMode::Read),
                Access::new(destination.region(), AccessMode::Write),
            ],
        )?;
        // SAFETY: the view covers exactly the elements of the borrowed allocation and lives no longer than the borrow.
        unsafe { DeviceViewMut::from_raw_parts(destination.region(), destination.len()) }
    }

    /// Enqueues a rectangular array-to-host copy in `scope`.
    ///
    /// The scope retains the array and destination borrows until its queued work completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the scope belongs to another context, the destination length does not match the region, the region is out of bounds, or size calculations overflow. Also propagates scope submission and CUDA copy errors.
    pub fn copy_to_host_async<'env>(
        &'env self,
        region: ArrayRegion,
        destination: &'env mut [T],
        scope: &StreamScope<'_, 'env>,
    ) -> Result<()> {
        if self.context() != scope.stream().context() {
            return Err(Error::MemoryContextMismatch);
        }
        if destination.len() != region.len()? {
            return Err(Error::InvalidMemoryAccess);
        }
        let raw = transfer::<T>(
            region.extent,
            CopySource::Array(self.endpoint(region)?),
            CopyDestination::Host(destination.as_mut_ptr().cast()),
        )?;
        let accesses = [self.allocation.access(AccessMode::Read)];
        // SAFETY: `cuMemcpy3DAsync_v2` receives only plain values.
        unsafe {
            scope.enqueue(&accesses, || {
                try_ffi!(sys::cuMemcpy3DAsync_v2(
                    &raw const raw,
                    scope.stream().as_raw()
                ))
            })
        }
    }
}

/// Owns all levels and supplies one allocation identity with distinct level ranges.
#[derive(Debug)]
pub struct MipmappedAllocation {
    context: Arc<Context>,
    handle: sys::CUmipmappedArray,
    id: AllocationId,
    shape: ArrayShape,
    flags: ArrayFlags,
    levels: u32,
    ranges: Vec<Range<usize>>,
}

// SAFETY: immutable descriptors and raw handles are shareable; typed borrows govern mutation.
unsafe impl Send for MipmappedAllocation {}

unsafe impl Sync for MipmappedAllocation {}

impl MipmappedAllocation {
    /// Returns the context that owns the mipmap chain.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Borrows the native mipmapped-array handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUmipmappedArray {
        self.handle
    }

    /// Returns the number of mip levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Returns the shape of mip level zero.
    #[must_use]
    pub fn shape(&self) -> ArrayShape {
        self.shape
    }

    /// Returns the usage flags, excluding flags derived from the shape.
    #[must_use]
    pub fn flags(&self) -> ArrayFlags {
        self.flags
    }

    /// Returns the allocation identity shared by all mip levels.
    #[must_use]
    pub fn allocation_id(&self) -> AllocationId {
        self.id
    }

    /// Describes an access spanning every mip level for dependency tracking.
    #[must_use]
    pub fn access(&self, mode: AccessMode) -> Access {
        Access::allocation(
            &self.context,
            self.id,
            0..self.ranges.last().map_or(0, |range| range.end),
            mode,
        )
    }
}

impl Drop for MipmappedAllocation {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Err(error) = self
            .context
            .bind()
            // SAFETY: `cuMipmappedArrayDestroy` receives live handles owned by the wrappers.
            .and_then(|()| unsafe { try_ffi!(sys::cuMipmappedArrayDestroy(self.handle)) })
        {
            #[cfg(debug_assertions)]
            eprintln!("failed to destroy cuda mipmapped array: {error}");
        }
    }
}

/// Owns a typed mipmap chain with one initialization state covering every level.
#[derive(Debug)]
pub struct MipmappedArray<T: ArrayElement, S = Initialized> {
    allocation: Arc<MipmappedAllocation>,
    marker: PhantomData<(T, S)>,
}

impl<T: ArrayElement, S> MipmappedArray<T, S> {
    /// Returns the shared allocation that owns all mip levels.
    #[must_use]
    pub fn allocation(&self) -> &Arc<MipmappedAllocation> {
        &self.allocation
    }

    /// Borrows the native mipmapped-array handle without transferring ownership.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUmipmappedArray {
        self.allocation.as_raw()
    }

    /// Returns the context that owns the mipmap chain.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Returns the number of mip levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.allocation.levels
    }

    fn array_at_level(&self, level: u32) -> Result<Array<T, S>> {
        if level >= self.levels() {
            return Err(Error::out_of_range("mipmap level"));
        }
        self.context().bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cuMipmappedArrayGetLevel` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuMipmappedArrayGetLevel(
                &raw mut handle,
                self.as_raw(),
                level
            ))?;
        }
        Ok(Array {
            allocation: Arc::new(ArrayAllocation {
                context: Arc::clone(self.context()),
                handle,
                id: self.allocation.id,
                range: self.allocation.ranges[level as usize].clone(),
                shape: self.allocation.shape.at_level(level),
                flags: self.allocation.flags,
                format: T::FORMAT,
                channels: T::CHANNELS,
                parent: Some(Arc::clone(&self.allocation)),
            }),
            marker: PhantomData,
        })
    }

    /// Borrows a mip level for reading while retaining its parent allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `level` is outside `0..self.levels()`, context binding fails, or CUDA cannot retrieve the level handle.
    pub fn level(&self, level: u32) -> Result<ArrayLevel<'_, T, S>> {
        Ok(ArrayLevel {
            array: self.array_at_level(level)?,
            parent: PhantomData,
        })
    }

    /// Borrows a mip level exclusively while retaining its parent allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `level` is outside `0..self.levels()`, context binding fails, or CUDA cannot retrieve the level handle.
    pub fn level_mut(&mut self, level: u32) -> Result<ArrayLevelMut<'_, T, S>> {
        Ok(ArrayLevelMut {
            array: self.array_at_level(level)?,
            parent: PhantomData,
        })
    }

    /// Transfers ownership of the complete native mipmap chain and its context.
    ///
    /// # Errors
    ///
    /// Returns the original mipmapped array if another owner retains the allocation, including a retained mip level.
    pub fn into_raw(self) -> StdResult<(sys::CUmipmappedArray, Arc<Context>), Self> {
        match Arc::try_unwrap(self.allocation) {
            Ok(mut allocation) => Ok((
                mem::replace(&mut allocation.handle, ptr::null_mut()),
                Arc::clone(&allocation.context),
            )),
            Err(allocation) => Err(Self {
                allocation,
                marker: PhantomData,
            }),
        }
    }
}

impl<T: ArrayElement> MipmappedArray<T, Uninitialized> {
    /// Adopts a complete mip chain with independently owned destruction responsibility.
    ///
    /// # Safety
    /// The handle must belong to `context`, match the exact descriptor and level count,
    /// and have dense mapped storage before any transfer or sampling operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is null, the shape or byte ranges are invalid, the requested level count is zero or exceeds the shape maximum, or level zero has a different typed descriptor. Also propagates context-binding and CUDA level or descriptor-query errors.
    pub unsafe fn from_raw(
        handle: sys::CUmipmappedArray,
        context: Arc<Context>,
        shape: ArrayShape,
        flags: ArrayFlags,
        levels: u32,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        let (expected, _) = descriptor::<T>(shape, flags)?;
        let ranges = mip_level_ranges::<T>(shape, levels)?;
        context.bind()?;
        let mut level = ptr::null_mut();
        // SAFETY: `cuMipmappedArrayGetLevel` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuMipmappedArrayGetLevel(&raw mut level, handle, 0))?;
        }
        let mut actual = MaybeUninit::<sys::CUDA_ARRAY3D_DESCRIPTOR>::uninit();
        // SAFETY: `cuArray3DGetDescriptor_v2` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuArray3DGetDescriptor_v2(actual.as_mut_ptr(), level))?;
        }
        // SAFETY: `cuArray3DGetDescriptor_v2` succeeded, so it wrote the whole descriptor.
        if expected != unsafe { actual.assume_init() } {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: `cuMipmappedArrayGetLevel` receives valid output locations.
        unsafe {
            try_ffi!(sys::cuMipmappedArrayGetLevel(
                &raw mut level,
                handle,
                levels - 1
            ))?;
        }
        Ok(Self {
            allocation: Arc::new(MipmappedAllocation {
                context,
                handle,
                id: AllocationId::create(),
                shape,
                flags,
                levels,
                ranges,
            }),
            marker: PhantomData,
        })
    }

    /// Allocates a mipmap chain without initializing its texels.
    ///
    /// Each level halves spatial dimensions, rounded down to a minimum of one;
    /// layer and cube counts remain constant.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are zero, size calculations overflow, flags contain unknown bits, or `levels` is zero or exceeds the shape maximum. Also propagates context-binding and CUDA allocation errors and rejects a null returned handle.
    pub fn create(
        context: &Arc<Context>,
        shape: ArrayShape,
        flags: ArrayFlags,
        levels: u32,
    ) -> Result<Self> {
        if flags.bits() & !ArrayFlags::all().bits() != 0 {
            return Err(Error::InvalidValue);
        }
        let (desc, _) = descriptor::<T>(shape, flags)?;
        let ranges = mip_level_ranges::<T>(shape, levels)?;
        context.bind()?;
        let mut handle = ptr::null_mut();
        // SAFETY: `cuMipmappedArrayCreate` receives valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuMipmappedArrayCreate(
                &raw mut handle,
                &raw const desc,
                levels
            ))?;
        }
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            allocation: Arc::new(MipmappedAllocation {
                context: Arc::clone(context),
                handle,
                id: AllocationId::create(),
                shape,
                flags,
                levels,
                ranges,
            }),
            marker: PhantomData,
        })
    }

    /// Initializes every mip level from one contiguous host slice per level.
    ///
    /// `levels[0]` supplies the largest level. Each slice must contain all texels of
    /// that level, including its layers or cube faces.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice count differs from the mip level count, any slice has the wrong length, or retrieving or copying a level fails.
    ///
    /// # Panics
    ///
    /// Panics if a level index cannot fit in `u32`. Construction bounds the level count by `u32`, so this condition cannot occur for a valid mipmapped array.
    pub fn initialize(mut self, levels: &[&[T]]) -> Result<MipmappedArray<T>> {
        if levels.len() != self.levels() as usize {
            return Err(Error::InvalidMemoryAccess);
        }
        for (level, values) in levels.iter().enumerate() {
            let mut array = self.level_mut(
                u32::try_from(level).expect("mipmap level is bounded by the u32 level count"),
            )?;
            let region = ArrayRegion::full(array.shape())?;
            array.copy_from_host(region, values)?;
        }
        // SAFETY: the copy above wrote every element.
        Ok(unsafe { self.assume_init() })
    }

    /// Marks every mip level initialized after external producers complete.
    ///
    /// # Safety
    /// Every texel in every level must be initialized and all producers must be ordered before readers.
    #[must_use]
    pub unsafe fn assume_init(self) -> MipmappedArray<T> {
        MipmappedArray {
            allocation: self.allocation,
            marker: PhantomData,
        }
    }
}

fn mip_level_ranges<T: ArrayElement>(shape: ArrayShape, levels: u32) -> Result<Vec<Range<usize>>> {
    if levels == 0 || levels > shape.maximum_levels() {
        return Err(Error::out_of_range("mipmap level count"));
    }
    let mut end = 0usize;
    (0..levels)
        .map(|level| {
            let bytes = volume(shape.at_level(level).extent()?)?
                .checked_mul(size_of::<T>())
                .ok_or(Error::InvalidMemoryAllocationRequest)?;
            let start = end;
            end = end
                .checked_add(bytes)
                .ok_or(Error::InvalidMemoryAllocationRequest)?;
            Ok(start..end)
        })
        .collect()
}

/// Shared borrowed mip level retaining the actual parent allocation.
#[derive(Debug)]
pub struct ArrayLevel<'a, T: ArrayElement, S = Initialized> {
    array: Array<T, S>,
    parent: PhantomData<&'a MipmappedArray<T, S>>,
}

impl<T: ArrayElement, S> Deref for ArrayLevel<'_, T, S> {
    type Target = Array<T, S>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

/// Exclusive borrowed mip level; writes cannot overlap another safe parent borrow.
#[derive(Debug)]
pub struct ArrayLevelMut<'a, T: ArrayElement, S = Initialized> {
    array: Array<T, S>,
    parent: PhantomData<&'a mut MipmappedArray<T, S>>,
}

impl<T: ArrayElement, S> Deref for ArrayLevelMut<'_, T, S> {
    type Target = Array<T, S>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl<T: ArrayElement, S> DerefMut for ArrayLevelMut<'_, T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.array
    }
}
