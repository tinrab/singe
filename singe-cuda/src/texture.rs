//! Modern texture and surface objects retaining their backing storage.
//! Legacy module references remain available through `module`.

use std::{
    any::Any,
    fmt::{self, Debug, Formatter},
    mem, ptr,
    sync::Arc,
};

use bitflags::parser;
use singe_core::report_cleanup_failure;
use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    memory::array::{ArrayAllocation, ArrayElement, ArrayFlags, MipmappedAllocation},
    submission::{Access, AccessMode, Resources},
    try_ffi,
    utility::{checked_pitched_extent, driver_enum, to_u64},
    view::{DeviceRegionProvider, OwnedDeviceRegion},
};

driver_enum! {
    /// Texture reference addressing modes
    #[derive(Default)]
    pub enum AddressMode: sys::CUaddress_mode {
        /// Wrapping address mode.
        Wrap = CU_TR_ADDRESS_MODE_WRAP,
        /// Clamp to edge address mode.
        #[default]
        Clamp = CU_TR_ADDRESS_MODE_CLAMP,
        /// Mirror address mode.
        Mirror = CU_TR_ADDRESS_MODE_MIRROR,
        /// Border address mode.
        Border = CU_TR_ADDRESS_MODE_BORDER,
    }
}

driver_enum! {
    /// Texture reference filtering modes
    #[derive(Default)]
    pub enum FilterMode: sys::CUfilter_mode {
        /// Point filter mode.
        #[default]
        Point = CU_TR_FILTER_MODE_POINT,
        /// Linear filter mode.
        Linear = CU_TR_FILTER_MODE_LINEAR,
    }
}

bitflags::bitflags! {
    /// Texture coordinate, value-conversion, and filtering flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureFlags: u32 {
        /// Read integer texels as integers instead of promoting them to normalized floating-point values.
        const READ_AS_INTEGER = sys::CU_TRSF_READ_AS_INTEGER;
        /// Use normalized coordinates in `[0, 1)` instead of `[0, dimension)`.
        const NORMALIZED_COORDINATES = sys::CU_TRSF_NORMALIZED_COORDINATES;
        /// Convert sRGB samples to linear color values during texture reads.
        const SRGB = sys::CU_TRSF_SRGB;
        /// Disable trilinear filtering optimizations.
        const DISABLE_TRILINEAR_OPTIMIZATION = sys::CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION;
        /// Enable seamless cube-map filtering.
        const SEAMLESS_CUBEMAP = sys::CU_TRSF_SEAMLESS_CUBEMAP;
    }
}

impl fmt::Display for TextureFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Texture addressing, filtering, mipmap, and border-color settings.
///
/// Defaults clamp all axes, use point filtering and integer reads, set anisotropy to one, cover the resource's available mipmap levels, and use a zero border color.
#[derive(Debug, Clone, Copy)]
pub struct TextureOptions {
    /// Addressing behavior for each of the three texture coordinate axes.
    pub address_modes: [AddressMode; 3],
    /// Filtering within a texture level.
    pub filter: FilterMode,
    /// Coordinate, value-conversion, and filtering flags.
    pub flags: TextureFlags,
    /// Maximum anisotropy ratio requested from CUDA.
    pub max_anisotropy: u32,
    /// Filtering between mipmap levels.
    pub mipmap_filter: FilterMode,
    /// Bias added to the mipmap level used for sampling.
    pub mipmap_bias: f32,
    /// `None` covers every level available in the resource.
    pub mipmap_level_clamp: Option<[f32; 2]>,
    /// Four-component color used by border addressing.
    pub border_color: [f32; 4],
}

impl Default for TextureOptions {
    fn default() -> Self {
        Self {
            address_modes: [AddressMode::Clamp; 3],
            filter: FilterMode::Point,
            flags: TextureFlags::READ_AS_INTEGER,
            max_anisotropy: 1,
            mipmap_filter: FilterMode::Point,
            mipmap_bias: 0.0,
            mipmap_level_clamp: None,
            border_color: [0.0; 4],
        }
    }
}

impl TextureOptions {
    /// Returns the default sampler settings with normalized coordinates enabled for mipmapped resources.
    #[must_use]
    pub fn for_mipmaps() -> Self {
        Self {
            flags: TextureFlags::READ_AS_INTEGER | TextureFlags::NORMALIZED_COORDINATES,
            ..Self::default()
        }
    }

    fn native(self, levels: u32) -> sys::CUDA_TEXTURE_DESC {
        let clamp = self.mipmap_level_clamp.unwrap_or_else(|| {
            [
                0.0,
                f32::from(u16::try_from(levels.saturating_sub(1)).expect("mipmap levels fit u16")),
            ]
        });
        sys::CUDA_TEXTURE_DESC {
            addressMode: self.address_modes.map(Into::into),
            filterMode: self.filter.into(),
            flags: self.flags.bits(),
            maxAnisotropy: self.max_anisotropy,
            mipmapFilterMode: self.mipmap_filter.into(),
            mipmapLevelBias: self.mipmap_bias,
            minMipmapLevelClamp: clamp[0],
            maxMipmapLevelClamp: clamp[1],
            borderColor: self.border_color,
            reserved: [0; 12],
        }
    }
}

/// Backing storage, without an initialization or sampling-permission claim.
pub enum TextureResource {
    /// A retained CUDA array allocation.
    Array(Arc<ArrayAllocation>),
    /// A retained CUDA mipmapped array allocation.
    MipmappedArray(Arc<MipmappedAllocation>),
    /// A retained linear device-memory region and its texture element format.
    Linear(LinearTextureResource),
    /// A retained pitched two-dimensional device-memory region and its texture element format.
    Pitched2D(PitchedTextureResource),
}

impl Debug for TextureResource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextureResource")
            .field("context", &self.context())
            .finish_non_exhaustive()
    }
}

/// Retained linear texture descriptor; construct through [`TextureResource::linear`].
#[derive(Debug)]
pub struct LinearTextureResource {
    storage: TextureStorage,
    bytes: u64,
}

#[derive(Debug)]
struct TextureStorage {
    context: Arc<Context>,
    _owner: Arc<dyn Any + Send + Sync>,
    access: Access,
    pointer: sys::CUdeviceptr,
    format: sys::CUarray_format,
    channels: u32,
}

/// Retained pitched texture descriptor; construct through [`TextureResource::pitched_2d`].
pub struct PitchedTextureResource {
    storage: TextureStorage,
    width: u64,
    height: u64,
    pitch: u64,
}

impl Debug for PitchedTextureResource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PitchedTextureResource")
            .finish_non_exhaustive()
    }
}

impl TextureStorage {
    fn from_region<T: ArrayElement, O: DeviceRegionProvider + Send + Sync + 'static>(
        region: OwnedDeviceRegion<O>,
    ) -> Self {
        Self {
            context: Arc::clone(region.context()),
            access: Access::new(region.region(), AccessMode::Read),
            pointer: region.region().as_ptr().addr() as sys::CUdeviceptr,
            format: T::FORMAT.into(),
            channels: T::CHANNELS,
            _owner: region.into_owner() as Arc<dyn Any + Send + Sync>,
        }
    }
}

impl TextureResource {
    /// Describes a linear texture covering the entire retained region with texels of `T`.
    ///
    /// No CUDA texture object is created yet; native alignment and device limits are checked when creating the sampler.
    ///
    /// # Errors
    ///
    /// Returns an error if the region is empty, its byte length is not a multiple of `size_of::<T>()`, or the byte length does not fit a `u64`.
    pub fn linear<T: ArrayElement, O: DeviceRegionProvider + Send + Sync + 'static>(
        region: OwnedDeviceRegion<O>,
    ) -> Result<Self> {
        if region.byte_len() == 0 {
            return Err(Error::empty("texture region"));
        }

        if !region.byte_len().is_multiple_of(size_of::<T>()) {
            return Err(Error::misaligned("texture byte length", size_of::<T>()));
        }
        Ok(Self::Linear(LinearTextureResource {
            bytes: to_u64(region.byte_len(), "texture byte size")?,
            storage: TextureStorage::from_region::<T, O>(region),
        }))
    }

    /// Describes a retained two-dimensional texture with `width` texels per row, `height` rows, and a byte pitch between rows.
    ///
    /// No CUDA texture object is created yet; native alignment and device limits are checked when creating the sampler.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero, the row size or byte span overflows, the pitch is shorter than one row, the region is too small, or a dimension or the pitch does not fit a `u64`.
    pub fn pitched_2d<T: ArrayElement, O: DeviceRegionProvider + Send + Sync + 'static>(
        region: OwnedDeviceRegion<O>,
        width: usize,
        height: usize,
        pitch_bytes: usize,
    ) -> Result<Self> {
        if width == 0 {
            return Err(Error::not_positive("texture width"));
        }

        if height == 0 {
            return Err(Error::not_positive("texture height"));
        }

        let row = width
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::out_of_range("texture row bytes"))?;
        let bytes = checked_pitched_extent(height, pitch_bytes, row, "texture byte span")?;
        if row > pitch_bytes {
            return Err(Error::buffer_too_small("texture pitch", row, pitch_bytes));
        }

        if bytes > region.byte_len() {
            return Err(Error::buffer_too_small(
                "texture region",
                bytes,
                region.byte_len(),
            ));
        }

        Ok(Self::Pitched2D(PitchedTextureResource {
            storage: TextureStorage::from_region::<T, O>(region),
            width: to_u64(width, "texture width")?,
            height: to_u64(height, "texture height")?,
            pitch: to_u64(pitch_bytes, "texture pitch")?,
        }))
    }

    /// Returns the context owning this texture resource.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        match self {
            Self::Array(array) => array.context(),
            Self::MipmappedArray(array) => array.context(),
            Self::Linear(linear) => &linear.storage.context,
            Self::Pitched2D(pitched) => &pitched.storage.context,
        }
    }

    /// Creates a tracked access declaration for the retained resource using `mode`.
    ///
    /// The declaration alone does not retain storage, initialize it, or grant access permissions.
    #[must_use]
    pub fn access(&self, mode: AccessMode) -> Access {
        match self {
            Self::Array(array) => array.access(mode),
            Self::MipmappedArray(array) => array.access(mode),
            Self::Linear(linear) => {
                let mut access = linear.storage.access.clone();
                access.mode = mode;
                access
            }
            Self::Pitched2D(pitched) => {
                let mut access = pitched.storage.access.clone();
                access.mode = mode;
                access
            }
        }
    }

    /// Borrows raw resource parameters while this storage owner remains live.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUDA_RESOURCE_DESC {
        let mut raw = sys::CUDA_RESOURCE_DESC::default();
        match self {
            Self::Array(array) => {
                raw.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
                raw.res.array.hArray = array.as_raw();
            }
            Self::MipmappedArray(array) => {
                raw.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
                raw.res.mipmap.hMipmappedArray = array.as_raw();
            }
            Self::Linear(linear) => {
                raw.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_LINEAR;
                raw.res.linear = sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
                    devPtr: linear.storage.pointer,
                    format: linear.storage.format,
                    numChannels: linear.storage.channels,
                    sizeInBytes: linear.bytes,
                };
            }
            Self::Pitched2D(pitched) => {
                raw.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_PITCH2D;
                raw.res.pitch2D = sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
                    devPtr: pitched.storage.pointer,
                    format: pitched.storage.format,
                    numChannels: pitched.storage.channels,
                    width: pitched.width,
                    height: pitched.height,
                    pitchInBytes: pitched.pitch,
                };
            }
        }
        raw
    }

    fn levels(&self) -> u32 {
        match self {
            Self::MipmappedArray(array) => array.levels(),
            _ => 1,
        }
    }
}

/// Raw ownership of a texture object and the resource it samples.
///
/// Destroy `handle` before releasing `resource`, or re-adopt the value with [`TextureObject::from_raw`].
#[derive(Debug)]
pub struct RawTextureObject {
    /// Native object handle.
    pub handle: sys::CUtexObject,
    /// Backing resource the texture samples.
    pub resource: TextureResource,
}

/// Owns a native sampler and retains its resource through destruction.
#[derive(Debug)]
pub struct TextureObject {
    handle: sys::CUtexObject,
    resource: TextureResource,
}

// SAFETY: a `CUtexObject` is a plain value usable from any thread, and destruction binds the retained context first.
unsafe impl Send for TextureObject {}

// SAFETY: shared references only read the handle and the retained resource; kernels that sample it order their own access.
unsafe impl Sync for TextureObject {}

impl TextureObject {
    /// Creates a native texture sampler and retains its backing resource.
    ///
    /// The sampler does not initialize storage or authorize concurrent sampling and writes.
    /// Kernel callers must establish initialization, lifetime, and access ordering.
    ///
    /// # Errors
    ///
    /// Returns an error if the resource context cannot be bound, CUDA rejects the resource format, dimensions, alignment, device limits, or sampler options, or CUDA returns a zero handle.
    pub fn create(resource: TextureResource, options: TextureOptions) -> Result<Self> {
        let native = options.native(resource.levels());
        resource.context().bind()?;
        let mut handle = 0;
        // SAFETY: `resource` retains the storage the descriptor names in the bound context, the descriptors live through the call, and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cuTexObjectCreate(
                &raw mut handle,
                &resource.as_raw(),
                &raw const native,
                ptr::null()
            ))?;
        }

        if handle == 0 {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, resource })
    }

    /// Creates a sampler using a raw CUDA resource view descriptor.
    ///
    /// # Safety
    ///
    /// The view must describe the retained resource and satisfy CUDA's format and subresource constraints.
    ///
    /// # Errors
    ///
    /// Returns an error if the resource context cannot be bound, CUDA rejects the resource, sampler options, or view descriptor, or CUDA returns a zero handle.
    pub unsafe fn create_with_raw_view(
        resource: TextureResource,
        options: TextureOptions,
        view: &sys::CUDA_RESOURCE_VIEW_DESC,
    ) -> Result<Self> {
        resource.context().bind()?;
        let mut handle = 0;
        // SAFETY: `resource` retains the storage the descriptor names in the bound context, the caller guarantees `view` describes it, and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cuTexObjectCreate(
                &raw mut handle,
                &resource.as_raw(),
                &options.native(resource.levels()),
                view
            ))?;
        }

        if handle == 0 {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, resource })
    }

    /// Adopts a live native texture object and retains its backing resource.
    ///
    /// # Safety
    ///
    /// Transfer the live texture handle's destruction responsibility and retain its actual resource in the matching context.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is zero.
    pub unsafe fn from_raw(raw: RawTextureObject) -> Result<Self> {
        if raw.handle == 0 {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            resource: raw.resource,
        })
    }

    /// Returns the borrowed native texture handle without retaining another resource owner.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUtexObject {
        self.handle
    }

    /// Returns the context owning the retained texture resource.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.resource.context()
    }

    /// Borrows the backing texture resource.
    #[must_use]
    pub fn resource(&self) -> &TextureResource {
        &self.resource
    }

    /// Creates a resource ledger retaining this sampler and declaring read access to its backing storage.
    ///
    /// This preserves resource readiness metadata without establishing initialization or granting sampling permissions.
    #[must_use]
    pub fn resources(self: &Arc<Self>) -> Resources {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self));
        resources.declare_access(self.resource.access(AccessMode::Read));
        resources
    }

    /// Transfers the handle together with its resource ownership.
    #[must_use]
    pub fn into_raw(mut self) -> RawTextureObject {
        let handle = mem::replace(&mut self.handle, 0);
        let this = mem::ManuallyDrop::new(self);
        RawTextureObject {
            handle,
            // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
            resource: unsafe { ptr::read(&raw const this.resource) },
        }
    }
}

impl Drop for TextureObject {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }

        if let Err(error) = self.context().bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cuda texture object",
                &error,
            );
            return;
        }

        // SAFETY: this wrapper owns the nonzero texture object, its context is bound, and `self.resource` outlives the destroy.
        if let Err(error) = unsafe { try_ffi!(sys::cuTexObjectDestroy(self.handle)) } {
            report_cleanup_failure("destroy", "cuda texture object", &error);
        }
    }
}

/// Raw ownership of a surface object and the array it refers to.
///
/// Destroy `handle` before releasing `allocation`, or re-adopt the value with [`SurfaceObject::from_raw`].
#[derive(Debug)]
pub struct RawSurfaceObject {
    /// Native object handle.
    pub handle: sys::CUsurfObject,
    /// Array allocation the surface refers to.
    pub allocation: Arc<ArrayAllocation>,
}

/// Owns a native surface retaining its writable array allocation.
#[derive(Debug)]
pub struct SurfaceObject {
    handle: sys::CUsurfObject,
    allocation: Arc<ArrayAllocation>,
}

// SAFETY: a `CUsurfObject` is a plain value usable from any thread, and destruction binds the retained context first.
unsafe impl Send for SurfaceObject {}

// SAFETY: shared references only read the handle and the retained array; kernels that use the surface order their own access.
unsafe impl Sync for SurfaceObject {}

impl SurfaceObject {
    /// Creates a CUDA surface object retaining `allocation`.
    ///
    /// The array must have been allocated with [`ArrayFlags::SURFACE_LOAD_STORE`].
    /// Surface writes and reads still require caller-established ordering and access permissions.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation lacks [`ArrayFlags::SURFACE_LOAD_STORE`], its context cannot be bound, CUDA rejects surface creation, or CUDA returns a zero handle.
    pub fn create(allocation: Arc<ArrayAllocation>) -> Result<Self> {
        if !allocation.flags().contains(ArrayFlags::SURFACE_LOAD_STORE) {
            return Err(Error::out_of_range("array flags"));
        }

        allocation.context().bind()?;
        let mut raw = sys::CUDA_RESOURCE_DESC {
            resType: sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY,
            ..Default::default()
        };
        raw.res.array.hArray = allocation.as_raw();
        let mut handle = 0;
        // SAFETY: `raw` names the array `allocation` retains in the bound context, and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cuSurfObjectCreate(&raw mut handle, &raw const raw))?;
        }

        if handle == 0 {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, allocation })
    }

    /// Adopts a live native surface object and retains its backing array.
    ///
    /// # Safety
    ///
    /// Transfer destruction responsibility for a live surface referring to this exact retained array.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is zero.
    pub unsafe fn from_raw(raw: RawSurfaceObject) -> Result<Self> {
        if raw.handle == 0 {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            allocation: raw.allocation,
        })
    }

    /// Returns the borrowed native surface handle without retaining another allocation owner.
    #[must_use]
    pub fn as_raw(&self) -> sys::CUsurfObject {
        self.handle
    }

    /// Returns the context owning the retained surface allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.allocation.context()
    }

    /// Borrows the array allocation retained by this surface.
    #[must_use]
    pub fn allocation(&self) -> &Arc<ArrayAllocation> {
        &self.allocation
    }

    /// Creates a resource ledger retaining this surface and declaring `mode` access to its backing array.
    ///
    /// The declaration does not initialize storage or grant access permissions.
    #[must_use]
    pub fn resources(self: &Arc<Self>, mode: AccessMode) -> Resources {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self));
        resources.declare_access(self.allocation.access(mode));
        resources
    }

    /// Transfers the native surface handle and its retained array allocation to the caller.
    ///
    /// The caller must destroy the surface and keep its allocation alive through every use of the handle.
    #[must_use]
    pub fn into_raw(mut self) -> RawSurfaceObject {
        let handle = mem::replace(&mut self.handle, 0);
        let this = mem::ManuallyDrop::new(self);
        RawSurfaceObject {
            handle,
            // SAFETY: the owner is wrapped in `ManuallyDrop`, so this field moves out exactly once.
            allocation: unsafe { ptr::read(&raw const this.allocation) },
        }
    }
}

impl Drop for SurfaceObject {
    fn drop(&mut self) {
        if self.handle == 0 {
            return;
        }

        if let Err(error) = self.context().bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cuda surface object",
                &error,
            );
            return;
        }

        // SAFETY: this wrapper owns the nonzero surface object, its context is bound, and `self.allocation` outlives the destroy.
        if let Err(error) = unsafe { try_ffi!(sys::cuSurfObjectDestroy(self.handle)) } {
            report_cleanup_failure("destroy", "cuda surface object", &error);
        }
    }
}
