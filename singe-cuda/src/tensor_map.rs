//! Transferable TMA descriptors and independently retained backing allocations.

use singe_cuda_sys::driver as sys;

use crate::{
    context::Context,
    error::{Error, Result},
    submission::{AccessMode, Resources},
    try_ffi,
    types::DevicePtr,
    utility::{driver_enum, to_u32},
    view::{DeviceRegionProvider, DeviceRepr, NoUninitDeviceRepr, OwnedDeviceRegion},
};

driver_enum! {
    /// Tensor map data type
    pub enum TensorMapDataType: sys::CUtensorMapDataType {
        /// Unsigned 8-bit integer elements.
        U8 = CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /// Unsigned 16-bit integer elements.
        U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16,
        /// Unsigned 32-bit integer elements.
        U32 = CU_TENSOR_MAP_DATA_TYPE_UINT32,
        /// Signed 32-bit integer elements.
        I32 = CU_TENSOR_MAP_DATA_TYPE_INT32,
        /// Unsigned 64-bit integer elements.
        U64 = CU_TENSOR_MAP_DATA_TYPE_UINT64,
        /// Signed 64-bit integer elements.
        I64 = CU_TENSOR_MAP_DATA_TYPE_INT64,
        /// 16-bit floating-point elements.
        F16 = CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        /// 32-bit floating-point elements.
        F32 = CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        /// 64-bit floating-point elements.
        F64 = CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        /// 16-bit brain floating-point elements.
        Bf16 = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        /// 32-bit floating-point elements with subnormal values flushed to zero.
        F32Ftz = CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
        /// TensorFloat-32 elements.
        Tf32 = CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
        /// TensorFloat-32 elements with subnormal values flushed to zero.
        Tf32Ftz = CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
        /// Groups of sixteen unsigned 4-bit values packed into eight bytes, with no gaps between groups.
        U4x16Align8 = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        /// Groups of sixteen unsigned 4-bit values in 16-byte-aligned storage, with eight padding bytes after each eight-byte packed group.
        U4x16Align16 = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        /// Groups of sixteen unsigned 6-bit values in 16-byte-aligned storage, with four padding bytes after each twelve-byte packed group.
        U6x16Align16 = CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B,
    }
}

driver_enum! {
    /// Tensor map interleave layout type
    #[derive(Default)]
    pub enum Interleave: sys::CUtensorMapInterleave {
        /// No channel interleaving.
        #[default]
        None = CU_TENSOR_MAP_INTERLEAVE_NONE,
        /// Interleaves channels in 16-byte groups, such as eight 16-bit channels per group.
        Bytes16 = CU_TENSOR_MAP_INTERLEAVE_16B,
        /// Interleaves channels in 32-byte groups, such as sixteen 16-bit channels per group.
        Bytes32 = CU_TENSOR_MAP_INTERLEAVE_32B,
    }
}

driver_enum! {
    /// Tensor map swizzling mode of shared memory banks
    #[derive(Default)]
    pub enum Swizzle: sys::CUtensorMapSwizzle {
        /// No shared-memory swizzling.
        #[default]
        None = CU_TENSOR_MAP_SWIZZLE_NONE,
        /// Swizzles 16-byte chunks within a 32-byte span.
        Bytes32 = CU_TENSOR_MAP_SWIZZLE_32B,
        /// Swizzles 16-byte chunks within a 64-byte span.
        Bytes64 = CU_TENSOR_MAP_SWIZZLE_64B,
        /// Swizzles 16-byte chunks within a 128-byte span.
        Bytes128 = CU_TENSOR_MAP_SWIZZLE_128B,
        /// Swizzles 32-byte chunks within a 128-byte span.
        Bytes128Atom32 = CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
        /// Swizzles 32-byte chunks within a 128-byte span and exchanges the lower and upper eight bytes of each 16-byte unit on alternate rows.
        Bytes128Atom32Flip8 = CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
        /// Swizzles 64-byte chunks within a 128-byte span.
        Bytes128Atom64 = CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
    }
}

driver_enum! {
    /// Tensor map L2 promotion type
    #[derive(Default)]
    pub enum L2Promotion: sys::CUtensorMapL2promotion {
        /// No L2 fetch-size promotion.
        #[default]
        None = CU_TENSOR_MAP_L2_PROMOTION_NONE,
        /// Fills L2 requests from DRAM at 64-byte granularity.
        Bytes64 = CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        /// Fills L2 requests from DRAM at 128-byte granularity.
        Bytes128 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        /// Fills L2 requests from DRAM at 256-byte granularity.
        Bytes256 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    }
}

driver_enum! {
    /// Tensor map out-of-bounds fill type
    #[derive(Default)]
    pub enum OutOfBoundsFill: sys::CUtensorMapFloatOOBfill {
        /// Fills out-of-bounds elements with zero.
        #[default]
        Zero = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        /// Fills out-of-bounds elements with the special NaN value that requests zero contribution to FMA. Only floating-point tensor data types support this mode.
        NanRequestZeroFma = CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    }
}

driver_enum! {
    /// Tensor-map image-to-column loading width.
    #[derive(Default)]
    pub enum Im2colWideMode: sys::CUtensorMapIm2ColWideMode {
        /// Loads along the W dimension using the element count in `pixelsPerColumn`.
        #[default]
        Width = CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
        /// Uses native W128 loading along the W dimension; `pixelsPerColumn` is ignored.
        Width128 = CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128,
    }
}

driver_enum! {
    /// Tensor map swizzling mode of shared memory banks
    #[derive(Default)]
    pub enum Im2colWideSwizzle: sys::CUtensorMapSwizzle {
        /// Swizzles 16-byte chunks within a 64-byte span.
        #[default]
        Bytes64 = CU_TENSOR_MAP_SWIZZLE_64B,
        /// Swizzles 16-byte chunks within a 128-byte span.
        Bytes128 = CU_TENSOR_MAP_SWIZZLE_128B,
        /// Swizzles 32-byte chunks within a 128-byte span.
        Bytes128Atom32 = CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
    }
}

/// Interleaving, shared-memory swizzling, L2 promotion, and out-of-bounds behavior for tensor-map encoding.
///
/// `S` restricts the swizzle choices accepted by the selected encoding mode. Defaults use no interleaving, the default swizzle for `S`, no L2 promotion, and zero fill.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorMapOptions<S = Swizzle> {
    /// Channel interleave layout of the global tensor.
    pub interleave: Interleave,
    /// Shared-memory bank swizzle pattern supported by the encoding mode.
    pub swizzle: S,
    /// L2 fetch-size promotion used for tensor accesses.
    pub l2_promotion: L2Promotion,
    /// Value used for out-of-bounds tensor elements.
    pub out_of_bounds_fill: OutOfBoundsFill,
}

/// Fastest-varying dimension first; byte strides omit the implicit dimension-zero stride.
#[derive(Debug, Clone, Copy)]
pub struct TensorMapLayout<'a> {
    /// Scalar or packed element format of the global tensor.
    pub data_type: TensorMapDataType,
    /// Element count along each axis, fastest-varying first. Each must lie in `1..=2^32`; supported ranks depend on the encoding mode.
    pub dimensions: &'a [u64],
    /// Byte strides for axes one through rank minus one. The dimension-zero stride is implicit. Each supplied stride must be positive, a multiple of 16, and less than `2^40`.
    pub byte_strides: &'a [u64],
    /// Traversal step along every axis, with one entry per dimension. Each must lie in `1..=8`; CUDA ignores the first step when interleaving is disabled.
    pub element_strides: &'a [u32],
}

impl TensorMapLayout<'_> {
    fn validate(&self, minimum_rank: usize) -> Result<u32> {
        let rank = self.dimensions.len();
        if !(minimum_rank..=5).contains(&rank)
            || self.byte_strides.len() != rank - 1
            || self.element_strides.len() != rank
        {
            return Err(Error::InvalidMemoryAccess);
        }
        if self
            .dimensions
            .iter()
            .any(|&dimension| dimension == 0 || dimension > 1 << 32)
            || self
                .byte_strides
                .iter()
                .any(|&stride| stride == 0 || !stride.is_multiple_of(16) || stride >= 1 << 40)
            || self
                .element_strides
                .iter()
                .any(|&stride| !(1..=8).contains(&stride))
        {
            return Err(Error::InvalidMemoryAccess);
        }
        to_u32(rank, "tensor map rank")
    }

    fn required_bytes(&self) -> Result<u64> {
        self.validate(1)?;
        let width = self.dimensions[0];
        let row = match self.data_type {
            TensorMapDataType::U4x16Align8 => Some(width.div_ceil(2)),
            TensorMapDataType::U4x16Align16 | TensorMapDataType::U6x16Align16 => {
                width.div_ceil(16).checked_mul(16)
            }
            TensorMapDataType::U8 => Some(width),
            TensorMapDataType::U16 | TensorMapDataType::F16 | TensorMapDataType::Bf16 => {
                width.checked_mul(2)
            }
            TensorMapDataType::U32
            | TensorMapDataType::I32
            | TensorMapDataType::F32
            | TensorMapDataType::F32Ftz
            | TensorMapDataType::Tf32
            | TensorMapDataType::Tf32Ftz => width.checked_mul(4),
            TensorMapDataType::U64 | TensorMapDataType::I64 | TensorMapDataType::F64 => {
                width.checked_mul(8)
            }
        }
        .ok_or(Error::InvalidMemoryAllocationRequest)?;
        self.dimensions[1..].iter().zip(self.byte_strides).try_fold(
            row,
            |bytes, (&dimension, &stride)| {
                (dimension - 1)
                    .checked_mul(stride)
                    .and_then(|offset| bytes.checked_add(offset))
                    .ok_or(Error::InvalidMemoryAllocationRequest)
            },
        )
    }
}

/// Spatial pixel-box corners and channel counts for image-to-column tensor-map encoding.
///
/// Each corner slice must have rank minus two entries.
#[derive(Debug, Clone, Copy)]
pub struct Im2colBox<'a> {
    /// Spatial coordinates of the lower pixel-box corner.
    pub lower_corner: &'a [i32],
    /// Spatial coordinates of the upper pixel-box corner.
    pub upper_corner: &'a [i32],
    /// Number of channels per pixel.
    pub channels_per_pixel: u32,
    /// Number of pixels per output column.
    pub pixels_per_column: u32,
}

/// Width-only pixel-box bounds and channel counts for wide image-to-column encoding.
///
/// The box has a size of one along its depth and height axes. Native wide encoding requires compute capability 10.0 or higher.
#[derive(Debug, Clone, Copy)]
pub struct Im2colWideBox {
    /// Width coordinate offset of the left box corner.
    pub lower_width: i32,
    /// Width coordinate offset of the right box corner.
    pub upper_width: i32,
    /// Number of channels per pixel.
    pub channels_per_pixel: u32,
    /// Number of pixels per output column; ignored in W128 mode.
    pub pixels_per_column: u32,
    /// Native W or W128 width-loading mode.
    pub mode: Im2colWideMode,
}

/// CUDA-aligned descriptor bytes suitable for kernel arguments and device transfer.
/// This value owns no backing memory and makes no initialization or access claim.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct TensorMapDescriptor(sys::CUtensorMap);

// SAFETY: the native descriptor is exactly 128 initialized bytes with no host ownership or padding.
unsafe impl DeviceRepr for TensorMapDescriptor {}

unsafe impl NoUninitDeviceRepr for TensorMapDescriptor {}

impl TensorMapDescriptor {
    /// Borrows the native descriptor bytes without retaining backing memory.
    #[must_use]
    pub fn as_raw(&self) -> &sys::CUtensorMap {
        &self.0
    }

    /// Returns the native descriptor bytes by value without retaining backing memory.
    #[must_use]
    pub fn into_raw(self) -> sys::CUtensorMap {
        self.0
    }

    /// Wraps initialized native tensor-map descriptor bytes without retaining their backing memory.
    ///
    /// # Safety
    /// The value must be a successfully encoded CUDA tensor map with initialized descriptor bytes.
    #[must_use]
    pub unsafe fn from_raw(raw: sys::CUtensorMap) -> Self {
        Self(raw)
    }

    /// Encodes a tiled layout at a raw global address.
    ///
    /// # Safety
    /// The address must satisfy CUDA's global-memory, alignment and layout restrictions.
    /// Retain backing memory and establish initialization and ordering before device use.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` for invalid rank, slice lengths, dimension sizes, byte strides, or traversal strides. Also propagates context binding failures and native encoding errors for unsupported layouts, options, addresses, or device capabilities. The box must have one size per tensor dimension, each in `1..=256`.
    pub unsafe fn encode_tiled(
        context: &Context,
        address: DevicePtr,
        layout: TensorMapLayout<'_>,
        box_dimensions: &[u32],
        options: TensorMapOptions,
    ) -> Result<Self> {
        let rank = layout.validate(1)?;
        if box_dimensions.len() != rank as usize
            || box_dimensions
                .iter()
                .any(|&dimension| !(1..=256).contains(&dimension))
        {
            return Err(Error::InvalidMemoryAccess);
        }
        context.bind()?;
        let mut raw = sys::CUtensorMap::default();
        // SAFETY: `cuTensorMapEncodeTiled` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuTensorMapEncodeTiled(
                &raw mut raw,
                layout.data_type.into(),
                rank,
                address.as_raw().cast(),
                layout.dimensions.as_ptr(),
                layout.byte_strides.as_ptr(),
                box_dimensions.as_ptr(),
                layout.element_strides.as_ptr(),
                options.interleave.into(),
                options.swizzle.into(),
                options.l2_promotion.into(),
                options.out_of_bounds_fill.into()
            ))?;
        }
        Ok(Self(raw))
    }

    /// Encodes an im2col layout with rank-minus-two spatial box corners.
    ///
    /// # Safety
    /// Uphold [`Self::encode_tiled`]'s memory and lifetime contract for this layout.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` for invalid rank, slice lengths, dimension sizes, byte strides, or traversal strides. Also propagates context binding failures and native encoding errors for unsupported layouts, options, addresses, or device capabilities. The tensor rank must be three through five, and both corner slices must have rank minus two entries.
    pub unsafe fn encode_im2col(
        context: &Context,
        address: DevicePtr,
        layout: TensorMapLayout<'_>,
        pixel_box: Im2colBox<'_>,
        options: TensorMapOptions,
    ) -> Result<Self> {
        let rank = layout.validate(3)?;
        if pixel_box.lower_corner.len() != rank as usize - 2
            || pixel_box.upper_corner.len() != rank as usize - 2
        {
            return Err(Error::InvalidMemoryAccess);
        }
        context.bind()?;
        let mut raw = sys::CUtensorMap::default();
        // SAFETY: `cuTensorMapEncodeIm2col` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuTensorMapEncodeIm2col(
                &raw mut raw,
                layout.data_type.into(),
                rank,
                address.as_raw().cast(),
                layout.dimensions.as_ptr(),
                layout.byte_strides.as_ptr(),
                pixel_box.lower_corner.as_ptr(),
                pixel_box.upper_corner.as_ptr(),
                pixel_box.channels_per_pixel,
                pixel_box.pixels_per_column,
                layout.element_strides.as_ptr(),
                options.interleave.into(),
                options.swizzle.into(),
                options.l2_promotion.into(),
                options.out_of_bounds_fill.into()
            ))?;
        }
        Ok(Self(raw))
    }

    /// Encodes wide-im2col layouts on devices supporting the native mode.
    ///
    /// # Safety
    /// Uphold [`Self::encode_tiled`]'s memory and lifetime contract for this layout.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` for invalid rank, slice lengths, dimension sizes, byte strides, or traversal strides. Also propagates context binding failures and native encoding errors for unsupported layouts, options, addresses, or device capabilities. The tensor rank must be three through five; CUDA additionally validates the width box and wide-loading mode.
    pub unsafe fn encode_im2col_wide(
        context: &Context,
        address: DevicePtr,
        layout: TensorMapLayout<'_>,
        pixel_box: Im2colWideBox,
        options: TensorMapOptions<Im2colWideSwizzle>,
    ) -> Result<Self> {
        let rank = layout.validate(3)?;
        context.bind()?;
        let mut raw = sys::CUtensorMap::default();
        // SAFETY: `cuTensorMapEncodeIm2colWide` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cuTensorMapEncodeIm2colWide(
                &raw mut raw,
                layout.data_type.into(),
                rank,
                address.as_raw().cast(),
                layout.dimensions.as_ptr(),
                layout.byte_strides.as_ptr(),
                pixel_box.lower_width,
                pixel_box.upper_width,
                pixel_box.channels_per_pixel,
                pixel_box.pixels_per_column,
                layout.element_strides.as_ptr(),
                options.interleave.into(),
                pixel_box.mode.into(),
                options.swizzle.into(),
                options.l2_promotion.into(),
                options.out_of_bounds_fill.into()
            ))?;
        }
        Ok(Self(raw))
    }

    /// Replaces the global base while preserving the encoded layout.
    ///
    /// # Safety
    /// The new storage must satisfy the original descriptor's size, alignment and memory restrictions.
    /// Existing descriptor copies and in-flight users are unaffected and retain their own lifetime obligations.
    ///
    /// # Errors
    ///
    /// Returns an error if `context` cannot be bound or CUDA rejects the replacement address for the descriptor's encoded layout.
    pub unsafe fn replace_address(&mut self, context: &Context, address: DevicePtr) -> Result<()> {
        context.bind()?;
        // SAFETY: `cuTensorMapReplaceAddress` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cuTensorMapReplaceAddress(
                &raw mut self.0,
                address.as_raw().cast()
            ))
        }
    }
}

/// Host ownership accompanying a transferable tensor-map descriptor.
#[derive(Debug)]
pub struct RetainedTensorMap<O: DeviceRegionProvider + Send + Sync + 'static> {
    descriptor: TensorMapDescriptor,
    region: OwnedDeviceRegion<O>,
    required_bytes: u64,
}

impl<O: DeviceRegionProvider + Send + Sync + 'static> RetainedTensorMap<O> {
    fn encode(
        region: OwnedDeviceRegion<O>,
        layout: TensorMapLayout<'_>,
        encode: impl FnOnce(&Context, DevicePtr) -> Result<TensorMapDescriptor>,
    ) -> Result<Self> {
        let required_bytes = layout.required_bytes()?;
        if required_bytes > region.byte_len() as u64 {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the address belongs to a live device allocation that stays borrowed for the operation that uses it.
        let descriptor = encode(region.context(), unsafe {
            DevicePtr::from_raw(region.region().as_ptr().cast())
        })?;
        Ok(Self {
            descriptor,
            region,
            required_bytes,
        })
    }

    /// Encodes a tiled tensor map while retaining its backing device region.
    ///
    /// Checks that the region covers the byte span described by `layout`. Retention does not establish initialization or order later device accesses.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` for an invalid layout or insufficient backing region, or `Error::InvalidMemoryAllocationRequest` if the required byte-span calculation overflows. Also propagates encoding and context binding errors from [`TensorMapDescriptor::encode_tiled`].
    pub fn tiled(
        region: OwnedDeviceRegion<O>,
        layout: TensorMapLayout<'_>,
        box_dimensions: &[u32],
        options: TensorMapOptions,
    ) -> Result<Self> {
        // SAFETY: the region was validated against `layout`, and the descriptor retains it.
        Self::encode(region, layout, |context, address| unsafe {
            TensorMapDescriptor::encode_tiled(context, address, layout, box_dimensions, options)
        })
    }

    /// Encodes a image-to-column tensor map while retaining its backing device region.
    ///
    /// Checks that the region covers the byte span described by `layout`. Retention does not establish initialization or order later device accesses.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` for an invalid layout or insufficient backing region, or `Error::InvalidMemoryAllocationRequest` if the required byte-span calculation overflows. Also propagates encoding and context binding errors from [`TensorMapDescriptor::encode_im2col`].
    pub fn im2col(
        region: OwnedDeviceRegion<O>,
        layout: TensorMapLayout<'_>,
        pixel_box: Im2colBox<'_>,
        options: TensorMapOptions,
    ) -> Result<Self> {
        // SAFETY: the region was validated against `layout`, and the descriptor retains it.
        Self::encode(region, layout, |context, address| unsafe {
            TensorMapDescriptor::encode_im2col(context, address, layout, pixel_box, options)
        })
    }

    /// Encodes a wide image-to-column tensor map while retaining its backing device region.
    ///
    /// Checks that the region covers the byte span described by `layout`. Retention does not establish initialization or order later device accesses.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidMemoryAccess` for an invalid layout or insufficient backing region, or `Error::InvalidMemoryAllocationRequest` if the required byte-span calculation overflows. Also propagates encoding and context binding errors from [`TensorMapDescriptor::encode_im2col_wide`].
    pub fn im2col_wide(
        region: OwnedDeviceRegion<O>,
        layout: TensorMapLayout<'_>,
        pixel_box: Im2colWideBox,
        options: TensorMapOptions<Im2colWideSwizzle>,
    ) -> Result<Self> {
        // SAFETY: the region was validated against `layout`, and the descriptor retains it.
        Self::encode(region, layout, |context, address| unsafe {
            TensorMapDescriptor::encode_im2col_wide(context, address, layout, pixel_box, options)
        })
    }

    /// Borrows the transferable descriptor.
    ///
    /// Copies of its bytes do not retain this object or the backing region.
    #[must_use]
    pub fn descriptor(&self) -> &TensorMapDescriptor {
        &self.descriptor
    }

    /// Borrows the owned backing region retained alongside the descriptor.
    #[must_use]
    pub fn region(&self) -> &OwnedDeviceRegion<O> {
        &self.region
    }

    /// Creates a resource ledger retaining the backing region and declaring `mode` access to it.
    ///
    /// The declaration preserves region readiness but does not initialize storage or grant access permissions.
    #[must_use]
    pub fn resources(&self, mode: AccessMode) -> Resources {
        let mut resources = Resources::new();
        resources.retain_access(self.region.clone(), mode);
        resources
    }

    /// Replaces the descriptor's base address and retained backing region, preserving its encoded layout.
    ///
    /// Previously copied descriptors still point to the old region. Keep that region alive for any executions that use those copies.
    ///
    /// # Errors
    ///
    /// Returns `Error::MemoryContextMismatch` if the new region belongs to another context, or `Error::InvalidMemoryAccess` if it is too short for the encoded layout. Also propagates context binding and native address-replacement errors. The retained region changes only after replacement succeeds.
    pub fn replace_region(&mut self, region: OwnedDeviceRegion<O>) -> Result<()> {
        if region.context() != self.region.context() {
            return Err(Error::MemoryContextMismatch);
        }
        if self.required_bytes > region.byte_len() as u64 {
            return Err(Error::InvalidMemoryAccess);
        }
        // SAFETY: the new region was validated against the descriptor layout and is retained by `self`.
        unsafe {
            self.descriptor.replace_address(
                region.context(),
                DevicePtr::from_raw(region.region().as_ptr().cast()),
            )?;
        }
        self.region = region;
        Ok(())
    }
}
