use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::driver;

bitflags::bitflags! {
    /// Flags to register a graphics resource
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphicsRegisterFlags: u32 {
        /// Default registration behavior without additional access hints.
        const NONE = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_NONE.0;
        /// Hints that CUDA will only read the registered resource.
        const READ_ONLY = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY.0;
        /// Hints that CUDA will overwrite the registered resource without reading its previous contents.
        const WRITE_DISCARD = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD.0;
        /// Allows binding the registered image to a CUDA surface for load and store operations.
        const SURFACE_LDST = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST.0;
        /// Allows texture-gather operations on the registered image.
        const TEXTURE_GATHER = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER.0;
    }
}

impl Display for GraphicsRegisterFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    /// Flags for mapping and unmapping interop resources
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphicsMapResourceFlags: u32 {
        /// Specifies no hints about how this resource will be used.
        /// It is therefore assumed that this resource will be read from and written to by CUDA kernels.
        /// This is the default value.
        const NONE = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE.0;
        /// Hints that CUDA will only read the mapped resource.
        const READ_ONLY = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY.0;
        /// Hints that CUDA will overwrite the mapped resource without reading its previous contents.
        const WRITE_DISCARD = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD.0;
    }
}

impl Display for GraphicsMapResourceFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// Array indices for cube faces
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CubemapFace: driver::CUarray_cubemap_face {
        /// Positive X face of cubemap.
        PositiveX = CU_CUBEMAP_FACE_POSITIVE_X,
        /// Negative X face of cubemap.
        NegativeX = CU_CUBEMAP_FACE_NEGATIVE_X,
        /// Positive Y face of cubemap.
        PositiveY = CU_CUBEMAP_FACE_POSITIVE_Y,
        /// Negative Y face of cubemap.
        NegativeY = CU_CUBEMAP_FACE_NEGATIVE_Y,
        /// Positive Z face of cubemap.
        PositiveZ = CU_CUBEMAP_FACE_POSITIVE_Z,
        /// Negative Z face of cubemap.
        NegativeZ = CU_CUBEMAP_FACE_NEGATIVE_Z,
    }
}

impl_enum_display!(CubemapFace, {
    Self::PositiveX => "CU_CUBEMAP_FACE_POSITIVE_X",
    Self::NegativeX => "CU_CUBEMAP_FACE_NEGATIVE_X",
    Self::PositiveY => "CU_CUBEMAP_FACE_POSITIVE_Y",
    Self::NegativeY => "CU_CUBEMAP_FACE_NEGATIVE_Y",
    Self::PositiveZ => "CU_CUBEMAP_FACE_POSITIVE_Z",
    Self::NegativeZ => "CU_CUBEMAP_FACE_NEGATIVE_Z",
});

singe_core::native_enum! {
    /// Resource types
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ResourceType: driver::CUresourcetype {
        /// Array resource.
        Array = CU_RESOURCE_TYPE_ARRAY,
        /// Mipmapped array resource.
        MipmappedArray = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
        /// Linear resource.
        Linear = CU_RESOURCE_TYPE_LINEAR,
        /// Pitch 2D resource.
        Pitch2d = CU_RESOURCE_TYPE_PITCH2D,
    }
}

impl_enum_display!(ResourceType, {
    Self::Array => "CU_RESOURCE_TYPE_ARRAY",
    Self::MipmappedArray => "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY",
    Self::Linear => "CU_RESOURCE_TYPE_LINEAR",
    Self::Pitch2d => "CU_RESOURCE_TYPE_PITCH2D",
});
