//! CUDA handle, pointer, scalar-storage, and configuration types.

#![expect(
    deprecated,
    non_camel_case_types,
    reason = "wraps deprecated upstream API values that remain part of the supported surface; names mirror the upstream C API"
)]

mod emulation;
mod function;
mod graphics;
mod memory;
mod pointer;
mod storage;

pub use half::{bf16, f16};
pub use num_complex::{Complex, Complex32, Complex64};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda_sys::library_types;

pub use self::{
    emulation::{EmulationMantissaControl, EmulationSpecialValuesSupport, EmulationStrategy},
    function::{
        ClusterSchedulingPolicy, FunctionAttribute, FunctionCache, SharedMemoryCarveout,
        SharedMemoryConfig, SharedMemoryMode,
    },
    graphics::{CubemapFace, GraphicsMapResourceFlags, GraphicsRegisterFlags, ResourceType},
    memory::{AccessProperty, MemoryAdvise, MemoryRangeAttribute, MemoryType, PointerAttribute},
    pointer::{DeviceFunction, DevicePtr, HostFunction},
    storage::{f4e2m1, f6e2m3, f6e3m2, f8e4m3, f8e5m2, f8ue5m3, f8ue8m0},
};

singe_core::native_enum! {
    /// Component of a numerical library version queried through its property API.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum LibraryProperty: library_types::libraryPropertyType {
        /// Major library version.
        Major = MAJOR_VERSION,
        /// Minor library version.
        Minor = MINOR_VERSION,
        /// Patch level of the library version.
        Patch = PATCH_LEVEL,
    }
}

impl_enum_display!(LibraryProperty, {
    Self::Major => "MAJOR_VERSION",
    Self::Minor => "MINOR_VERSION",
    Self::Patch => "PATCH_LEVEL",
});
