#![allow(deprecated, non_camel_case_types)]

use std::{
    fmt::{self, Display, Formatter},
    ptr,
};

pub use num_complex::{Complex, Complex32, Complex64};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_cuda_sys::{driver, library_types};

pub use half::{bf16, f16};

use singe_core::{impl_enum_conversion, impl_enum_display};

use crate::view::{DeviceRepr, ZeroableDeviceRepr};

macro_rules! impl_float_storage {
    ($name:ident, $bits:ty) => {
        #[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(pub $bits);

        impl $name {
            pub const fn from_bits(bits: $bits) -> Self {
                Self(bits)
            }

            pub const fn to_bits(self) -> $bits {
                self.0
            }
        }
    };
}

impl_float_storage!(f8e4m3, u8);
impl_float_storage!(f8e5m2, u8);
impl_float_storage!(f8ue8m0, u8);
impl_float_storage!(f6e2m3, u8);
impl_float_storage!(f6e3m2, u8);
impl_float_storage!(f4e2m1, u8);

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
/// CUDA host callback function pointer.
///
/// This is a borrowed function pointer value, not an owned CUDA resource.
pub struct HostFunction(driver::CUhostFn);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// CUDA device function handle.
///
/// CUDA owns the underlying function as part of a module, library, or runtime
/// registration. This wrapper is a copyable handle value and does not unload or
/// destroy the function.
pub struct DeviceFunction(driver::CUfunction);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct DevicePtr(*mut ());

unsafe impl DeviceRepr for DevicePtr {}
unsafe impl ZeroableDeviceRepr for DevicePtr {}

impl HostFunction {
    /// Wraps a raw CUDA host callback function pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a callback function pointer with the ABI and lifetime
    /// required by CUDA for every operation that uses the returned handle.
    pub const unsafe fn new(raw: driver::CUhostFn) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA host callback function pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a callback function pointer with the ABI and lifetime
    /// required by CUDA for every operation that uses the returned handle.
    pub const unsafe fn from_raw(raw: driver::CUhostFn) -> Self {
        unsafe { Self::new(raw) }
    }

    pub const fn as_raw(self) -> driver::CUhostFn {
        self.0
    }
}

impl DeviceFunction {
    /// Wraps a raw CUDA device function handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid CUDA function handle whose owning module,
    /// library, or runtime registration remains loaded for every operation that
    /// uses the returned handle.
    pub const unsafe fn new(raw: driver::CUfunction) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA device function handle.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid CUDA function handle whose owning module,
    /// library, or runtime registration remains loaded for every operation that
    /// uses the returned handle.
    pub const unsafe fn from_raw(raw: driver::CUfunction) -> Self {
        unsafe { Self::new(raw) }
    }

    pub const fn as_raw(self) -> driver::CUfunction {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl DevicePtr {
    pub const fn null() -> Self {
        Self(ptr::null_mut())
    }

    /// Wraps a raw CUDA device pointer value.
    ///
    /// # Safety
    ///
    /// `raw` must be either null or a pointer value that is valid for the CUDA
    /// operation it is passed to. This wrapper does not prove allocation
    /// ownership, size, lifetime, context association, or access permissions.
    pub const unsafe fn new(raw: *mut ()) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA device pointer value.
    ///
    /// # Safety
    ///
    /// `raw` must be either null or a pointer value that is valid for the CUDA
    /// operation it is passed to. This wrapper does not prove allocation
    /// ownership, size, lifetime, context association, or access permissions.
    pub const unsafe fn from_raw(raw: *mut ()) -> Self {
        unsafe { Self::new(raw.cast()) }
    }

    pub const fn as_raw(self) -> *mut () {
        self.0.cast()
    }

    pub const fn as_ptr(self) -> *mut () {
        self.0
    }

    pub const fn as_const_ptr(self) -> *const () {
        self.0.cast_const()
    }

    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }

    pub const fn cast<T>(self) -> *mut T {
        self.0.cast()
    }
}

impl From<HostFunction> for driver::CUhostFn {
    fn from(value: HostFunction) -> Self {
        value.as_raw()
    }
}

impl From<DeviceFunction> for driver::CUfunction {
    fn from(value: DeviceFunction) -> Self {
        value.as_raw()
    }
}

impl From<DevicePtr> for *mut () {
    fn from(value: DevicePtr) -> Self {
        value.as_raw()
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphicsRegisterFlags: u32 {
        const NONE = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_NONE as _;
        const READ_ONLY = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY as _;
        const WRITE_DISCARD = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD as _;
        const SURFACE_LDST = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST as _;
        const TEXTURE_GATHER = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER as _;
    }
}

impl Display for GraphicsRegisterFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphicsMapResourceFlags: u32 {
        const NONE = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE as _;
        const READ_ONLY = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY as _;
        const WRITE_DISCARD = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD as _;
    }
}

impl Display for GraphicsMapResourceFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LibraryProperty {
    Major = library_types::libraryPropertyType::MAJOR_VERSION as _,
    Minor = library_types::libraryPropertyType::MINOR_VERSION as _,
    Patch = library_types::libraryPropertyType::PATCH_LEVEL as _,
}

impl_enum_conversion!(library_types::libraryPropertyType, LibraryProperty);

impl_enum_display!(LibraryProperty, {
    Self::Major => "MAJOR_VERSION",
    Self::Minor => "MINOR_VERSION",
    Self::Patch => "PATCH_LEVEL",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EmulationStrategy {
    Default = library_types::cudaEmulationStrategy::CUDA_EMULATION_STRATEGY_DEFAULT as _,
    Performant = library_types::cudaEmulationStrategy::CUDA_EMULATION_STRATEGY_PERFORMANT as _,
    Eager = library_types::cudaEmulationStrategy::CUDA_EMULATION_STRATEGY_EAGER as _,
}

impl_enum_conversion!(library_types::cudaEmulationStrategy, EmulationStrategy);

impl_enum_display!(EmulationStrategy, {
    Self::Default => "CUDA_EMULATION_STRATEGY_DEFAULT",
    Self::Performant => "CUDA_EMULATION_STRATEGY_PERFORMANT",
    Self::Eager => "CUDA_EMULATION_STRATEGY_EAGER",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EmulationMantissaControl {
    Dynamic =
        library_types::cudaEmulationMantissaControl::CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC as _,
    Fixed = library_types::cudaEmulationMantissaControl::CUDA_EMULATION_MANTISSA_CONTROL_FIXED as _,
}

impl_enum_conversion!(
    library_types::cudaEmulationMantissaControl,
    EmulationMantissaControl
);

impl_enum_display!(EmulationMantissaControl, {
    Self::Dynamic => "CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC",
    Self::Fixed => "CUDA_EMULATION_MANTISSA_CONTROL_FIXED",
});

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EmulationSpecialValuesSupport: u32 {
        const NONE = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE as _;
        const INFINITY = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY as _;
        const NAN = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN as _;
        const DEFAULT = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT as _;
    }
}

impl Display for EmulationSpecialValuesSupport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        bitflags::parser::to_writer(self, f)
    }
}

impl From<library_types::cudaEmulationSpecialValuesSupport> for EmulationSpecialValuesSupport {
    fn from(value: library_types::cudaEmulationSpecialValuesSupport) -> Self {
        Self::from_bits_retain(value as u32)
    }
}

impl From<EmulationSpecialValuesSupport> for library_types::cudaEmulationSpecialValuesSupport {
    fn from(value: EmulationSpecialValuesSupport) -> Self {
        unsafe {
            core::mem::transmute::<u32, library_types::cudaEmulationSpecialValuesSupport>(
                value.bits(),
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PointerAttribute {
    Context = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_CONTEXT as _,
    MemoryType = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE as _,
    DevicePointer = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_POINTER as _,
    HostPointer = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_HOST_POINTER as _,
    P2pTokens = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_P2P_TOKENS as _,
    SyncMemops = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS as _,
    BufferId = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_BUFFER_ID as _,
    IsManaged = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_MANAGED as _,
    DeviceOrdinal = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL as _,
    IsLegacyCudaIpcCapable =
        driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE as _,
    RangeStartAddr = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR as _,
    RangeSize = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_RANGE_SIZE as _,
    Mapped = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MAPPED as _,
    AllowedHandleTypes =
        driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES as _,
    IsGpuDirectRdmaCapable =
        driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE as _,
    AccessFlags = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_ACCESS_FLAGS as _,
    MempoolHandle = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE as _,
    MappingSize = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MAPPING_SIZE as _,
    MappingBaseAddr = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR as _,
    MemoryBlockId = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID as _,
}

impl_enum_conversion!(u32, driver::CUpointer_attribute, PointerAttribute);

impl_enum_display!(PointerAttribute, {
    Self::Context => "CU_POINTER_ATTRIBUTE_CONTEXT",
    Self::MemoryType => "CU_POINTER_ATTRIBUTE_MEMORY_TYPE",
    Self::DevicePointer => "CU_POINTER_ATTRIBUTE_DEVICE_POINTER",
    Self::HostPointer => "CU_POINTER_ATTRIBUTE_HOST_POINTER",
    Self::P2pTokens => "CU_POINTER_ATTRIBUTE_P2P_TOKENS",
    Self::SyncMemops => "CU_POINTER_ATTRIBUTE_SYNC_MEMOPS",
    Self::BufferId => "CU_POINTER_ATTRIBUTE_BUFFER_ID",
    Self::IsManaged => "CU_POINTER_ATTRIBUTE_IS_MANAGED",
    Self::DeviceOrdinal => "CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL",
    Self::IsLegacyCudaIpcCapable => "CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE",
    Self::RangeStartAddr => "CU_POINTER_ATTRIBUTE_RANGE_START_ADDR",
    Self::RangeSize => "CU_POINTER_ATTRIBUTE_RANGE_SIZE",
    Self::Mapped => "CU_POINTER_ATTRIBUTE_MAPPED",
    Self::AllowedHandleTypes => "CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES",
    Self::IsGpuDirectRdmaCapable => "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE",
    Self::AccessFlags => "CU_POINTER_ATTRIBUTE_ACCESS_FLAGS",
    Self::MempoolHandle => "CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE",
    Self::MappingSize => "CU_POINTER_ATTRIBUTE_MAPPING_SIZE",
    Self::MappingBaseAddr => "CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR",
    Self::MemoryBlockId => "CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FunctionAttribute {
    MaxThreadsPerBlock =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _,
    SharedSizeBytes = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES as _,
    ConstSizeBytes = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES as _,
    LocalSizeBytes = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES as _,
    NumRegs = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS as _,
    PtxVersion = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION as _,
    BinaryVersion = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION as _,
    CacheModeCa = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA as _,
    MaxDynamicSharedSizeBytes =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES as _,
    PreferredSharedMemoryCarveout =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT as _,
    ClusterSizeMustBeSet =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET as _,
    RequiredClusterWidth =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH as _,
    RequiredClusterHeight =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT as _,
    RequiredClusterDepth =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH as _,
    NonPortableClusterSizeAllowed =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED as _,
    ClusterSchedulingPolicyPreference =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
            as _,
    Max = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX as _,
}

impl_enum_conversion!(u32, driver::CUfunction_attribute, FunctionAttribute);

impl_enum_display!(FunctionAttribute, {
    Self::MaxThreadsPerBlock => "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK",
    Self::SharedSizeBytes => "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES",
    Self::ConstSizeBytes => "CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES",
    Self::LocalSizeBytes => "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
    Self::NumRegs => "CU_FUNC_ATTRIBUTE_NUM_REGS",
    Self::PtxVersion => "CU_FUNC_ATTRIBUTE_PTX_VERSION",
    Self::BinaryVersion => "CU_FUNC_ATTRIBUTE_BINARY_VERSION",
    Self::CacheModeCa => "CU_FUNC_ATTRIBUTE_CACHE_MODE_CA",
    Self::MaxDynamicSharedSizeBytes => "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    Self::PreferredSharedMemoryCarveout => "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    Self::ClusterSizeMustBeSet => "CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET",
    Self::RequiredClusterWidth => "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH",
    Self::RequiredClusterHeight => "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT",
    Self::RequiredClusterDepth => "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH",
    Self::NonPortableClusterSizeAllowed => "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED",
    Self::ClusterSchedulingPolicyPreference => "CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE",
    Self::Max => "CU_FUNC_ATTRIBUTE_MAX",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FunctionCache {
    PreferNone = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_NONE as _,
    PreferShared = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_SHARED as _,
    PreferL1 = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_L1 as _,
    PreferEqual = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_EQUAL as _,
}

impl_enum_conversion!(u32, driver::CUfunc_cache, FunctionCache);

impl_enum_display!(FunctionCache, {
    Self::PreferNone => "CU_FUNC_CACHE_PREFER_NONE",
    Self::PreferShared => "CU_FUNC_CACHE_PREFER_SHARED",
    Self::PreferL1 => "CU_FUNC_CACHE_PREFER_L1",
    Self::PreferEqual => "CU_FUNC_CACHE_PREFER_EQUAL",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[deprecated]
#[non_exhaustive]
pub enum SharedMemoryConfig {
    DefaultBankSize = driver::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE as _,
    FourByteBankSize = driver::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE as _,
    EightByteBankSize = driver::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE as _,
}

impl_enum_conversion!(u32, driver::CUsharedconfig, SharedMemoryConfig);

impl_enum_display!(SharedMemoryConfig, {
    Self::DefaultBankSize => "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",
    Self::FourByteBankSize => "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",
    Self::EightByteBankSize => "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum SharedMemoryCarveout {
    Default = driver::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_DEFAULT as _,
    MaxShared = driver::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as _,
    MaxL1 = driver::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_MAX_L1 as _,
}

impl_enum_conversion!(i32, driver::CUshared_carveout, SharedMemoryCarveout);

impl_enum_display!(SharedMemoryCarveout, {
    Self::Default => "CU_SHAREDMEM_CARVEOUT_DEFAULT",
    Self::MaxShared => "CU_SHAREDMEM_CARVEOUT_MAX_SHARED",
    Self::MaxL1 => "CU_SHAREDMEM_CARVEOUT_MAX_L1",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryType {
    Host = driver::CUmemorytype_enum::CU_MEMORYTYPE_HOST as _,
    Device = driver::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE as _,
    Array = driver::CUmemorytype_enum::CU_MEMORYTYPE_ARRAY as _,
    Unified = driver::CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED as _,
}

impl_enum_conversion!(u32, driver::CUmemorytype, MemoryType);

impl_enum_display!(MemoryType, {
    Self::Host => "CU_MEMORYTYPE_HOST",
    Self::Device => "CU_MEMORYTYPE_DEVICE",
    Self::Array => "CU_MEMORYTYPE_ARRAY",
    Self::Unified => "CU_MEMORYTYPE_UNIFIED",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeMode {
    Default = driver::CUcomputemode_enum::CU_COMPUTEMODE_DEFAULT as _,
    Prohibited = driver::CUcomputemode_enum::CU_COMPUTEMODE_PROHIBITED as _,
    ExclusiveProcess = driver::CUcomputemode_enum::CU_COMPUTEMODE_EXCLUSIVE_PROCESS as _,
}

impl_enum_conversion!(u32, driver::CUcomputemode, ComputeMode);

impl_enum_display!(ComputeMode, {
    Self::Default => "CU_COMPUTEMODE_DEFAULT",
    Self::Prohibited => "CU_COMPUTEMODE_PROHIBITED",
    Self::ExclusiveProcess => "CU_COMPUTEMODE_EXCLUSIVE_PROCESS",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAdvise {
    SetReadMostly = driver::CUmem_advise_enum::CU_MEM_ADVISE_SET_READ_MOSTLY as _,
    UnsetReadMostly = driver::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_READ_MOSTLY as _,
    SetPreferredLocation = driver::CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION as _,
    UnsetPreferredLocation = driver::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION as _,
    SetAccessedBy = driver::CUmem_advise_enum::CU_MEM_ADVISE_SET_ACCESSED_BY as _,
    UnsetAccessedBy = driver::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_ACCESSED_BY as _,
}

impl_enum_conversion!(u32, driver::CUmem_advise, MemoryAdvise);

impl_enum_display!(MemoryAdvise, {
    Self::SetReadMostly => "CU_MEM_ADVISE_SET_READ_MOSTLY",
    Self::UnsetReadMostly => "CU_MEM_ADVISE_UNSET_READ_MOSTLY",
    Self::SetPreferredLocation => "CU_MEM_ADVISE_SET_PREFERRED_LOCATION",
    Self::UnsetPreferredLocation => "CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION",
    Self::SetAccessedBy => "CU_MEM_ADVISE_SET_ACCESSED_BY",
    Self::UnsetAccessedBy => "CU_MEM_ADVISE_UNSET_ACCESSED_BY",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryRangeAttribute {
    ReadMostly = driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY as _,
    PreferredLocation =
        driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION as _,
    AccessedBy = driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY as _,
    LastPrefetchLocation =
        driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION as _,
}

impl_enum_conversion!(u32, driver::CUmem_range_attribute, MemoryRangeAttribute);

impl_enum_display!(MemoryRangeAttribute, {
    Self::ReadMostly => "CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY",
    Self::PreferredLocation => "CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION",
    Self::AccessedBy => "CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY",
    Self::LastPrefetchLocation => "CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CubemapFace {
    PositiveX = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_POSITIVE_X as _,
    NegativeX = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_NEGATIVE_X as _,
    PositiveY = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_POSITIVE_Y as _,
    NegativeY = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_NEGATIVE_Y as _,
    PositiveZ = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_POSITIVE_Z as _,
    NegativeZ = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_NEGATIVE_Z as _,
}

impl_enum_conversion!(u32, driver::CUarray_cubemap_face, CubemapFace);

impl_enum_display!(CubemapFace, {
    Self::PositiveX => "CU_CUBEMAP_FACE_POSITIVE_X",
    Self::NegativeX => "CU_CUBEMAP_FACE_NEGATIVE_X",
    Self::PositiveY => "CU_CUBEMAP_FACE_POSITIVE_Y",
    Self::NegativeY => "CU_CUBEMAP_FACE_NEGATIVE_Y",
    Self::PositiveZ => "CU_CUBEMAP_FACE_POSITIVE_Z",
    Self::NegativeZ => "CU_CUBEMAP_FACE_NEGATIVE_Z",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Limit {
    StackSize = driver::CUlimit_enum::CU_LIMIT_STACK_SIZE as _,
    PrintfFifoSize = driver::CUlimit_enum::CU_LIMIT_PRINTF_FIFO_SIZE as _,
    MallocHeapSize = driver::CUlimit_enum::CU_LIMIT_MALLOC_HEAP_SIZE as _,
    DevRuntimeSyncDepth = driver::CUlimit_enum::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH as _,
    DevRuntimePendingLaunchCount =
        driver::CUlimit_enum::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT as _,
    MaxL2FetchGranularity = driver::CUlimit_enum::CU_LIMIT_MAX_L2_FETCH_GRANULARITY as _,
    PersistingL2CacheSize = driver::CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE as _,
    Max = driver::CUlimit_enum::CU_LIMIT_MAX as _,
}

impl_enum_conversion!(u32, driver::CUlimit, Limit);

impl_enum_display!(Limit, {
    Self::StackSize => "CU_LIMIT_STACK_SIZE",
    Self::PrintfFifoSize => "CU_LIMIT_PRINTF_FIFO_SIZE",
    Self::MallocHeapSize => "CU_LIMIT_MALLOC_HEAP_SIZE",
    Self::DevRuntimeSyncDepth => "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH",
    Self::DevRuntimePendingLaunchCount => "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT",
    Self::MaxL2FetchGranularity => "CU_LIMIT_MAX_L2_FETCH_GRANULARITY",
    Self::PersistingL2CacheSize => "CU_LIMIT_PERSISTING_L2_CACHE_SIZE",
    Self::Max => "CU_LIMIT_MAX",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ResourceType {
    Array = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY as _,
    MipmappedArray = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY as _,
    Linear = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_LINEAR as _,
    Pitch2d = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_PITCH2D as _,
}

impl_enum_conversion!(u32, driver::CUresourcetype, ResourceType);

impl_enum_display!(ResourceType, {
    Self::Array => "CU_RESOURCE_TYPE_ARRAY",
    Self::MipmappedArray => "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY",
    Self::Linear => "CU_RESOURCE_TYPE_LINEAR",
    Self::Pitch2d => "CU_RESOURCE_TYPE_PITCH2D",
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AccessProperty {
    Normal = driver::CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL as _,
    Streaming = driver::CUaccessProperty_enum::CU_ACCESS_PROPERTY_STREAMING as _,
    Persisting = driver::CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING as _,
}

impl_enum_conversion!(u32, driver::CUaccessProperty, AccessProperty);

impl_enum_display!(AccessProperty, {
    Self::Normal => "CU_ACCESS_PROPERTY_NORMAL",
    Self::Streaming => "CU_ACCESS_PROPERTY_STREAMING",
    Self::Persisting => "CU_ACCESS_PROPERTY_PERSISTING",
});
