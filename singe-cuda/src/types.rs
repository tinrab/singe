#![expect(
    deprecated,
    non_camel_case_types,
    reason = "wraps deprecated upstream API values that remain part of the supported surface; names mirror the upstream C API"
)]

use std::{
    fmt::{self, Display, Formatter},
    ptr,
};

use bitflags::parser;
pub use half::{bf16, f16};
pub use num_complex::{Complex, Complex32, Complex64};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda_sys::{driver, library_types};

use crate::view::{AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr};

macro_rules! impl_float_storage {
    ($(#[$meta:meta])* $name:ident, $bits:ty) => {
        $(#[$meta])*
        #[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(transparent)]
        pub struct $name(
            /// Raw storage bits, preserved without conversion or validation.
            pub $bits,
        );

        impl $name {
            /// Wraps storage bits without conversion, masking, or validation.
            pub const fn from_bits(bits: $bits) -> Self {
                Self(bits)
            }

            /// Returns the unchanged storage bits.
            pub const fn to_bits(self) -> $bits {
                self.0
            }
        }
    };
}

impl_float_storage!(
    /// Raw storage for an FP8 E4M3 encoding with four exponent and three fraction bits.
    f8e4m3,
    u8
);

impl_float_storage!(
    /// Raw storage for an FP8 E5M2 encoding with five exponent and two fraction bits.
    f8e5m2,
    u8
);

impl_float_storage!(
    /// Raw storage for an unsigned FP8 E8M0 encoding with eight exponent bits and no fraction bits.
    f8ue8m0,
    u8
);

impl_float_storage!(
    /// Real as an unsigned `nv_fp8_ue5m3`.
    /// Stores the unsigned UE5M3 encoding without converting it.
    f8ue5m3,
    u8
);

impl_float_storage!(
    /// Byte storage for a six-bit E2M3 floating-point encoding, without masking unused bits.
    f6e2m3,
    u8
);

impl_float_storage!(
    /// Byte storage for a six-bit E3M2 floating-point encoding, without masking unused bits.
    f6e3m2,
    u8
);

impl_float_storage!(
    /// Byte storage for a four-bit E2M1 floating-point encoding, without masking unused bits.
    f4e2m1,
    u8
);

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

/// Unowned CUDA device address. Copying this value retains no allocation and proves no bounds, lifetime, context association, alignment, or access permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct DevicePtr(*mut ());

unsafe impl DeviceRepr for DevicePtr {}

unsafe impl NoUninitDeviceRepr for DevicePtr {}

unsafe impl ZeroableDeviceRepr for DevicePtr {}

// DevicePtr is an opaque address; constructing it never dereferences the address.
unsafe impl AnyBitPatternDeviceRepr for DevicePtr {}

impl HostFunction {
    /// Wraps a raw CUDA host callback function pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a callback function pointer with the ABI and lifetime
    /// required by CUDA for every operation that uses the returned handle.
    #[must_use]
    pub const unsafe fn new(raw: driver::CUhostFn) -> Self {
        Self(raw)
    }

    /// Wraps a raw CUDA host callback function pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a callback function pointer with the ABI and lifetime
    /// required by CUDA for every operation that uses the returned handle.
    #[must_use]
    pub const unsafe fn from_raw(raw: driver::CUhostFn) -> Self {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::new(raw) }
    }

    /// Returns the stored native CUDA host callback pointer.
    #[must_use]
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
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::new(raw) }
    }

    /// Returns the stored native CUDA device function handle without transferring ownership.
    #[must_use]
    pub const fn as_raw(self) -> driver::CUfunction {
        self.0
    }

    /// Returns whether the stored native function handle is null.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl DevicePtr {
    /// Creates a null device address without allocating memory.
    #[must_use]
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
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Self::new(raw.cast()) }
    }

    /// Returns the untyped device pointer without retaining or validating its allocation.
    #[must_use]
    pub const fn as_raw(self) -> *mut () {
        self.0.cast()
    }

    /// Returns the stored mutable device pointer without dereferencing it.
    #[must_use]
    pub const fn as_ptr(self) -> *mut () {
        self.0
    }

    /// Returns the stored device address as a const pointer without dereferencing it.
    #[must_use]
    pub const fn as_const_ptr(self) -> *const () {
        self.0.cast_const()
    }

    /// Returns whether the device address is null.
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// The numeric device address.
    #[must_use]
    pub fn addr(self) -> usize {
        self.0.addr()
    }

    /// Whether the address is a multiple of `alignment`, which must be nonzero.
    #[must_use]
    pub fn is_aligned_to(self, alignment: usize) -> bool {
        self.addr().is_multiple_of(alignment)
    }

    /// Offsets the address by `bytes`, returning `None` when the address overflows.
    /// The result is an address computation only and is not checked against any allocation.
    #[must_use]
    pub fn checked_byte_add(self, bytes: usize) -> Option<Self> {
        self.addr()
            .checked_add(bytes)
            .map(|_| Self(self.0.wrapping_byte_add(bytes)))
    }

    /// Casts the device address to a pointer to `T` without checking alignment, bounds, or the allocation's element representation.
    #[must_use]
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
    /// Flags to register a graphics resource
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphicsRegisterFlags: u32 {
        /// Default registration behavior without additional access hints.
        const NONE = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_NONE as _;
        /// Hints that CUDA will only read the registered resource.
        const READ_ONLY = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY as _;
        /// Hints that CUDA will overwrite the registered resource without reading its previous contents.
        const WRITE_DISCARD = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD as _;
        /// Allows binding the registered image to a CUDA surface for load and store operations.
        const SURFACE_LDST = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST as _;
        /// Allows texture-gather operations on the registered image.
        const TEXTURE_GATHER = driver::CUgraphicsRegisterFlags::CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER as _;
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
        /// Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA kernels.  This is the default value.
        const NONE = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE as _;
        /// Hints that CUDA will only read the mapped resource.
        const READ_ONLY = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY as _;
        /// Hints that CUDA will overwrite the mapped resource without reading its previous contents.
        const WRITE_DISCARD = driver::CUgraphicsMapResourceFlags::CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD as _;
    }
}

impl Display for GraphicsMapResourceFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Component of a numerical library version queried through its property API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LibraryProperty {
    /// Major library version.
    Major = library_types::libraryPropertyType::MAJOR_VERSION as _,
    /// Minor library version.
    Minor = library_types::libraryPropertyType::MINOR_VERSION as _,
    /// Patch level of the library version.
    Patch = library_types::libraryPropertyType::PATCH_LEVEL as _,
}

impl_enum_conversion!(library_types::libraryPropertyType, LibraryProperty);

impl_enum_display!(LibraryProperty, {
    Self::Major => "MAJOR_VERSION",
    Self::Minor => "MINOR_VERSION",
    Self::Patch => "PATCH_LEVEL",
});

/// Enum for specifying how to leverage floating-point emulation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EmulationStrategy {
    /// The default emulation strategy.  For emulated computations, this is equivalent to `CUDA_EMULATION_STRATEGY_PERFORMANT`, unless a library dependent environment variable is set
    Default = library_types::cudaEmulationStrategy::CUDA_EMULATION_STRATEGY_DEFAULT as _,
    /// An emulation strategy which configures libraries to use emulation when it provides a performance benefit
    Performant = library_types::cudaEmulationStrategy::CUDA_EMULATION_STRATEGY_PERFORMANT as _,
    /// An emulation strategy which configures libraries to use emulation whenever possible
    Eager = library_types::cudaEmulationStrategy::CUDA_EMULATION_STRATEGY_EAGER as _,
}

impl_enum_conversion!(library_types::cudaEmulationStrategy, EmulationStrategy);

impl_enum_display!(EmulationStrategy, {
    Self::Default => "CUDA_EMULATION_STRATEGY_DEFAULT",
    Self::Performant => "CUDA_EMULATION_STRATEGY_PERFORMANT",
    Self::Eager => "CUDA_EMULATION_STRATEGY_EAGER",
});

/// Enum to configure the mantissa related parameters for floating-point emulation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum EmulationMantissaControl {
    /// The number of retained mantissa bits are computed at runtime to ensure the same or better accuracy than the floating point representation being emulated
    Dynamic =
        library_types::cudaEmulationMantissaControl::CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC as _,
    /// The number of retained mantissa bits are known at runtime
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
    /// Special input and output values that CUDA emulation algorithms must support.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct EmulationSpecialValuesSupport: u32 {
        /// Places no requirement on emulation algorithms to support special values.
        const NONE = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE.0;
        /// Requires support for signed infinity inputs and outputs.
        const INFINITY = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY.0;
        /// Requires support for NaN inputs and outputs.
        const NAN = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN.0;
        /// Requires the default support for both signed infinities and NaNs.
        const DEFAULT = library_types::cudaEmulationSpecialValuesSupport::CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT.0;
    }
}

impl Display for EmulationSpecialValuesSupport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

impl From<library_types::cudaEmulationSpecialValuesSupport> for EmulationSpecialValuesSupport {
    fn from(value: library_types::cudaEmulationSpecialValuesSupport) -> Self {
        Self::from_bits_retain(value.0)
    }
}

impl From<EmulationSpecialValuesSupport> for library_types::cudaEmulationSpecialValuesSupport {
    fn from(value: EmulationSpecialValuesSupport) -> Self {
        Self(value.bits())
    }
}

/// Pointer information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PointerAttribute {
    /// Returns in `*data` an integer representing the locality domain ordinal of the memory allocation, or -1 if the allocation is not localized to a locality domain.
    LocalityDomainOrdinal =
        driver::CUpointer_attribute::CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL as _,
    /// The `CUcontext` on which a pointer was allocated or registered.
    Context = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_CONTEXT as _,
    /// The [CUmemorytype](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga8a114cc994ad2e865c44ef3838eaec72) describing the physical location of a pointer.
    MemoryType = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE as _,
    /// The address at which a pointer's memory may be accessed on the device.
    DevicePointer = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_POINTER as _,
    /// The address at which a pointer's memory may be accessed on the host.
    HostPointer = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_HOST_POINTER as _,
    /// A pair of tokens for use with the nv-p2p.h Linux kernel interface.
    P2pTokens = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_P2P_TOKENS as _,
    /// Synchronize every synchronous memory operation initiated on this region.
    SyncMemops = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS as _,
    /// A process-wide unique ID for an allocated memory region.
    BufferId = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_BUFFER_ID as _,
    /// Indicates if the pointer points to managed memory.
    IsManaged = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_MANAGED as _,
    /// A device ordinal of a device on which a pointer was allocated or registered.
    DeviceOrdinal = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL as _,
    /// Reports whether the allocation supports legacy CUDA IPC memory handles.
    IsLegacyCudaIpcCapable =
        driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE as _,
    /// Starting address for this requested pointer.
    RangeStartAddr = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR as _,
    /// Size of the address range for this requested pointer.
    RangeSize = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_RANGE_SIZE as _,
    /// 1 if this pointer is in a valid address range that is mapped to a backing allocation, 0 otherwise.
    Mapped = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MAPPED as _,
    /// Bitmask of allowed [CUmemAllocationHandleType](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html#group__cuda__types_1ga450a23153d86fce0afe30e25d63caef9) for this allocation.
    AllowedHandleTypes =
        driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES as _,
    /// 1 if the memory this pointer is referencing can be used with the `GPUDirect` RDMA API.
    IsGpuDirectRdmaCapable =
        driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE as _,
    /// Returns the access flags the device associated with the current context has on the corresponding memory referenced by the pointer given.
    AccessFlags = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_ACCESS_FLAGS as _,
    /// Returns the mempool handle for the allocation if it was allocated from a mempool.
    ///
    /// Otherwise returns NULL.
    MempoolHandle = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE as _,
    /// Size of the actual underlying mapping that the pointer belongs to.
    MappingSize = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MAPPING_SIZE as _,
    /// The start address of the mapping that the pointer belongs to.
    MappingBaseAddr = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR as _,
    /// A process-wide unique id corresponding to the physical allocation the pointer belongs to.
    MemoryBlockId = driver::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID as _,
}

impl_enum_conversion!(u32, driver::CUpointer_attribute, PointerAttribute);

impl_enum_display!(PointerAttribute, {
    Self::LocalityDomainOrdinal => "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL",
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

/// Function properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FunctionAttribute {
    /// The shared memory mode of a function.
    ///
    /// The value type is `CUsharedMemoryMode` / cudaSharedMemoryMode.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    SharedMemoryMode = driver::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE as _,
    /// The maximum number of threads per block, beyond which a launch of the function would fail.
    ///
    /// This number depends on both the function and the device on which the function is currently loaded.
    MaxThreadsPerBlock =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _,
    /// The size in bytes of statically-allocated shared memory required by this function.
    ///
    /// This does not include dynamically-allocated shared memory requested by the user at runtime.
    SharedSizeBytes = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES as _,
    /// The size in bytes of user-allocated constant memory required by this function.
    ConstSizeBytes = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES as _,
    /// The size in bytes of local memory used by each thread of this function.
    LocalSizeBytes = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES as _,
    /// The number of registers used by each thread of this function.
    NumRegs = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS as _,
    /// The PTX virtual architecture version for which the function was compiled.
    ///
    /// This value is the major PTX version \* 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.
    PtxVersion = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PTX_VERSION as _,
    /// The binary architecture version for which the function was compiled.
    ///
    /// This value is the major binary version \* 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.
    BinaryVersion = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_BINARY_VERSION as _,
    /// The attribute to indicate whether the function has been compiled with user specified option "-Xptxas –dlcm=ca" set .
    CacheModeCa = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA as _,
    /// The maximum size in bytes of dynamically-allocated shared memory that can be used by this function.
    ///
    /// If the user-specified dynamic shared memory size is larger than this value, the launch will fail.
    ///
    /// The default value of this attribute is `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` - `CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`, except when `CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` is greater than `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, then the default value of this attribute is 0. The value can be increased to `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` - `CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`.
    ///
    /// This attribute is ignored if `CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE` or `CUlaunchAttributeID_enum::CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE` is set.
    ///
    /// This attribute cannot be used to access oversized shared memory. Oversized shared memory can only be accessed by setting `CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE` or `CUlaunchAttributeID_enum::CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE`.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    MaxDynamicSharedSizeBytes =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES as _,
    /// On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory.
    ///
    /// Refer to `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`. This is only a hint, and the driver can choose a different ratio if required to execute the function.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    PreferredSharedMemoryCarveout =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT as _,
    /// If this attribute is set, the kernel must launch with a valid cluster size specified.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    ClusterSizeMustBeSet =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET as _,
    /// The required cluster width in blocks.
    ///
    /// The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.
    ///
    /// If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return `CUresult::CUDA_ERROR_NOT_PERMITTED`.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    RequiredClusterWidth =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH as _,
    /// The required cluster height in blocks.
    ///
    /// The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.
    ///
    /// If the value is set during compile time, it cannot be set at runtime. Setting it at runtime should return `CUresult::CUDA_ERROR_NOT_PERMITTED`.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    RequiredClusterHeight =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT as _,
    /// The required cluster depth in blocks.
    ///
    /// The values must either all be 0 or all be positive. The validity of the cluster dimensions is otherwise checked at launch time.
    ///
    /// If the value is set during compile time, it cannot be set at runtime. Setting it at runtime should return `CUresult::CUDA_ERROR_NOT_PERMITTED`.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    RequiredClusterDepth =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH as _,
    /// Whether the function can be launched with non-portable cluster size.
    ///
    /// 1 is allowed, 0 is disallowed. A non-portable cluster size may only function on the specific SKUs the program is tested on. The launch might fail if the program is run on a different hardware platform.
    ///
    /// CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking whether the desired size can be launched on the current device.
    ///
    /// Portable Cluster Size
    ///
    /// A portable cluster size is guaranteed to be functional on all compute capabilities higher than the target compute capability. The portable cluster size for `sm_90` is 8 blocks per cluster. This value may increase for future compute capabilities.
    ///
    /// The specific hardware unit may support higher cluster sizes that's not guaranteed to be portable.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    NonPortableClusterSizeAllowed =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED as _,
    /// The block scheduling policy of a function.
    ///
    /// The value type is `CUclusterSchedulingPolicy` / cudaClusterSchedulingPolicy.
    ///
    /// See `cuFuncSetAttribute`, `cuKernelSetAttribute`.
    ClusterSchedulingPolicyPreference =
        driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
            as _,
    /// Upper-bound sentinel for CUDA function attribute selectors.
    Max = driver::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX as _,
}

impl_enum_conversion!(u32, driver::CUfunction_attribute, FunctionAttribute);

impl_enum_display!(FunctionAttribute, {
    Self::SharedMemoryMode => "CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE",
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

/// Function cache configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FunctionCache {
    /// no preference for shared memory or L1 (default).
    PreferNone = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_NONE as _,
    /// prefer larger shared memory and smaller L1 cache.
    PreferShared = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_SHARED as _,
    /// prefer larger L1 cache and smaller shared memory.
    PreferL1 = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_L1 as _,
    /// prefer equal sized L1 cache and shared memory.
    PreferEqual = driver::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_EQUAL as _,
}

impl_enum_conversion!(u32, driver::CUfunc_cache, FunctionCache);

impl_enum_display!(FunctionCache, {
    Self::PreferNone => "CU_FUNC_CACHE_PREFER_NONE",
    Self::PreferShared => "CU_FUNC_CACHE_PREFER_SHARED",
    Self::PreferL1 => "CU_FUNC_CACHE_PREFER_L1",
    Self::PreferEqual => "CU_FUNC_CACHE_PREFER_EQUAL",
});

/// Deprecated:
///
/// Shared memory configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[deprecated]
#[non_exhaustive]
pub enum SharedMemoryConfig {
    /// set default shared memory bank size.
    DefaultBankSize = driver::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE as _,
    /// set shared memory bank width to four bytes.
    FourByteBankSize = driver::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE as _,
    /// set shared memory bank width to eight bytes.
    EightByteBankSize = driver::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE as _,
}

impl_enum_conversion!(u32, driver::CUsharedconfig, SharedMemoryConfig);

impl_enum_display!(SharedMemoryConfig, {
    Self::DefaultBankSize => "CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE",
    Self::FourByteBankSize => "CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE",
    Self::EightByteBankSize => "CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE",
});

/// Shared memory carveout configurations. These may be passed to `cuFuncSetAttribute` or `cuKernelSetAttribute`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
#[non_exhaustive]
pub enum SharedMemoryCarveout {
    /// No preference for shared memory or L1 (default).
    Default = driver::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_DEFAULT as _,
    /// Prefer maximum available shared memory, minimum L1 cache.
    MaxShared = driver::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_MAX_SHARED as _,
    /// Prefer maximum available L1 cache, minimum shared memory.
    MaxL1 = driver::CUshared_carveout_enum::CU_SHAREDMEM_CARVEOUT_MAX_L1 as _,
}

impl_enum_conversion!(i32, driver::CUshared_carveout, SharedMemoryCarveout);

impl_enum_display!(SharedMemoryCarveout, {
    Self::Default => "CU_SHAREDMEM_CARVEOUT_DEFAULT",
    Self::MaxShared => "CU_SHAREDMEM_CARVEOUT_MAX_SHARED",
    Self::MaxL1 => "CU_SHAREDMEM_CARVEOUT_MAX_L1",
});

/// Memory types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryType {
    /// Host memory.
    Host = driver::CUmemorytype_enum::CU_MEMORYTYPE_HOST as _,
    /// Device memory.
    Device = driver::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE as _,
    /// Array memory.
    Array = driver::CUmemorytype_enum::CU_MEMORYTYPE_ARRAY as _,
    /// Unified device or host memory.
    Unified = driver::CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED as _,
}

impl_enum_conversion!(u32, driver::CUmemorytype, MemoryType);

impl_enum_display!(MemoryType, {
    Self::Host => "CU_MEMORYTYPE_HOST",
    Self::Device => "CU_MEMORYTYPE_DEVICE",
    Self::Array => "CU_MEMORYTYPE_ARRAY",
    Self::Unified => "CU_MEMORYTYPE_UNIFIED",
});

/// Compute Modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ComputeMode {
    /// Default compute mode (Multiple contexts allowed per device).
    Default = driver::CUcomputemode_enum::CU_COMPUTEMODE_DEFAULT as _,
    /// Compute-prohibited mode (No contexts can be created on this device at this time).
    Prohibited = driver::CUcomputemode_enum::CU_COMPUTEMODE_PROHIBITED as _,
    /// Compute-exclusive-process mode (Only one context used by a single process can be present on this device at a time).
    ExclusiveProcess = driver::CUcomputemode_enum::CU_COMPUTEMODE_EXCLUSIVE_PROCESS as _,
}

impl_enum_conversion!(u32, driver::CUcomputemode, ComputeMode);

impl_enum_display!(ComputeMode, {
    Self::Default => "CU_COMPUTEMODE_DEFAULT",
    Self::Prohibited => "CU_COMPUTEMODE_PROHIBITED",
    Self::ExclusiveProcess => "CU_COMPUTEMODE_EXCLUSIVE_PROCESS",
});

/// Memory advise values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryAdvise {
    /// Data will mostly be read and only occasionally be written to.
    SetReadMostly = driver::CUmem_advise_enum::CU_MEM_ADVISE_SET_READ_MOSTLY as _,
    /// Undo the effect of `CUmem_advise_enum::CU_MEM_ADVISE_SET_READ_MOSTLY`.
    UnsetReadMostly = driver::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_READ_MOSTLY as _,
    /// Set the preferred location for the data as the specified device.
    SetPreferredLocation = driver::CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION as _,
    /// Clear the preferred location for the data.
    UnsetPreferredLocation = driver::CUmem_advise_enum::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION as _,
    /// Data will be accessed by the specified device, so prevent page faults as much as possible.
    SetAccessedBy = driver::CUmem_advise_enum::CU_MEM_ADVISE_SET_ACCESSED_BY as _,
    /// Let the Unified Memory subsystem decide on the page faulting policy for the specified device.
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

/// Attribute queried for a managed-memory address range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MemoryRangeAttribute {
    /// Whether the range will mostly be read and only occasionally be written to.
    ReadMostly = driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY as _,
    /// The preferred location of the range.
    PreferredLocation =
        driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION as _,
    /// Memory range has `CUmem_advise_enum::CU_MEM_ADVISE_SET_ACCESSED_BY` set for specified device.
    AccessedBy = driver::CUmem_range_attribute_enum::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY as _,
    /// The last location to which the range was prefetched.
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

/// Array indices for cube faces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CubemapFace {
    /// Positive X face of cubemap.
    PositiveX = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_POSITIVE_X as _,
    /// Negative X face of cubemap.
    NegativeX = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_NEGATIVE_X as _,
    /// Positive Y face of cubemap.
    PositiveY = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_POSITIVE_Y as _,
    /// Negative Y face of cubemap.
    NegativeY = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_NEGATIVE_Y as _,
    /// Positive Z face of cubemap.
    PositiveZ = driver::CUarray_cubemap_face_enum::CU_CUBEMAP_FACE_POSITIVE_Z as _,
    /// Negative Z face of cubemap.
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

/// Limits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Limit {
    /// Per-block memory size.
    PerBlockMemorySize = driver::CUlimit::CU_LIMIT_PER_BLOCK_MEMORY_SIZE as _,
    /// GPU thread stack size.
    StackSize = driver::CUlimit_enum::CU_LIMIT_STACK_SIZE as _,
    /// GPU printf FIFO size.
    PrintfFifoSize = driver::CUlimit_enum::CU_LIMIT_PRINTF_FIFO_SIZE as _,
    /// GPU malloc heap size.
    MallocHeapSize = driver::CUlimit_enum::CU_LIMIT_MALLOC_HEAP_SIZE as _,
    /// GPU device runtime launch synchronize depth.
    DevRuntimeSyncDepth = driver::CUlimit_enum::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH as _,
    /// GPU device runtime pending launch count.
    DevRuntimePendingLaunchCount =
        driver::CUlimit_enum::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT as _,
    /// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes).
    ///
    /// This is a hint.
    MaxL2FetchGranularity = driver::CUlimit_enum::CU_LIMIT_MAX_L2_FETCH_GRANULARITY as _,
    /// A size in bytes for L2 persisting lines cache size.
    PersistingL2CacheSize = driver::CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE as _,
    /// Reserved upper-bound sentinel for native device limits.
    Max = driver::CUlimit_enum::CU_LIMIT_MAX as _,
}

impl_enum_conversion!(u32, driver::CUlimit, Limit);

impl_enum_display!(Limit, {
    Self::PerBlockMemorySize => "CU_LIMIT_PER_BLOCK_MEMORY_SIZE",
    Self::StackSize => "CU_LIMIT_STACK_SIZE",
    Self::PrintfFifoSize => "CU_LIMIT_PRINTF_FIFO_SIZE",
    Self::MallocHeapSize => "CU_LIMIT_MALLOC_HEAP_SIZE",
    Self::DevRuntimeSyncDepth => "CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH",
    Self::DevRuntimePendingLaunchCount => "CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT",
    Self::MaxL2FetchGranularity => "CU_LIMIT_MAX_L2_FETCH_GRANULARITY",
    Self::PersistingL2CacheSize => "CU_LIMIT_PERSISTING_L2_CACHE_SIZE",
    Self::Max => "CU_LIMIT_MAX",
});

/// Resource types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ResourceType {
    /// Array resource.
    Array = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_ARRAY as _,
    /// Mipmapped array resource.
    MipmappedArray = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY as _,
    /// Linear resource.
    Linear = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_LINEAR as _,
    /// Pitch 2D resource.
    Pitch2d = driver::CUresourcetype_enum::CU_RESOURCE_TYPE_PITCH2D as _,
}

impl_enum_conversion!(u32, driver::CUresourcetype, ResourceType);

impl_enum_display!(ResourceType, {
    Self::Array => "CU_RESOURCE_TYPE_ARRAY",
    Self::MipmappedArray => "CU_RESOURCE_TYPE_MIPMAPPED_ARRAY",
    Self::Linear => "CU_RESOURCE_TYPE_LINEAR",
    Self::Pitch2d => "CU_RESOURCE_TYPE_PITCH2D",
});

/// Cache performance hint for the hit and miss portions of an access-policy window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum AccessProperty {
    /// Normal cache persistence.
    Normal = driver::CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL as _,
    /// Streaming access is less likely to persit from cache.
    Streaming = driver::CUaccessProperty_enum::CU_ACCESS_PROPERTY_STREAMING as _,
    /// Persisting access is more likely to persist in cache.
    Persisting = driver::CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING as _,
}

impl_enum_conversion!(u32, driver::CUaccessProperty, AccessProperty);

impl_enum_display!(AccessProperty, {
    Self::Normal => "CU_ACCESS_PROPERTY_NORMAL",
    Self::Streaming => "CU_ACCESS_PROPERTY_STREAMING",
    Self::Persisting => "CU_ACCESS_PROPERTY_PERSISTING",
});

/// Shared memory configuration for a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SharedMemoryMode {
    /// The default to use for shared memory on launch - uses current function attribute for `CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
    Default = driver::CUsharedMemoryMode_enum::CU_SHARED_MEMORY_MODE_DEFAULT as _,
    /// Specifies that the dynamic shared size bytes requested must be a portable size within the bounds of `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`.
    RequirePortable = driver::CUsharedMemoryMode_enum::CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE as _,
    /// Specifies that the dynamic shared size bytes requested may be a non-portable size but still within the bounds of `CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
    AllowNonPortable =
        driver::CUsharedMemoryMode_enum::CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE as _,
    /// Specifies that oversized shared memory configurations may be used (with the limitation of only 8kB L1 cache).
    AllowOversizedSharedMemory =
        driver::CUsharedMemoryMode_enum::CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY as _,
    /// Specifies that oversized shared memory configurations may be used (with the limitation of only 8kB L1 cache), and prefer an oversized shared memory configuration.
    PreferOversizedSharedMemory =
        driver::CUsharedMemoryMode_enum::CU_SHARED_MEMORY_MODE_PREFER_OVERSIZED_SHARED_MEMORY as _,
}

impl_enum_conversion!(u32, driver::CUsharedMemoryMode_enum, SharedMemoryMode);

/// Cluster scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ClusterSchedulingPolicy {
    /// the default policy.
    Default = driver::CUclusterSchedulingPolicy_enum::CU_CLUSTER_SCHEDULING_POLICY_DEFAULT as _,
    /// spread the blocks within a cluster to the SMs.
    Spread = driver::CUclusterSchedulingPolicy_enum::CU_CLUSTER_SCHEDULING_POLICY_SPREAD as _,
    /// allow the hardware to load-balance the blocks in a cluster to the SMs.
    LoadBalancing =
        driver::CUclusterSchedulingPolicy_enum::CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING as _,
    /// Spread the blocks within a cluster to the SMs, and prefer placement that prioritizes Distributed Shared Memory locality for Rubin-class GPUs.
    ///
    /// This is a preference only. The driver may ignore this request when the launch is not eligible, when the device does not support the requested placement behavior, or when other launch constraints take precedence.
    ///
    /// This policy is intended for clustered kernels whose performance depends on communication through Distributed Shared Memory among blocks in the same cluster.
    RubinDsmemLocality =
        driver::CUclusterSchedulingPolicy_enum::CU_CLUSTER_SCHEDULING_POLICY_RUBIN_DSMEM_LOCALITY
            as _,
}

impl_enum_conversion!(
    u32,
    driver::CUclusterSchedulingPolicy_enum,
    ClusterSchedulingPolicy
);

impl_enum_display!(SharedMemoryMode, {
    SharedMemoryMode::Default => "CU_SHARED_MEMORY_MODE_DEFAULT",
    SharedMemoryMode::RequirePortable => "CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE",
    SharedMemoryMode::AllowNonPortable => "CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE",
    SharedMemoryMode::AllowOversizedSharedMemory => "CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY",
    SharedMemoryMode::PreferOversizedSharedMemory => "CU_SHARED_MEMORY_MODE_PREFER_OVERSIZED_SHARED_MEMORY",
});

impl_enum_display!(ClusterSchedulingPolicy, {
    ClusterSchedulingPolicy::Default => "CU_CLUSTER_SCHEDULING_POLICY_DEFAULT",
    ClusterSchedulingPolicy::Spread => "CU_CLUSTER_SCHEDULING_POLICY_SPREAD",
    ClusterSchedulingPolicy::LoadBalancing => "CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING",
    ClusterSchedulingPolicy::RubinDsmemLocality => "CU_CLUSTER_SCHEDULING_POLICY_RUBIN_DSMEM_LOCALITY",
});
