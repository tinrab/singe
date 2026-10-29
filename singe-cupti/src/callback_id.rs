use singe_cupti_sys as sys;

use crate::types::CallbackId;

/// CUDA driver API callback IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DriverCallbackId(u32);

impl DriverCallbackId {
    /// Returns the raw callback identifier.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Callback identifier for `cuInit`.
    pub const CU_INIT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuInit as u32);

    /// Callback identifier for `cuDriverGetVersion`.
    pub const CU_DRIVER_GET_VERSION: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDriverGetVersion as u32,
    );

    /// Callback identifier for `cuDeviceGet`.
    pub const CU_DEVICE_GET: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGet as u32);

    /// Callback identifier for `cuDeviceGetCount`.
    pub const CU_DEVICE_GET_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetCount as u32,
    );

    /// Callback identifier for `cuDeviceGetName`.
    pub const CU_DEVICE_GET_NAME: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetName as u32);

    /// Callback identifier for `cuDeviceComputeCapability`.
    pub const CU_DEVICE_COMPUTE_CAPABILITY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceComputeCapability
            as u32,
    );

    /// Callback identifier for `cuDeviceTotalMem`.
    pub const CU_DEVICE_TOTAL_MEM: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceTotalMem as u32,
    );

    /// Callback identifier for `cuDeviceGetProperties`.
    pub const CU_DEVICE_GET_PROPERTIES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetProperties as u32,
    );

    /// Callback identifier for `cuDeviceGetAttribute`.
    pub const CU_DEVICE_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetAttribute as u32,
    );

    /// Callback identifier for `cuCtxCreate`.
    pub const CU_CTX_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxCreate as u32);

    /// Callback identifier for `cuCtxDestroy`.
    pub const CU_CTX_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy as u32);

    /// Callback identifier for `cuCtxAttach`.
    pub const CU_CTX_ATTACH: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxAttach as u32);

    /// Callback identifier for `cuCtxDetach`.
    pub const CU_CTX_DETACH: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxDetach as u32);

    /// Callback identifier for `cuCtxPushCurrent`.
    pub const CU_CTX_PUSH_CURRENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent as u32,
    );

    /// Callback identifier for `cuCtxPopCurrent`.
    pub const CU_CTX_POP_CURRENT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent as u32);

    /// Callback identifier for `cuCtxGetDevice`.
    pub const CU_CTX_GET_DEVICE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevice as u32);

    /// Callback identifier for `cuCtxSynchronize`.
    pub const CU_CTX_SYNCHRONIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize as u32,
    );

    /// Callback identifier for `cuModuleLoad`.
    pub const CU_MODULE_LOAD: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleLoad as u32);

    /// Callback identifier for `cuModuleLoadData`.
    pub const CU_MODULE_LOAD_DATA: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData as u32,
    );

    /// Callback identifier for `cuModuleLoadDataEx`.
    pub const CU_MODULE_LOAD_DATA_EX: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx as u32,
    );

    /// Callback identifier for `cuModuleLoadFatBinary`.
    pub const CU_MODULE_LOAD_FAT_BINARY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary as u32,
    );

    /// Callback identifier for `cuModuleUnload`.
    pub const CU_MODULE_UNLOAD: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleUnload as u32);

    /// Callback identifier for `cuModuleGetFunction`.
    pub const CU_MODULE_GET_FUNCTION: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction as u32,
    );

    /// Callback identifier for `cuModuleGetGlobal`.
    pub const CU_MODULE_GET_GLOBAL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal as u32,
    );

    /// Callback identifier for `cu64ModuleGetGlobal`.
    pub const CU64_MODULE_GET_GLOBAL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64ModuleGetGlobal as u32,
    );

    /// Callback identifier for `cuModuleGetTexRef`.
    pub const CU_MODULE_GET_TEX_REF: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetTexRef as u32,
    );

    /// Callback identifier for `cuMemGetInfo`.
    pub const CU_MEM_GET_INFO: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetInfo as u32);

    /// Callback identifier for `cu64MemGetInfo`.
    pub const CU64_MEM_GET_INFO: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemGetInfo as u32);

    /// Callback identifier for `cuMemAlloc`.
    pub const CU_MEM_ALLOC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAlloc as u32);

    /// Callback identifier for `cu64MemAlloc`.
    pub const CU64_MEM_ALLOC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemAlloc as u32);

    /// Callback identifier for `cuMemAllocPitch`.
    pub const CU_MEM_ALLOC_PITCH: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch as u32);

    /// Callback identifier for `cu64MemAllocPitch`.
    pub const CU64_MEM_ALLOC_PITCH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemAllocPitch as u32,
    );

    /// Callback identifier for `cuMemFree`.
    pub const CU_MEM_FREE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemFree as u32);

    /// Callback identifier for `cu64MemFree`.
    pub const CU64_MEM_FREE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemFree as u32);

    /// Callback identifier for `cuMemGetAddressRange`.
    pub const CU_MEM_GET_ADDRESS_RANGE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange as u32,
    );

    /// Callback identifier for `cu64MemGetAddressRange`.
    pub const CU64_MEM_GET_ADDRESS_RANGE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemGetAddressRange
            as u32,
    );

    /// Callback identifier for `cuMemAllocHost`.
    pub const CU_MEM_ALLOC_HOST: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost as u32);

    /// Callback identifier for `cuMemFreeHost`.
    pub const CU_MEM_FREE_HOST: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost as u32);

    /// Callback identifier for `cuMemHostAlloc`.
    pub const CU_MEM_HOST_ALLOC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc as u32);

    /// Callback identifier for `cuMemHostGetDevicePointer`.
    pub const CU_MEM_HOST_GET_DEVICE_POINTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostGetDevicePointer
            as u32,
    );

    /// Callback identifier for `cu64MemHostGetDevicePointer`.
    pub const CU64_MEM_HOST_GET_DEVICE_POINTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemHostGetDevicePointer
            as u32,
    );

    /// Callback identifier for `cuMemHostGetFlags`.
    pub const CU_MEM_HOST_GET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostGetFlags as u32,
    );

    /// Callback identifier for `cuMemcpyHtoD`.
    pub const CU_MEMCPY_HTO_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD as u32);

    /// Callback identifier for `cu64MemcpyHtoD`.
    pub const CU64_MEMCPY_HTO_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyHtoD as u32);

    /// Callback identifier for `cuMemcpyDtoH`.
    pub const CU_MEMCPY_DTO_H: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH as u32);

    /// Callback identifier for `cu64MemcpyDtoH`.
    pub const CU64_MEMCPY_DTO_H: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoH as u32);

    /// Callback identifier for `cuMemcpyDtoD`.
    pub const CU_MEMCPY_DTO_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD as u32);

    /// Callback identifier for `cu64MemcpyDtoD`.
    pub const CU64_MEMCPY_DTO_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoD as u32);

    /// Callback identifier for `cuMemcpyDtoA`.
    pub const CU_MEMCPY_DTO_A: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA as u32);

    /// Callback identifier for `cu64MemcpyDtoA`.
    pub const CU64_MEMCPY_DTO_A: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoA as u32);

    /// Callback identifier for `cuMemcpyAtoD`.
    pub const CU_MEMCPY_ATO_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD as u32);

    /// Callback identifier for `cu64MemcpyAtoD`.
    pub const CU64_MEMCPY_ATO_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyAtoD as u32);

    /// Callback identifier for `cuMemcpyHtoA`.
    pub const CU_MEMCPY_HTO_A: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA as u32);

    /// Callback identifier for `cuMemcpyAtoH`.
    pub const CU_MEMCPY_ATO_H: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH as u32);

    /// Callback identifier for `cuMemcpyAtoA`.
    pub const CU_MEMCPY_ATO_A: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA as u32);

    /// Callback identifier for `cuMemcpy2D`.
    pub const CU_MEMCPY2_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D as u32);

    /// Callback identifier for `cuMemcpy2DUnaligned`.
    pub const CU_MEMCPY2_D_UNALIGNED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned as u32,
    );

    /// Callback identifier for `cuMemcpy3D`.
    pub const CU_MEMCPY3_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D as u32);

    /// Callback identifier for `cu64Memcpy3D`.
    pub const CU64_MEMCPY3_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Memcpy3D as u32);

    /// Callback identifier for `cuMemcpyHtoDAsync`.
    pub const CU_MEMCPY_HTO_D_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync as u32,
    );

    /// Callback identifier for `cu64MemcpyHtoDAsync`.
    pub const CU64_MEMCPY_HTO_D_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyHtoDAsync as u32,
    );

    /// Callback identifier for `cuMemcpyDtoHAsync`.
    pub const CU_MEMCPY_DTO_H_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync as u32,
    );

    /// Callback identifier for `cu64MemcpyDtoHAsync`.
    pub const CU64_MEMCPY_DTO_H_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoHAsync as u32,
    );

    /// Callback identifier for `cuMemcpyDtoDAsync`.
    pub const CU_MEMCPY_DTO_D_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync as u32,
    );

    /// Callback identifier for `cu64MemcpyDtoDAsync`.
    pub const CU64_MEMCPY_DTO_D_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemcpyDtoDAsync as u32,
    );

    /// Callback identifier for `cuMemcpyHtoAAsync`.
    pub const CU_MEMCPY_HTO_A_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync as u32,
    );

    /// Callback identifier for `cuMemcpyAtoHAsync`.
    pub const CU_MEMCPY_ATO_H_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync as u32,
    );

    /// Callback identifier for `cuMemcpy2DAsync`.
    pub const CU_MEMCPY2_D_ASYNC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync as u32);

    /// Callback identifier for `cuMemcpy3DAsync`.
    pub const CU_MEMCPY3_D_ASYNC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync as u32);

    /// Callback identifier for `cu64Memcpy3DAsync`.
    pub const CU64_MEMCPY3_D_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Memcpy3DAsync as u32,
    );

    /// Callback identifier for `cuMemsetD8`.
    pub const CU_MEMSET_D8: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD8 as u32);

    /// Callback identifier for `cu64MemsetD8`.
    pub const CU64_MEMSET_D8: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD8 as u32);

    /// Callback identifier for `cuMemsetD16`.
    pub const CU_MEMSET_D16: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD16 as u32);

    /// Callback identifier for `cu64MemsetD16`.
    pub const CU64_MEMSET_D16: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD16 as u32);

    /// Callback identifier for `cuMemsetD32`.
    pub const CU_MEMSET_D32: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD32 as u32);

    /// Callback identifier for `cu64MemsetD32`.
    pub const CU64_MEMSET_D32: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD32 as u32);

    /// Callback identifier for `cuMemsetD2D8`.
    pub const CU_MEMSET_D2_D8: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8 as u32);

    /// Callback identifier for `cu64MemsetD2D8`.
    pub const CU64_MEMSET_D2_D8: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D8 as u32);

    /// Callback identifier for `cuMemsetD2D16`.
    pub const CU_MEMSET_D2_D16: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16 as u32);

    /// Callback identifier for `cu64MemsetD2D16`.
    pub const CU64_MEMSET_D2_D16: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D16 as u32);

    /// Callback identifier for `cuMemsetD2D32`.
    pub const CU_MEMSET_D2_D32: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32 as u32);

    /// Callback identifier for `cu64MemsetD2D32`.
    pub const CU64_MEMSET_D2_D32: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D32 as u32);

    /// Callback identifier for `cuFuncSetBlockShape`.
    pub const CU_FUNC_SET_BLOCK_SHAPE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncSetBlockShape as u32,
    );

    /// Callback identifier for `cuFuncSetSharedSize`.
    pub const CU_FUNC_SET_SHARED_SIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncSetSharedSize as u32,
    );

    /// Callback identifier for `cuFuncGetAttribute`.
    pub const CU_FUNC_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncGetAttribute as u32,
    );

    /// Callback identifier for `cuFuncSetCacheConfig`.
    pub const CU_FUNC_SET_CACHE_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncSetCacheConfig as u32,
    );

    /// Callback identifier for `cuArrayCreate`.
    pub const CU_ARRAY_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayCreate as u32);

    /// Callback identifier for `cuArrayGetDescriptor`.
    pub const CU_ARRAY_GET_DESCRIPTOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor as u32,
    );

    /// Callback identifier for `cuArrayDestroy`.
    pub const CU_ARRAY_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy as u32);

    /// Callback identifier for `cuArray3DCreate`.
    pub const CU_ARRAY3_D_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate as u32);

    /// Callback identifier for `cuArray3DGetDescriptor`.
    pub const CU_ARRAY3_D_GET_DESCRIPTOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor
            as u32,
    );

    /// Callback identifier for `cuTexRefCreate`.
    pub const CU_TEX_REF_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefCreate as u32);

    /// Callback identifier for `cuTexRefDestroy`.
    pub const CU_TEX_REF_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefDestroy as u32);

    /// Callback identifier for `cuTexRefSetArray`.
    pub const CU_TEX_REF_SET_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetArray as u32,
    );

    /// Callback identifier for `cuTexRefSetAddress`.
    pub const CU_TEX_REF_SET_ADDRESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress as u32,
    );

    /// Callback identifier for `cu64TexRefSetAddress`.
    pub const CU64_TEX_REF_SET_ADDRESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64TexRefSetAddress as u32,
    );

    /// Callback identifier for `cuTexRefSetAddress2D`.
    pub const CU_TEX_REF_SET_ADDRESS2_D: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D as u32,
    );

    /// Callback identifier for `cu64TexRefSetAddress2D`.
    pub const CU64_TEX_REF_SET_ADDRESS2_D: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64TexRefSetAddress2D
            as u32,
    );

    /// Callback identifier for `cuTexRefSetFormat`.
    pub const CU_TEX_REF_SET_FORMAT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFormat as u32,
    );

    /// Callback identifier for `cuTexRefSetAddressMode`.
    pub const CU_TEX_REF_SET_ADDRESS_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddressMode
            as u32,
    );

    /// Callback identifier for `cuTexRefSetFilterMode`.
    pub const CU_TEX_REF_SET_FILTER_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFilterMode as u32,
    );

    /// Callback identifier for `cuTexRefSetFlags`.
    pub const CU_TEX_REF_SET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetFlags as u32,
    );

    /// Callback identifier for `cuTexRefGetAddress`.
    pub const CU_TEX_REF_GET_ADDRESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddress as u32,
    );

    /// Callback identifier for `cu64TexRefGetAddress`.
    pub const CU64_TEX_REF_GET_ADDRESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64TexRefGetAddress as u32,
    );

    /// Callback identifier for `cuTexRefGetArray`.
    pub const CU_TEX_REF_GET_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetArray as u32,
    );

    /// Callback identifier for `cuTexRefGetAddressMode`.
    pub const CU_TEX_REF_GET_ADDRESS_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddressMode
            as u32,
    );

    /// Callback identifier for `cuTexRefGetFilterMode`.
    pub const CU_TEX_REF_GET_FILTER_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFilterMode as u32,
    );

    /// Callback identifier for `cuTexRefGetFormat`.
    pub const CU_TEX_REF_GET_FORMAT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFormat as u32,
    );

    /// Callback identifier for `cuTexRefGetFlags`.
    pub const CU_TEX_REF_GET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetFlags as u32,
    );

    /// Callback identifier for `cuParamSetSize`.
    pub const CU_PARAM_SET_SIZE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuParamSetSize as u32);

    /// Callback identifier for `cuParamSeti`.
    pub const CU_PARAM_SETI: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuParamSeti as u32);

    /// Callback identifier for `cuParamSetf`.
    pub const CU_PARAM_SETF: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuParamSetf as u32);

    /// Callback identifier for `cuParamSetv`.
    pub const CU_PARAM_SETV: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuParamSetv as u32);

    /// Callback identifier for `cuParamSetTexRef`.
    pub const CU_PARAM_SET_TEX_REF: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuParamSetTexRef as u32,
    );

    /// Callback identifier for `cuLaunch`.
    pub const CU_LAUNCH: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunch as u32);

    /// Callback identifier for `cuLaunchGrid`.
    pub const CU_LAUNCH_GRID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid as u32);

    /// Callback identifier for `cuLaunchGridAsync`.
    pub const CU_LAUNCH_GRID_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync as u32,
    );

    /// Callback identifier for `cuEventCreate`.
    pub const CU_EVENT_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventCreate as u32);

    /// Callback identifier for `cuEventRecord`.
    pub const CU_EVENT_RECORD: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventRecord as u32);

    /// Callback identifier for `cuEventQuery`.
    pub const CU_EVENT_QUERY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventQuery as u32);

    /// Callback identifier for `cuEventSynchronize`.
    pub const CU_EVENT_SYNCHRONIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize as u32,
    );

    /// Callback identifier for `cuEventDestroy`.
    pub const CU_EVENT_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventDestroy as u32);

    /// Callback identifier for `cuEventElapsedTime`.
    pub const CU_EVENT_ELAPSED_TIME: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime as u32,
    );

    /// Callback identifier for `cuStreamCreate`.
    pub const CU_STREAM_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamCreate as u32);

    /// Callback identifier for `cuStreamQuery`.
    pub const CU_STREAM_QUERY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamQuery as u32);

    /// Callback identifier for `cuStreamSynchronize`.
    pub const CU_STREAM_SYNCHRONIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize as u32,
    );

    /// Callback identifier for `cuStreamDestroy`.
    pub const CU_STREAM_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy as u32);

    /// Callback identifier for `cuGraphicsUnregisterResource`.
    pub const CU_GRAPHICS_UNREGISTER_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnregisterResource
            as u32,
    );

    /// Callback identifier for `cuGraphicsSubResourceGetMappedArray`.
    pub const CU_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsSubResourceGetMappedArray as u32);

    /// Callback identifier for `cuGraphicsResourceGetMappedPointer`.
    pub const CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedPointer as u32);

    /// Callback identifier for `cu64GraphicsResourceGetMappedPointer`.
    pub const CU64_GRAPHICS_RESOURCE_GET_MAPPED_POINTER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64GraphicsResourceGetMappedPointer as u32);

    /// Callback identifier for `cuGraphicsResourceSetMapFlags`.
    pub const CU_GRAPHICS_RESOURCE_SET_MAP_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceSetMapFlags
            as u32,
    );

    /// Callback identifier for `cuGraphicsMapResources`.
    pub const CU_GRAPHICS_MAP_RESOURCES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsMapResources
            as u32,
    );

    /// Callback identifier for `cuGraphicsUnmapResources`.
    pub const CU_GRAPHICS_UNMAP_RESOURCES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnmapResources
            as u32,
    );

    /// Callback identifier for `cuGetExportTable`.
    pub const CU_GET_EXPORT_TABLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGetExportTable as u32,
    );

    /// Callback identifier for `cuCtxSetLimit`.
    pub const CU_CTX_SET_LIMIT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSetLimit as u32);

    /// Callback identifier for `cuCtxGetLimit`.
    pub const CU_CTX_GET_LIMIT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetLimit as u32);

    /// Callback identifier for `cuD3D10GetDevice`.
    pub const CU_D3_D10_GET_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDevice as u32,
    );

    /// Callback identifier for `cuD3D10CtxCreate`.
    pub const CU_D3_D10_CTX_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreate as u32,
    );

    /// Callback identifier for `cuGraphicsD3D10RegisterResource`.
    pub const CU_GRAPHICS_D3_D10_REGISTER_RESOURCE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D10RegisterResource as u32);

    /// Callback identifier for `cuD3D10RegisterResource`.
    pub const CU_D3_D10_REGISTER_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10RegisterResource
            as u32,
    );

    /// Callback identifier for `cuD3D10UnregisterResource`.
    pub const CU_D3_D10_UNREGISTER_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10UnregisterResource
            as u32,
    );

    /// Callback identifier for `cuD3D10MapResources`.
    pub const CU_D3_D10_MAP_RESOURCES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10MapResources as u32,
    );

    /// Callback identifier for `cuD3D10UnmapResources`.
    pub const CU_D3_D10_UNMAP_RESOURCES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10UnmapResources as u32,
    );

    /// Callback identifier for `cuD3D10ResourceSetMapFlags`.
    pub const CU_D3_D10_RESOURCE_SET_MAP_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceSetMapFlags
            as u32,
    );

    /// Callback identifier for `cuD3D10ResourceGetMappedArray`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedArray
            as u32,
    );

    /// Callback identifier for `cuD3D10ResourceGetMappedPointer`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_POINTER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPointer as u32);

    /// Callback identifier for `cuD3D10ResourceGetMappedSize`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_SIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedSize
            as u32,
    );

    /// Callback identifier for `cuD3D10ResourceGetMappedPitch`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_PITCH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPitch
            as u32,
    );

    /// Callback identifier for `cuD3D10ResourceGetSurfaceDimensions`.
    pub const CU_D3_D10_RESOURCE_GET_SURFACE_DIMENSIONS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetSurfaceDimensions as u32);

    /// Callback identifier for `cuD3D11GetDevice`.
    pub const CU_D3_D11_GET_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDevice as u32,
    );

    /// Callback identifier for `cuD3D11CtxCreate`.
    pub const CU_D3_D11_CTX_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreate as u32,
    );

    /// Callback identifier for `cuGraphicsD3D11RegisterResource`.
    pub const CU_GRAPHICS_D3_D11_REGISTER_RESOURCE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D11RegisterResource as u32);

    /// Callback identifier for `cuD3D9GetDevice`.
    pub const CU_D3_D9_GET_DEVICE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDevice as u32);

    /// Callback identifier for `cuD3D9CtxCreate`.
    pub const CU_D3_D9_CTX_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreate as u32);

    /// Callback identifier for `cuGraphicsD3D9RegisterResource`.
    pub const CU_GRAPHICS_D3_D9_REGISTER_RESOURCE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsD3D9RegisterResource as u32);

    /// Callback identifier for `cuD3D9GetDirect3DDevice`.
    pub const CU_D3_D9_GET_DIRECT3_D_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDirect3DDevice
            as u32,
    );

    /// Callback identifier for `cuD3D9RegisterResource`.
    pub const CU_D3_D9_REGISTER_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9RegisterResource
            as u32,
    );

    /// Callback identifier for `cuD3D9UnregisterResource`.
    pub const CU_D3_D9_UNREGISTER_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9UnregisterResource
            as u32,
    );

    /// Callback identifier for `cuD3D9MapResources`.
    pub const CU_D3_D9_MAP_RESOURCES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9MapResources as u32,
    );

    /// Callback identifier for `cuD3D9UnmapResources`.
    pub const CU_D3_D9_UNMAP_RESOURCES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapResources as u32,
    );

    /// Callback identifier for `cuD3D9ResourceSetMapFlags`.
    pub const CU_D3_D9_RESOURCE_SET_MAP_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceSetMapFlags
            as u32,
    );

    /// Callback identifier for `cuD3D9ResourceGetSurfaceDimensions`.
    pub const CU_D3_D9_RESOURCE_GET_SURFACE_DIMENSIONS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetSurfaceDimensions as u32);

    /// Callback identifier for `cuD3D9ResourceGetMappedArray`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedArray
            as u32,
    );

    /// Callback identifier for `cuD3D9ResourceGetMappedPointer`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_POINTER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPointer as u32);

    /// Callback identifier for `cuD3D9ResourceGetMappedSize`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_SIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedSize
            as u32,
    );

    /// Callback identifier for `cuD3D9ResourceGetMappedPitch`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_PITCH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPitch
            as u32,
    );

    /// Callback identifier for `cuD3D9Begin`.
    pub const CU_D3_D9_BEGIN: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9Begin as u32);

    /// Callback identifier for `cuD3D9End`.
    pub const CU_D3_D9_END: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9End as u32);

    /// Callback identifier for `cuD3D9RegisterVertexBuffer`.
    pub const CU_D3_D9_REGISTER_VERTEX_BUFFER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9RegisterVertexBuffer
            as u32,
    );

    /// Callback identifier for `cuD3D9MapVertexBuffer`.
    pub const CU_D3_D9_MAP_VERTEX_BUFFER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer as u32,
    );

    /// Callback identifier for `cuD3D9UnmapVertexBuffer`.
    pub const CU_D3_D9_UNMAP_VERTEX_BUFFER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9UnmapVertexBuffer
            as u32,
    );

    /// Callback identifier for `cuD3D9UnregisterVertexBuffer`.
    pub const CU_D3_D9_UNREGISTER_VERTEX_BUFFER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9UnregisterVertexBuffer
            as u32,
    );

    /// Callback identifier for `cuGLCtxCreate`.
    pub const CU_GL_CTX_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLCtxCreate as u32);

    /// Callback identifier for `cuGraphicsGLRegisterBuffer`.
    pub const CU_GRAPHICS_GL_REGISTER_BUFFER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsGLRegisterBuffer
            as u32,
    );

    /// Callback identifier for `cuGraphicsGLRegisterImage`.
    pub const CU_GRAPHICS_GL_REGISTER_IMAGE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsGLRegisterImage
            as u32,
    );

    /// Callback identifier for `cuWGLGetDevice`.
    pub const CU_WGL_GET_DEVICE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuWGLGetDevice as u32);

    /// Callback identifier for `cuGLInit`.
    pub const CU_GL_INIT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLInit as u32);

    /// Callback identifier for `cuGLRegisterBufferObject`.
    pub const CU_GL_REGISTER_BUFFER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLRegisterBufferObject
            as u32,
    );

    /// Callback identifier for `cuGLMapBufferObject`.
    pub const CU_GL_MAP_BUFFER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject as u32,
    );

    /// Callback identifier for `cuGLUnmapBufferObject`.
    pub const CU_GL_UNMAP_BUFFER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObject as u32,
    );

    /// Callback identifier for `cuGLUnregisterBufferObject`.
    pub const CU_GL_UNREGISTER_BUFFER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLUnregisterBufferObject
            as u32,
    );

    /// Callback identifier for `cuGLSetBufferObjectMapFlags`.
    pub const CU_GL_SET_BUFFER_OBJECT_MAP_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLSetBufferObjectMapFlags
            as u32,
    );

    /// Callback identifier for `cuGLMapBufferObjectAsync`.
    pub const CU_GL_MAP_BUFFER_OBJECT_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync
            as u32,
    );

    /// Callback identifier for `cuGLUnmapBufferObjectAsync`.
    pub const CU_GL_UNMAP_BUFFER_OBJECT_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLUnmapBufferObjectAsync
            as u32,
    );

    /// Callback identifier for `cuVDPAUGetDevice`.
    pub const CU_VDPAU_GET_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuVDPAUGetDevice as u32,
    );

    /// Callback identifier for `cuVDPAUCtxCreate`.
    pub const CU_VDPAU_CTX_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuVDPAUCtxCreate as u32,
    );

    /// Callback identifier for `cuGraphicsVDPAURegisterVideoSurface`.
    pub const CU_GRAPHICS_VDPAU_REGISTER_VIDEO_SURFACE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsVDPAURegisterVideoSurface as u32);

    /// Callback identifier for `cuGraphicsVDPAURegisterOutputSurface`.
    pub const CU_GRAPHICS_VDPAU_REGISTER_OUTPUT_SURFACE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsVDPAURegisterOutputSurface as u32);

    /// Callback identifier for `cuModuleGetSurfRef`.
    pub const CU_MODULE_GET_SURF_REF: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetSurfRef as u32,
    );

    /// Callback identifier for `cuSurfRefCreate`.
    pub const CU_SURF_REF_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfRefCreate as u32);

    /// Callback identifier for `cuSurfRefDestroy`.
    pub const CU_SURF_REF_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfRefDestroy as u32,
    );

    /// Callback identifier for `cuSurfRefSetFormat`.
    pub const CU_SURF_REF_SET_FORMAT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfRefSetFormat as u32,
    );

    /// Callback identifier for `cuSurfRefSetArray`.
    pub const CU_SURF_REF_SET_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfRefSetArray as u32,
    );

    /// Callback identifier for `cuSurfRefGetFormat`.
    pub const CU_SURF_REF_GET_FORMAT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfRefGetFormat as u32,
    );

    /// Callback identifier for `cuSurfRefGetArray`.
    pub const CU_SURF_REF_GET_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfRefGetArray as u32,
    );

    /// Callback identifier for `cu64DeviceTotalMem`.
    pub const CU64_DEVICE_TOTAL_MEM: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64DeviceTotalMem as u32,
    );

    /// Callback identifier for `cu64D3D10ResourceGetMappedPointer`.
    pub const CU64_D3_D10_RESOURCE_GET_MAPPED_POINTER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedPointer as u32);

    /// Callback identifier for `cu64D3D10ResourceGetMappedSize`.
    pub const CU64_D3_D10_RESOURCE_GET_MAPPED_SIZE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedSize as u32);

    /// Callback identifier for `cu64D3D10ResourceGetMappedPitch`.
    pub const CU64_D3_D10_RESOURCE_GET_MAPPED_PITCH: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetMappedPitch as u32);

    /// Callback identifier for `cu64D3D10ResourceGetSurfaceDimensions`.
    pub const CU64_D3_D10_RESOURCE_GET_SURFACE_DIMENSIONS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D10ResourceGetSurfaceDimensions as u32);

    /// Callback identifier for `cu64D3D9ResourceGetSurfaceDimensions`.
    pub const CU64_D3_D9_RESOURCE_GET_SURFACE_DIMENSIONS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetSurfaceDimensions as u32);

    /// Callback identifier for `cu64D3D9ResourceGetMappedPointer`.
    pub const CU64_D3_D9_RESOURCE_GET_MAPPED_POINTER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedPointer as u32);

    /// Callback identifier for `cu64D3D9ResourceGetMappedSize`.
    pub const CU64_D3_D9_RESOURCE_GET_MAPPED_SIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedSize
            as u32,
    );

    /// Callback identifier for `cu64D3D9ResourceGetMappedPitch`.
    pub const CU64_D3_D9_RESOURCE_GET_MAPPED_PITCH: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D9ResourceGetMappedPitch as u32);

    /// Callback identifier for `cu64D3D9MapVertexBuffer`.
    pub const CU64_D3_D9_MAP_VERTEX_BUFFER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64D3D9MapVertexBuffer
            as u32,
    );

    /// Callback identifier for `cu64GLMapBufferObject`.
    pub const CU64_GL_MAP_BUFFER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObject as u32,
    );

    /// Callback identifier for `cu64GLMapBufferObjectAsync`.
    pub const CU64_GL_MAP_BUFFER_OBJECT_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObjectAsync
            as u32,
    );

    /// Callback identifier for `cuD3D11GetDevices`.
    pub const CU_D3_D11_GET_DEVICES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDevices as u32,
    );

    /// Callback identifier for `cuD3D11CtxCreateOnDevice`.
    pub const CU_D3_D11_CTX_CREATE_ON_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreateOnDevice
            as u32,
    );

    /// Callback identifier for `cuD3D10GetDevices`.
    pub const CU_D3_D10_GET_DEVICES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDevices as u32,
    );

    /// Callback identifier for `cuD3D10CtxCreateOnDevice`.
    pub const CU_D3_D10_CTX_CREATE_ON_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreateOnDevice
            as u32,
    );

    /// Callback identifier for `cuD3D9GetDevices`.
    pub const CU_D3_D9_GET_DEVICES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9GetDevices as u32,
    );

    /// Callback identifier for `cuD3D9CtxCreateOnDevice`.
    pub const CU_D3_D9_CTX_CREATE_ON_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreateOnDevice
            as u32,
    );

    /// Callback identifier for `cu64MemHostAlloc`.
    pub const CU64_MEM_HOST_ALLOC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemHostAlloc as u32,
    );

    /// Callback identifier for `cuMemsetD8Async`.
    pub const CU_MEMSET_D8_ASYNC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async as u32);

    /// Callback identifier for `cu64MemsetD8Async`.
    pub const CU64_MEMSET_D8_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD8Async as u32,
    );

    /// Callback identifier for `cuMemsetD16Async`.
    pub const CU_MEMSET_D16_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async as u32,
    );

    /// Callback identifier for `cu64MemsetD16Async`.
    pub const CU64_MEMSET_D16_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD16Async as u32,
    );

    /// Callback identifier for `cuMemsetD32Async`.
    pub const CU_MEMSET_D32_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async as u32,
    );

    /// Callback identifier for `cu64MemsetD32Async`.
    pub const CU64_MEMSET_D32_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD32Async as u32,
    );

    /// Callback identifier for `cuMemsetD2D8Async`.
    pub const CU_MEMSET_D2_D8_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async as u32,
    );

    /// Callback identifier for `cu64MemsetD2D8Async`.
    pub const CU64_MEMSET_D2_D8_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D8Async as u32,
    );

    /// Callback identifier for `cuMemsetD2D16Async`.
    pub const CU_MEMSET_D2_D16_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async as u32,
    );

    /// Callback identifier for `cu64MemsetD2D16Async`.
    pub const CU64_MEMSET_D2_D16_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D16Async as u32,
    );

    /// Callback identifier for `cuMemsetD2D32Async`.
    pub const CU_MEMSET_D2_D32_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async as u32,
    );

    /// Callback identifier for `cu64MemsetD2D32Async`.
    pub const CU64_MEMSET_D2_D32_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64MemsetD2D32Async as u32,
    );

    /// Callback identifier for `cu64ArrayCreate`.
    pub const CU64_ARRAY_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64ArrayCreate as u32);

    /// Callback identifier for `cu64ArrayGetDescriptor`.
    pub const CU64_ARRAY_GET_DESCRIPTOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64ArrayGetDescriptor
            as u32,
    );

    /// Callback identifier for `cu64Array3DCreate`.
    pub const CU64_ARRAY3_D_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Array3DCreate as u32,
    );

    /// Callback identifier for `cu64Array3DGetDescriptor`.
    pub const CU64_ARRAY3_D_GET_DESCRIPTOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Array3DGetDescriptor
            as u32,
    );

    /// Callback identifier for `cu64Memcpy2D`.
    pub const CU64_MEMCPY2_D: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2D as u32);

    /// Callback identifier for `cu64Memcpy2DUnaligned`.
    pub const CU64_MEMCPY2_D_UNALIGNED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2DUnaligned as u32,
    );

    /// Callback identifier for `cu64Memcpy2DAsync`.
    pub const CU64_MEMCPY2_D_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cu64Memcpy2DAsync as u32,
    );

    /// Callback identifier for `cuCtxCreate_v2`.
    pub const CU_CTX_CREATE_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2 as u32);

    /// Callback identifier for `cuD3D10CtxCreate_v2`.
    pub const CU_D3_D10_CTX_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10CtxCreate_v2 as u32,
    );

    /// Callback identifier for `cuD3D11CtxCreate_v2`.
    pub const CU_D3_D11_CTX_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D11CtxCreate_v2 as u32,
    );

    /// Callback identifier for `cuD3D9CtxCreate_v2`.
    pub const CU_D3_D9_CTX_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9CtxCreate_v2 as u32,
    );

    /// Callback identifier for `cuGLCtxCreate_v2`.
    pub const CU_GL_CTX_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLCtxCreate_v2 as u32,
    );

    /// Callback identifier for `cuVDPAUCtxCreate_v2`.
    pub const CU_VDPAU_CTX_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuVDPAUCtxCreate_v2 as u32,
    );

    /// Callback identifier for `cuModuleGetGlobal_v2`.
    pub const CU_MODULE_GET_GLOBAL_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2 as u32,
    );

    /// Callback identifier for `cuMemGetInfo_v2`.
    pub const CU_MEM_GET_INFO_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetInfo_v2 as u32);

    /// Callback identifier for `cuMemAlloc_v2`.
    pub const CU_MEM_ALLOC_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2 as u32);

    /// Callback identifier for `cuMemAllocPitch_v2`.
    pub const CU_MEM_ALLOC_PITCH_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2 as u32,
    );

    /// Callback identifier for `cuMemFree_v2`.
    pub const CU_MEM_FREE_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2 as u32);

    /// Callback identifier for `cuMemGetAddressRange_v2`.
    pub const CU_MEM_GET_ADDRESS_RANGE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetAddressRange_v2
            as u32,
    );

    /// Callback identifier for `cuMemHostGetDevicePointer_v2`.
    pub const CU_MEM_HOST_GET_DEVICE_POINTER_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostGetDevicePointer_v2
            as u32,
    );

    /// Callback identifier for `cuMemcpy_v2`.
    pub const CU_MEMCPY_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2 as u32);

    /// Callback identifier for `cuMemsetD8_v2`.
    pub const CU_MEMSET_D8_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2 as u32);

    /// Callback identifier for `cuMemsetD16_v2`.
    pub const CU_MEMSET_D16_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2 as u32);

    /// Callback identifier for `cuMemsetD32_v2`.
    pub const CU_MEMSET_D32_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2 as u32);

    /// Callback identifier for `cuMemsetD2D8_v2`.
    pub const CU_MEMSET_D2_D8_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2 as u32);

    /// Callback identifier for `cuMemsetD2D16_v2`.
    pub const CU_MEMSET_D2_D16_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2 as u32,
    );

    /// Callback identifier for `cuMemsetD2D32_v2`.
    pub const CU_MEMSET_D2_D32_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2 as u32,
    );

    /// Callback identifier for `cuTexRefSetAddress_v2`.
    pub const CU_TEX_REF_SET_ADDRESS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress_v2 as u32,
    );

    /// Callback identifier for `cuTexRefSetAddress2D_v2`.
    pub const CU_TEX_REF_SET_ADDRESS2_D_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D_v2
            as u32,
    );

    /// Callback identifier for `cuTexRefGetAddress_v2`.
    pub const CU_TEX_REF_GET_ADDRESS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetAddress_v2 as u32,
    );

    /// Callback identifier for `cuGraphicsResourceGetMappedPointer_v2`.
    pub const CU_GRAPHICS_RESOURCE_GET_MAPPED_POINTER_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedPointer_v2 as u32);

    /// Callback identifier for `cuDeviceTotalMem_v2`.
    pub const CU_DEVICE_TOTAL_MEM_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceTotalMem_v2 as u32,
    );

    /// Callback identifier for `cuD3D10ResourceGetMappedPointer_v2`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_POINTER_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPointer_v2 as u32);

    /// Callback identifier for `cuD3D10ResourceGetMappedSize_v2`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_SIZE_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedSize_v2 as u32);

    /// Callback identifier for `cuD3D10ResourceGetMappedPitch_v2`.
    pub const CU_D3_D10_RESOURCE_GET_MAPPED_PITCH_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetMappedPitch_v2 as u32);

    /// Callback identifier for `cuD3D10ResourceGetSurfaceDimensions_v2`.
    pub const CU_D3_D10_RESOURCE_GET_SURFACE_DIMENSIONS_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10ResourceGetSurfaceDimensions_v2 as u32);

    /// Callback identifier for `cuD3D9ResourceGetSurfaceDimensions_v2`.
    pub const CU_D3_D9_RESOURCE_GET_SURFACE_DIMENSIONS_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetSurfaceDimensions_v2 as u32);

    /// Callback identifier for `cuD3D9ResourceGetMappedPointer_v2`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_POINTER_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPointer_v2 as u32);

    /// Callback identifier for `cuD3D9ResourceGetMappedSize_v2`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_SIZE_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedSize_v2 as u32);

    /// Callback identifier for `cuD3D9ResourceGetMappedPitch_v2`.
    pub const CU_D3_D9_RESOURCE_GET_MAPPED_PITCH_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9ResourceGetMappedPitch_v2 as u32);

    /// Callback identifier for `cuD3D9MapVertexBuffer_v2`.
    pub const CU_D3_D9_MAP_VERTEX_BUFFER_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D9MapVertexBuffer_v2
            as u32,
    );

    /// Callback identifier for `cuGLMapBufferObject_v2`.
    pub const CU_GL_MAP_BUFFER_OBJECT_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2
            as u32,
    );

    /// Callback identifier for `cuGLMapBufferObjectAsync_v2`.
    pub const CU_GL_MAP_BUFFER_OBJECT_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2
            as u32,
    );

    /// Callback identifier for `cuMemHostAlloc_v2`.
    pub const CU_MEM_HOST_ALLOC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2 as u32,
    );

    /// Callback identifier for `cuArrayCreate_v2`.
    pub const CU_ARRAY_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2 as u32,
    );

    /// Callback identifier for `cuArrayGetDescriptor_v2`.
    pub const CU_ARRAY_GET_DESCRIPTOR_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor_v2
            as u32,
    );

    /// Callback identifier for `cuArray3DCreate_v2`.
    pub const CU_ARRAY3_D_CREATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2 as u32,
    );

    /// Callback identifier for `cuArray3DGetDescriptor_v2`.
    pub const CU_ARRAY3_D_GET_DESCRIPTOR_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor_v2
            as u32,
    );

    /// Callback identifier for `cuMemcpyHtoD_v2`.
    pub const CU_MEMCPY_HTO_D_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 as u32);

    /// Callback identifier for `cuMemcpyHtoDAsync_v2`.
    pub const CU_MEMCPY_HTO_D_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyDtoH_v2`.
    pub const CU_MEMCPY_DTO_H_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 as u32);

    /// Callback identifier for `cuMemcpyDtoHAsync_v2`.
    pub const CU_MEMCPY_DTO_H_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyDtoD_v2`.
    pub const CU_MEMCPY_DTO_D_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 as u32);

    /// Callback identifier for `cuMemcpyDtoDAsync_v2`.
    pub const CU_MEMCPY_DTO_D_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyAtoH_v2`.
    pub const CU_MEMCPY_ATO_H_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2 as u32);

    /// Callback identifier for `cuMemcpyAtoHAsync_v2`.
    pub const CU_MEMCPY_ATO_H_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyAtoD_v2`.
    pub const CU_MEMCPY_ATO_D_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2 as u32);

    /// Callback identifier for `cuMemcpyDtoA_v2`.
    pub const CU_MEMCPY_DTO_A_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2 as u32);

    /// Callback identifier for `cuMemcpyAtoA_v2`.
    pub const CU_MEMCPY_ATO_A_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2 as u32);

    /// Callback identifier for `cuMemcpy2D_v2`.
    pub const CU_MEMCPY2_D_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2 as u32);

    /// Callback identifier for `cuMemcpy2DUnaligned_v2`.
    pub const CU_MEMCPY2_D_UNALIGNED_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2
            as u32,
    );

    /// Callback identifier for `cuMemcpy2DAsync_v2`.
    pub const CU_MEMCPY2_D_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpy3D_v2`.
    pub const CU_MEMCPY3_D_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2 as u32);

    /// Callback identifier for `cuMemcpy3DAsync_v2`.
    pub const CU_MEMCPY3_D_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyHtoA_v2`.
    pub const CU_MEMCPY_HTO_A_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2 as u32);

    /// Callback identifier for `cuMemcpyHtoAAsync_v2`.
    pub const CU_MEMCPY_HTO_A_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemAllocHost_v2`.
    pub const CU_MEM_ALLOC_HOST_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2 as u32,
    );

    /// Callback identifier for `cuStreamWaitEvent`.
    pub const CU_STREAM_WAIT_EVENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent as u32,
    );

    /// Callback identifier for `cuCtxGetApiVersion`.
    pub const CU_CTX_GET_API_VERSION: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetApiVersion as u32,
    );

    /// Callback identifier for `cuD3D10GetDirect3DDevice`.
    pub const CU_D3_D10_GET_DIRECT3_D_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D10GetDirect3DDevice
            as u32,
    );

    /// Callback identifier for `cuD3D11GetDirect3DDevice`.
    pub const CU_D3_D11_GET_DIRECT3_D_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuD3D11GetDirect3DDevice
            as u32,
    );

    /// Callback identifier for `cuCtxGetCacheConfig`.
    pub const CU_CTX_GET_CACHE_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetCacheConfig as u32,
    );

    /// Callback identifier for `cuCtxSetCacheConfig`.
    pub const CU_CTX_SET_CACHE_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSetCacheConfig as u32,
    );

    /// Callback identifier for `cuMemHostRegister`.
    pub const CU_MEM_HOST_REGISTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister as u32,
    );

    /// Callback identifier for `cuMemHostUnregister`.
    pub const CU_MEM_HOST_UNREGISTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister as u32,
    );

    /// Callback identifier for `cuCtxSetCurrent`.
    pub const CU_CTX_SET_CURRENT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent as u32);

    /// Callback identifier for `cuCtxGetCurrent`.
    pub const CU_CTX_GET_CURRENT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetCurrent as u32);

    /// Callback identifier for `cuMemcpy`.
    pub const CU_MEMCPY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy as u32);

    /// Callback identifier for `cuMemcpyAsync`.
    pub const CU_MEMCPY_ASYNC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync as u32);

    /// Callback identifier for `cuLaunchKernel`.
    pub const CU_LAUNCH_KERNEL: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel as u32);

    /// Callback identifier for `cuProfilerStart`.
    pub const CU_PROFILER_START: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuProfilerStart as u32);

    /// Callback identifier for `cuProfilerStop`.
    pub const CU_PROFILER_STOP: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuProfilerStop as u32);

    /// Callback identifier for `cuPointerGetAttribute`.
    pub const CU_POINTER_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttribute as u32,
    );

    /// Callback identifier for `cuProfilerInitialize`.
    pub const CU_PROFILER_INITIALIZE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuProfilerInitialize as u32,
    );

    /// Callback identifier for `cuDeviceCanAccessPeer`.
    pub const CU_DEVICE_CAN_ACCESS_PEER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceCanAccessPeer as u32,
    );

    /// Callback identifier for `cuCtxEnablePeerAccess`.
    pub const CU_CTX_ENABLE_PEER_ACCESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxEnablePeerAccess as u32,
    );

    /// Callback identifier for `cuCtxDisablePeerAccess`.
    pub const CU_CTX_DISABLE_PEER_ACCESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxDisablePeerAccess
            as u32,
    );

    /// Callback identifier for `cuMemPeerRegister`.
    pub const CU_MEM_PEER_REGISTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPeerRegister as u32,
    );

    /// Callback identifier for `cuMemPeerUnregister`.
    pub const CU_MEM_PEER_UNREGISTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPeerUnregister as u32,
    );

    /// Callback identifier for `cuMemPeerGetDevicePointer`.
    pub const CU_MEM_PEER_GET_DEVICE_POINTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPeerGetDevicePointer
            as u32,
    );

    /// Callback identifier for `cuMemcpyPeer`.
    pub const CU_MEMCPY_PEER: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer as u32);

    /// Callback identifier for `cuMemcpyPeerAsync`.
    pub const CU_MEMCPY_PEER_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync as u32,
    );

    /// Callback identifier for `cuMemcpy3DPeer`.
    pub const CU_MEMCPY3_D_PEER: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer as u32);

    /// Callback identifier for `cuMemcpy3DPeerAsync`.
    pub const CU_MEMCPY3_D_PEER_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync as u32,
    );

    /// Callback identifier for `cuCtxDestroy_v2`.
    pub const CU_CTX_DESTROY_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2 as u32);

    /// Callback identifier for `cuCtxPushCurrent_v2`.
    pub const CU_CTX_PUSH_CURRENT_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxPushCurrent_v2 as u32,
    );

    /// Callback identifier for `cuCtxPopCurrent_v2`.
    pub const CU_CTX_POP_CURRENT_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxPopCurrent_v2 as u32,
    );

    /// Callback identifier for `cuEventDestroy_v2`.
    pub const CU_EVENT_DESTROY_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventDestroy_v2 as u32,
    );

    /// Callback identifier for `cuStreamDestroy_v2`.
    pub const CU_STREAM_DESTROY_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy_v2 as u32,
    );

    /// Callback identifier for `cuTexRefSetAddress2D_v3`.
    pub const CU_TEX_REF_SET_ADDRESS2_D_V3: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetAddress2D_v3
            as u32,
    );

    /// Callback identifier for `cuIpcGetMemHandle`.
    pub const CU_IPC_GET_MEM_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuIpcGetMemHandle as u32,
    );

    /// Callback identifier for `cuIpcOpenMemHandle`.
    pub const CU_IPC_OPEN_MEM_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle as u32,
    );

    /// Callback identifier for `cuIpcCloseMemHandle`.
    pub const CU_IPC_CLOSE_MEM_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuIpcCloseMemHandle as u32,
    );

    /// Callback identifier for `cuDeviceGetByPCIBusId`.
    pub const CU_DEVICE_GET_BY_PCI_BUS_ID: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetByPCIBusId as u32,
    );

    /// Callback identifier for `cuDeviceGetPCIBusId`.
    pub const CU_DEVICE_GET_PCI_BUS_ID: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetPCIBusId as u32,
    );

    /// Callback identifier for `cuGLGetDevices`.
    pub const CU_GL_GET_DEVICES: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLGetDevices as u32);

    /// Callback identifier for `cuIpcGetEventHandle`.
    pub const CU_IPC_GET_EVENT_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuIpcGetEventHandle as u32,
    );

    /// Callback identifier for `cuIpcOpenEventHandle`.
    pub const CU_IPC_OPEN_EVENT_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuIpcOpenEventHandle as u32,
    );

    /// Callback identifier for `cuCtxSetSharedMemConfig`.
    pub const CU_CTX_SET_SHARED_MEM_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSetSharedMemConfig
            as u32,
    );

    /// Callback identifier for `cuCtxGetSharedMemConfig`.
    pub const CU_CTX_GET_SHARED_MEM_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetSharedMemConfig
            as u32,
    );

    /// Callback identifier for `cuFuncSetSharedMemConfig`.
    pub const CU_FUNC_SET_SHARED_MEM_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncSetSharedMemConfig
            as u32,
    );

    /// Callback identifier for `cuTexObjectCreate`.
    pub const CU_TEX_OBJECT_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexObjectCreate as u32,
    );

    /// Callback identifier for `cuTexObjectDestroy`.
    pub const CU_TEX_OBJECT_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexObjectDestroy as u32,
    );

    /// Callback identifier for `cuTexObjectGetResourceDesc`.
    pub const CU_TEX_OBJECT_GET_RESOURCE_DESC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetResourceDesc
            as u32,
    );

    /// Callback identifier for `cuTexObjectGetTextureDesc`.
    pub const CU_TEX_OBJECT_GET_TEXTURE_DESC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetTextureDesc
            as u32,
    );

    /// Callback identifier for `cuSurfObjectCreate`.
    pub const CU_SURF_OBJECT_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfObjectCreate as u32,
    );

    /// Callback identifier for `cuSurfObjectDestroy`.
    pub const CU_SURF_OBJECT_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfObjectDestroy as u32,
    );

    /// Callback identifier for `cuSurfObjectGetResourceDesc`.
    pub const CU_SURF_OBJECT_GET_RESOURCE_DESC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSurfObjectGetResourceDesc
            as u32,
    );

    /// Callback identifier for `cuStreamAddCallback`.
    pub const CU_STREAM_ADD_CALLBACK: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamAddCallback as u32,
    );

    /// Callback identifier for `cuMipmappedArrayCreate`.
    pub const CU_MIPMAPPED_ARRAY_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayCreate
            as u32,
    );

    /// Callback identifier for `cuMipmappedArrayGetLevel`.
    pub const CU_MIPMAPPED_ARRAY_GET_LEVEL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetLevel
            as u32,
    );

    /// Callback identifier for `cuMipmappedArrayDestroy`.
    pub const CU_MIPMAPPED_ARRAY_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayDestroy
            as u32,
    );

    /// Callback identifier for `cuTexRefSetMipmappedArray`.
    pub const CU_TEX_REF_SET_MIPMAPPED_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmappedArray
            as u32,
    );

    /// Callback identifier for `cuTexRefSetMipmapFilterMode`.
    pub const CU_TEX_REF_SET_MIPMAP_FILTER_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapFilterMode
            as u32,
    );

    /// Callback identifier for `cuTexRefSetMipmapLevelBias`.
    pub const CU_TEX_REF_SET_MIPMAP_LEVEL_BIAS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapLevelBias
            as u32,
    );

    /// Callback identifier for `cuTexRefSetMipmapLevelClamp`.
    pub const CU_TEX_REF_SET_MIPMAP_LEVEL_CLAMP: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMipmapLevelClamp
            as u32,
    );

    /// Callback identifier for `cuTexRefSetMaxAnisotropy`.
    pub const CU_TEX_REF_SET_MAX_ANISOTROPY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetMaxAnisotropy
            as u32,
    );

    /// Callback identifier for `cuTexRefGetMipmappedArray`.
    pub const CU_TEX_REF_GET_MIPMAPPED_ARRAY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmappedArray
            as u32,
    );

    /// Callback identifier for `cuTexRefGetMipmapFilterMode`.
    pub const CU_TEX_REF_GET_MIPMAP_FILTER_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapFilterMode
            as u32,
    );

    /// Callback identifier for `cuTexRefGetMipmapLevelBias`.
    pub const CU_TEX_REF_GET_MIPMAP_LEVEL_BIAS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapLevelBias
            as u32,
    );

    /// Callback identifier for `cuTexRefGetMipmapLevelClamp`.
    pub const CU_TEX_REF_GET_MIPMAP_LEVEL_CLAMP: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMipmapLevelClamp
            as u32,
    );

    /// Callback identifier for `cuTexRefGetMaxAnisotropy`.
    pub const CU_TEX_REF_GET_MAX_ANISOTROPY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetMaxAnisotropy
            as u32,
    );

    /// Callback identifier for `cuGraphicsResourceGetMappedMipmappedArray`.
    pub const CU_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedMipmappedArray as u32);

    /// Callback identifier for `cuTexObjectGetResourceViewDesc`.
    pub const CU_TEX_OBJECT_GET_RESOURCE_VIEW_DESC: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexObjectGetResourceViewDesc as u32);

    /// Callback identifier for `cuLinkCreate`.
    pub const CU_LINK_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkCreate as u32);

    /// Callback identifier for `cuLinkAddData`.
    pub const CU_LINK_ADD_DATA: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkAddData as u32);

    /// Callback identifier for `cuLinkAddFile`.
    pub const CU_LINK_ADD_FILE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkAddFile as u32);

    /// Callback identifier for `cuLinkComplete`.
    pub const CU_LINK_COMPLETE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkComplete as u32);

    /// Callback identifier for `cuLinkDestroy`.
    pub const CU_LINK_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkDestroy as u32);

    /// Callback identifier for `cuStreamCreateWithPriority`.
    pub const CU_STREAM_CREATE_WITH_PRIORITY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamCreateWithPriority
            as u32,
    );

    /// Callback identifier for `cuStreamGetPriority`.
    pub const CU_STREAM_GET_PRIORITY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetPriority as u32,
    );

    /// Callback identifier for `cuStreamGetFlags`.
    pub const CU_STREAM_GET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetFlags as u32,
    );

    /// Callback identifier for `cuCtxGetStreamPriorityRange`.
    pub const CU_CTX_GET_STREAM_PRIORITY_RANGE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetStreamPriorityRange
            as u32,
    );

    /// Callback identifier for `cuMemAllocManaged`.
    pub const CU_MEM_ALLOC_MANAGED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged as u32,
    );

    /// Callback identifier for `cuGetErrorString`.
    pub const CU_GET_ERROR_STRING: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGetErrorString as u32,
    );

    /// Callback identifier for `cuGetErrorName`.
    pub const CU_GET_ERROR_NAME: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGetErrorName as u32);

    /// Callback identifier for `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
    pub const CU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessor as u32);

    /// Callback identifier for `cuCompilePtx`.
    pub const CU_COMPILE_PTX: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCompilePtx as u32);

    /// Callback identifier for `cuBinaryFree`.
    pub const CU_BINARY_FREE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuBinaryFree as u32);

    /// Callback identifier for `cuStreamAttachMemAsync`.
    pub const CU_STREAM_ATTACH_MEM_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamAttachMemAsync
            as u32,
    );

    /// Callback identifier for `cuPointerSetAttribute`.
    pub const CU_POINTER_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuPointerSetAttribute as u32,
    );

    /// Callback identifier for `cuMemHostRegister_v2`.
    pub const CU_MEM_HOST_REGISTER_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2 as u32,
    );

    /// Callback identifier for `cuGraphicsResourceSetMapFlags_v2`.
    pub const CU_GRAPHICS_RESOURCE_SET_MAP_FLAGS_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceSetMapFlags_v2 as u32);

    /// Callback identifier for `cuLinkCreate_v2`.
    pub const CU_LINK_CREATE_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkCreate_v2 as u32);

    /// Callback identifier for `cuLinkAddData_v2`.
    pub const CU_LINK_ADD_DATA_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkAddData_v2 as u32,
    );

    /// Callback identifier for `cuLinkAddFile_v2`.
    pub const CU_LINK_ADD_FILE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLinkAddFile_v2 as u32,
    );

    /// Callback identifier for `cuOccupancyMaxPotentialBlockSize`.
    pub const CU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialBlockSize as u32);

    /// Callback identifier for `cuGLGetDevices_v2`.
    pub const CU_GL_GET_DEVICES_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLGetDevices_v2 as u32,
    );

    /// Callback identifier for `cuDevicePrimaryCtxRetain`.
    pub const CU_DEVICE_PRIMARY_CTX_RETAIN: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRetain
            as u32,
    );

    /// Callback identifier for `cuDevicePrimaryCtxRelease`.
    pub const CU_DEVICE_PRIMARY_CTX_RELEASE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease
            as u32,
    );

    /// Callback identifier for `cuDevicePrimaryCtxSetFlags`.
    pub const CU_DEVICE_PRIMARY_CTX_SET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxSetFlags
            as u32,
    );

    /// Callback identifier for `cuDevicePrimaryCtxReset`.
    pub const CU_DEVICE_PRIMARY_CTX_RESET: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset
            as u32,
    );

    /// Callback identifier for `cuGraphicsEGLRegisterImage`.
    pub const CU_GRAPHICS_EGL_REGISTER_IMAGE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsEGLRegisterImage
            as u32,
    );

    /// Callback identifier for `cuCtxGetFlags`.
    pub const CU_CTX_GET_FLAGS: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetFlags as u32);

    /// Callback identifier for `cuDevicePrimaryCtxGetState`.
    pub const CU_DEVICE_PRIMARY_CTX_GET_STATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxGetState
            as u32,
    );

    /// Callback identifier for `cuEGLStreamConsumerConnect`.
    pub const CU_EGL_STREAM_CONSUMER_CONNECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerConnect
            as u32,
    );

    /// Callback identifier for `cuEGLStreamConsumerDisconnect`.
    pub const CU_EGL_STREAM_CONSUMER_DISCONNECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerDisconnect
            as u32,
    );

    /// Callback identifier for `cuEGLStreamConsumerAcquireFrame`.
    pub const CU_EGL_STREAM_CONSUMER_ACQUIRE_FRAME: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerAcquireFrame as u32);

    /// Callback identifier for `cuEGLStreamConsumerReleaseFrame`.
    pub const CU_EGL_STREAM_CONSUMER_RELEASE_FRAME: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerReleaseFrame as u32);

    /// Callback identifier for `cuMemcpyHtoD_v2_ptds`.
    pub const CU_MEMCPY_HTO_D_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyDtoH_v2_ptds`.
    pub const CU_MEMCPY_DTO_H_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyDtoD_v2_ptds`.
    pub const CU_MEMCPY_DTO_D_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyDtoA_v2_ptds`.
    pub const CU_MEMCPY_DTO_A_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyAtoD_v2_ptds`.
    pub const CU_MEMCPY_ATO_D_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyHtoA_v2_ptds`.
    pub const CU_MEMCPY_HTO_A_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyAtoH_v2_ptds`.
    pub const CU_MEMCPY_ATO_H_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpyAtoA_v2_ptds`.
    pub const CU_MEMCPY_ATO_A_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpy2D_v2_ptds`.
    pub const CU_MEMCPY2_D_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpy2DUnaligned_v2_ptds`.
    pub const CU_MEMCPY2_D_UNALIGNED_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2_ptds
            as u32,
    );

    /// Callback identifier for `cuMemcpy3D_v2_ptds`.
    pub const CU_MEMCPY3_D_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemcpy_ptds`.
    pub const CU_MEMCPY_PTDS: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy_ptds as u32);

    /// Callback identifier for `cuMemcpyPeer_ptds`.
    pub const CU_MEMCPY_PEER_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer_ptds as u32,
    );

    /// Callback identifier for `cuMemcpy3DPeer_ptds`.
    pub const CU_MEMCPY3_D_PEER_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeer_ptds as u32,
    );

    /// Callback identifier for `cuMemsetD8_v2_ptds`.
    pub const CU_MEMSET_D8_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemsetD16_v2_ptds`.
    pub const CU_MEMSET_D16_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemsetD32_v2_ptds`.
    pub const CU_MEMSET_D32_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemsetD2D8_v2_ptds`.
    pub const CU_MEMSET_D2_D8_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemsetD2D16_v2_ptds`.
    pub const CU_MEMSET_D2_D16_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2_ptds as u32,
    );

    /// Callback identifier for `cuMemsetD2D32_v2_ptds`.
    pub const CU_MEMSET_D2_D32_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2_ptds as u32,
    );

    /// Callback identifier for `cuGLMapBufferObject_v2_ptds`.
    pub const CU_GL_MAP_BUFFER_OBJECT_V2_PTDS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObject_v2_ptds
            as u32,
    );

    /// Callback identifier for `cuMemcpyAsync_ptsz`.
    pub const CU_MEMCPY_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz as u32,
    );

    /// Callback identifier for `cuMemcpyHtoAAsync_v2_ptsz`.
    pub const CU_MEMCPY_HTO_A_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpyAtoHAsync_v2_ptsz`.
    pub const CU_MEMCPY_ATO_H_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpyHtoDAsync_v2_ptsz`.
    pub const CU_MEMCPY_HTO_D_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpyDtoHAsync_v2_ptsz`.
    pub const CU_MEMCPY_DTO_H_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpyDtoDAsync_v2_ptsz`.
    pub const CU_MEMCPY_DTO_D_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpy2DAsync_v2_ptsz`.
    pub const CU_MEMCPY2_D_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpy3DAsync_v2_ptsz`.
    pub const CU_MEMCPY3_D_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpyPeerAsync_ptsz`.
    pub const CU_MEMCPY_PEER_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpy3DPeerAsync_ptsz`.
    pub const CU_MEMCPY3_D_PEER_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DPeerAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemsetD8Async_ptsz`.
    pub const CU_MEMSET_D8_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async_ptsz as u32,
    );

    /// Callback identifier for `cuMemsetD16Async_ptsz`.
    pub const CU_MEMSET_D16_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async_ptsz as u32,
    );

    /// Callback identifier for `cuMemsetD32Async_ptsz`.
    pub const CU_MEMSET_D32_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async_ptsz as u32,
    );

    /// Callback identifier for `cuMemsetD2D8Async_ptsz`.
    pub const CU_MEMSET_D2_D8_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemsetD2D16Async_ptsz`.
    pub const CU_MEMSET_D2_D16_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemsetD2D32Async_ptsz`.
    pub const CU_MEMSET_D2_D32_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamGetPriority_ptsz`.
    pub const CU_STREAM_GET_PRIORITY_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetPriority_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamGetFlags_ptsz`.
    pub const CU_STREAM_GET_FLAGS_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetFlags_ptsz as u32,
    );

    /// Callback identifier for `cuStreamWaitEvent_ptsz`.
    pub const CU_STREAM_WAIT_EVENT_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamAddCallback_ptsz`.
    pub const CU_STREAM_ADD_CALLBACK_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamAddCallback_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamAttachMemAsync_ptsz`.
    pub const CU_STREAM_ATTACH_MEM_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamAttachMemAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamQuery_ptsz`.
    pub const CU_STREAM_QUERY_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamQuery_ptsz as u32,
    );

    /// Callback identifier for `cuStreamSynchronize_ptsz`.
    pub const CU_STREAM_SYNCHRONIZE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz
            as u32,
    );

    /// Callback identifier for `cuEventRecord_ptsz`.
    pub const CU_EVENT_RECORD_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventRecord_ptsz as u32,
    );

    /// Callback identifier for `cuLaunchKernel_ptsz`.
    pub const CU_LAUNCH_KERNEL_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz as u32,
    );

    /// Callback identifier for `cuGraphicsMapResources_ptsz`.
    pub const CU_GRAPHICS_MAP_RESOURCES_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsMapResources_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphicsUnmapResources_ptsz`.
    pub const CU_GRAPHICS_UNMAP_RESOURCES_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsUnmapResources_ptsz
            as u32,
    );

    /// Callback identifier for `cuGLMapBufferObjectAsync_v2_ptsz`.
    pub const CU_GL_MAP_BUFFER_OBJECT_ASYNC_V2_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGLMapBufferObjectAsync_v2_ptsz as u32);

    /// Callback identifier for `cuEGLStreamProducerConnect`.
    pub const CU_EGL_STREAM_PRODUCER_CONNECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerConnect
            as u32,
    );

    /// Callback identifier for `cuEGLStreamProducerDisconnect`.
    pub const CU_EGL_STREAM_PRODUCER_DISCONNECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerDisconnect
            as u32,
    );

    /// Callback identifier for `cuEGLStreamProducerPresentFrame`.
    pub const CU_EGL_STREAM_PRODUCER_PRESENT_FRAME: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerPresentFrame as u32);

    /// Callback identifier for `cuGraphicsResourceGetMappedEglFrame`.
    pub const CU_GRAPHICS_RESOURCE_GET_MAPPED_EGL_FRAME: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphicsResourceGetMappedEglFrame as u32);

    /// Callback identifier for `cuPointerGetAttributes`.
    pub const CU_POINTER_GET_ATTRIBUTES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuPointerGetAttributes
            as u32,
    );

    /// Callback identifier for `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`.
    pub const CU_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_WITH_FLAGS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags as u32);

    /// Callback identifier for `cuOccupancyMaxPotentialBlockSizeWithFlags`.
    pub const CU_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE_WITH_FLAGS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialBlockSizeWithFlags as u32);

    /// Callback identifier for `cuEGLStreamProducerReturnFrame`.
    pub const CU_EGL_STREAM_PRODUCER_RETURN_FRAME: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamProducerReturnFrame as u32);

    /// Callback identifier for `cuDeviceGetP2PAttribute`.
    pub const CU_DEVICE_GET_P2_P_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetP2PAttribute
            as u32,
    );

    /// Callback identifier for `cuTexRefSetBorderColor`.
    pub const CU_TEX_REF_SET_BORDER_COLOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefSetBorderColor
            as u32,
    );

    /// Callback identifier for `cuTexRefGetBorderColor`.
    pub const CU_TEX_REF_GET_BORDER_COLOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTexRefGetBorderColor
            as u32,
    );

    /// Callback identifier for `cuMemAdvise`.
    pub const CU_MEM_ADVISE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAdvise as u32);

    /// Callback identifier for `cuStreamWaitValue32`.
    pub const CU_STREAM_WAIT_VALUE32: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32 as u32,
    );

    /// Callback identifier for `cuStreamWaitValue32_ptsz`.
    pub const CU_STREAM_WAIT_VALUE32_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamWriteValue32`.
    pub const CU_STREAM_WRITE_VALUE32: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32 as u32,
    );

    /// Callback identifier for `cuStreamWriteValue32_ptsz`.
    pub const CU_STREAM_WRITE_VALUE32_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamBatchMemOp`.
    pub const CU_STREAM_BATCH_MEM_OP: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp as u32,
    );

    /// Callback identifier for `cuStreamBatchMemOp_ptsz`.
    pub const CU_STREAM_BATCH_MEM_OP_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_ptsz
            as u32,
    );

    /// Callback identifier for `cuNNSetAllocator`.
    pub const CU_NN_SET_ALLOCATOR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuNNSetAllocator as u32,
    );

    /// Callback identifier for `cuMemPrefetchAsync`.
    pub const CU_MEM_PREFETCH_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync as u32,
    );

    /// Callback identifier for `cuMemPrefetchAsync_ptsz`.
    pub const CU_MEM_PREFETCH_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuEGLStreamConsumerConnectWithFlags`.
    pub const CU_EGL_STREAM_CONSUMER_CONNECT_WITH_FLAGS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEGLStreamConsumerConnectWithFlags as u32);

    /// Callback identifier for `cuMemRangeGetAttribute`.
    pub const CU_MEM_RANGE_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttribute
            as u32,
    );

    /// Callback identifier for `cuMemRangeGetAttributes`.
    pub const CU_MEM_RANGE_GET_ATTRIBUTES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemRangeGetAttributes
            as u32,
    );

    /// Callback identifier for `cuStreamWaitValue64`.
    pub const CU_STREAM_WAIT_VALUE64: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64 as u32,
    );

    /// Callback identifier for `cuStreamWaitValue64_ptsz`.
    pub const CU_STREAM_WAIT_VALUE64_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamWriteValue64`.
    pub const CU_STREAM_WRITE_VALUE64: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64 as u32,
    );

    /// Callback identifier for `cuStreamWriteValue64_ptsz`.
    pub const CU_STREAM_WRITE_VALUE64_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_ptsz
            as u32,
    );

    /// Callback identifier for `cuLaunchCooperativeKernel`.
    pub const CU_LAUNCH_COOPERATIVE_KERNEL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel
            as u32,
    );

    /// Callback identifier for `cuLaunchCooperativeKernel_ptsz`.
    pub const CU_LAUNCH_COOPERATIVE_KERNEL_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz as u32);

    /// Callback identifier for `cuEventCreateFromEGLSync`.
    pub const CU_EVENT_CREATE_FROM_EGL_SYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventCreateFromEGLSync
            as u32,
    );

    /// Callback identifier for `cuLaunchCooperativeKernelMultiDevice`.
    pub const CU_LAUNCH_COOPERATIVE_KERNEL_MULTI_DEVICE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice as u32);

    /// Callback identifier for `cuFuncSetAttribute`.
    pub const CU_FUNC_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncSetAttribute as u32,
    );

    /// Callback identifier for `cuDeviceGetUuid`.
    pub const CU_DEVICE_GET_UUID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetUuid as u32);

    /// Callback identifier for `cuStreamGetCtx`.
    pub const CU_STREAM_GET_CTX: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx as u32);

    /// Callback identifier for `cuStreamGetCtx_ptsz`.
    pub const CU_STREAM_GET_CTX_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_ptsz as u32,
    );

    /// Callback identifier for `cuImportExternalMemory`.
    pub const CU_IMPORT_EXTERNAL_MEMORY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuImportExternalMemory
            as u32,
    );

    /// Callback identifier for `cuExternalMemoryGetMappedBuffer`.
    pub const CU_EXTERNAL_MEMORY_GET_MAPPED_BUFFER: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedBuffer as u32);

    /// Callback identifier for `cuExternalMemoryGetMappedMipmappedArray`.
    pub const CU_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedMipmappedArray as u32);

    /// Callback identifier for `cuDestroyExternalMemory`.
    pub const CU_DESTROY_EXTERNAL_MEMORY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDestroyExternalMemory
            as u32,
    );

    /// Callback identifier for `cuImportExternalSemaphore`.
    pub const CU_IMPORT_EXTERNAL_SEMAPHORE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuImportExternalSemaphore
            as u32,
    );

    /// Callback identifier for `cuSignalExternalSemaphoresAsync`.
    pub const CU_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSignalExternalSemaphoresAsync as u32);

    /// Callback identifier for `cuSignalExternalSemaphoresAsync_ptsz`.
    pub const CU_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuSignalExternalSemaphoresAsync_ptsz as u32);

    /// Callback identifier for `cuWaitExternalSemaphoresAsync`.
    pub const CU_WAIT_EXTERNAL_SEMAPHORES_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuWaitExternalSemaphoresAsync
            as u32,
    );

    /// Callback identifier for `cuWaitExternalSemaphoresAsync_ptsz`.
    pub const CU_WAIT_EXTERNAL_SEMAPHORES_ASYNC_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuWaitExternalSemaphoresAsync_ptsz as u32);

    /// Callback identifier for `cuDestroyExternalSemaphore`.
    pub const CU_DESTROY_EXTERNAL_SEMAPHORE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDestroyExternalSemaphore
            as u32,
    );

    /// Callback identifier for `cuStreamBeginCapture`.
    pub const CU_STREAM_BEGIN_CAPTURE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture as u32,
    );

    /// Callback identifier for `cuStreamBeginCapture_ptsz`.
    pub const CU_STREAM_BEGIN_CAPTURE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamEndCapture`.
    pub const CU_STREAM_END_CAPTURE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture as u32,
    );

    /// Callback identifier for `cuStreamEndCapture_ptsz`.
    pub const CU_STREAM_END_CAPTURE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamEndCapture_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamIsCapturing`.
    pub const CU_STREAM_IS_CAPTURING: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamIsCapturing as u32,
    );

    /// Callback identifier for `cuStreamIsCapturing_ptsz`.
    pub const CU_STREAM_IS_CAPTURING_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamIsCapturing_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphCreate`.
    pub const CU_GRAPH_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphCreate as u32);

    /// Callback identifier for `cuGraphAddKernelNode`.
    pub const CU_GRAPH_ADD_KERNEL_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode as u32,
    );

    /// Callback identifier for `cuGraphKernelNodeGetParams`.
    pub const CU_GRAPH_KERNEL_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetParams
            as u32,
    );

    /// Callback identifier for `cuGraphAddMemcpyNode`.
    pub const CU_GRAPH_ADD_MEMCPY_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemcpyNode as u32,
    );

    /// Callback identifier for `cuGraphMemcpyNodeGetParams`.
    pub const CU_GRAPH_MEMCPY_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphMemcpyNodeGetParams
            as u32,
    );

    /// Callback identifier for `cuGraphAddMemsetNode`.
    pub const CU_GRAPH_ADD_MEMSET_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemsetNode as u32,
    );

    /// Callback identifier for `cuGraphMemsetNodeGetParams`.
    pub const CU_GRAPH_MEMSET_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphMemsetNodeGetParams
            as u32,
    );

    /// Callback identifier for `cuGraphMemsetNodeSetParams`.
    pub const CU_GRAPH_MEMSET_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphMemsetNodeSetParams
            as u32,
    );

    /// Callback identifier for `cuGraphNodeGetType`.
    pub const CU_GRAPH_NODE_GET_TYPE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetType as u32,
    );

    /// Callback identifier for `cuGraphGetRootNodes`.
    pub const CU_GRAPH_GET_ROOT_NODES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphGetRootNodes as u32,
    );

    /// Callback identifier for `cuGraphNodeGetDependencies`.
    pub const CU_GRAPH_NODE_GET_DEPENDENCIES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependencies
            as u32,
    );

    /// Callback identifier for `cuGraphNodeGetDependentNodes`.
    pub const CU_GRAPH_NODE_GET_DEPENDENT_NODES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependentNodes
            as u32,
    );

    /// Callback identifier for `cuGraphInstantiate`.
    pub const CU_GRAPH_INSTANTIATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate as u32,
    );

    /// Callback identifier for `cuGraphLaunch`.
    pub const CU_GRAPH_LAUNCH: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch as u32);

    /// Callback identifier for `cuGraphLaunch_ptsz`.
    pub const CU_GRAPH_LAUNCH_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz as u32,
    );

    /// Callback identifier for `cuGraphExecDestroy`.
    pub const CU_GRAPH_EXEC_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecDestroy as u32,
    );

    /// Callback identifier for `cuGraphDestroy`.
    pub const CU_GRAPH_DESTROY: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphDestroy as u32);

    /// Callback identifier for `cuGraphAddDependencies`.
    pub const CU_GRAPH_ADD_DEPENDENCIES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddDependencies
            as u32,
    );

    /// Callback identifier for `cuGraphRemoveDependencies`.
    pub const CU_GRAPH_REMOVE_DEPENDENCIES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphRemoveDependencies
            as u32,
    );

    /// Callback identifier for `cuGraphMemcpyNodeSetParams`.
    pub const CU_GRAPH_MEMCPY_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphMemcpyNodeSetParams
            as u32,
    );

    /// Callback identifier for `cuGraphKernelNodeSetParams`.
    pub const CU_GRAPH_KERNEL_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams
            as u32,
    );

    /// Callback identifier for `cuGraphDestroyNode`.
    pub const CU_GRAPH_DESTROY_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphDestroyNode as u32,
    );

    /// Callback identifier for `cuGraphClone`.
    pub const CU_GRAPH_CLONE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphClone as u32);

    /// Callback identifier for `cuGraphNodeFindInClone`.
    pub const CU_GRAPH_NODE_FIND_IN_CLONE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeFindInClone
            as u32,
    );

    /// Callback identifier for `cuGraphAddChildGraphNode`.
    pub const CU_GRAPH_ADD_CHILD_GRAPH_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddChildGraphNode
            as u32,
    );

    /// Callback identifier for `cuGraphAddEmptyNode`.
    pub const CU_GRAPH_ADD_EMPTY_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddEmptyNode as u32,
    );

    /// Callback identifier for `cuLaunchHostFunc`.
    pub const CU_LAUNCH_HOST_FUNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc as u32,
    );

    /// Callback identifier for `cuLaunchHostFunc_ptsz`.
    pub const CU_LAUNCH_HOST_FUNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_ptsz as u32,
    );

    /// Callback identifier for `cuGraphChildGraphNodeGetGraph`.
    pub const CU_GRAPH_CHILD_GRAPH_NODE_GET_GRAPH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphChildGraphNodeGetGraph
            as u32,
    );

    /// Callback identifier for `cuGraphAddHostNode`.
    pub const CU_GRAPH_ADD_HOST_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddHostNode as u32,
    );

    /// Callback identifier for `cuGraphHostNodeGetParams`.
    pub const CU_GRAPH_HOST_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphHostNodeGetParams
            as u32,
    );

    /// Callback identifier for `cuDeviceGetLuid`.
    pub const CU_DEVICE_GET_LUID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetLuid as u32);

    /// Callback identifier for `cuGraphHostNodeSetParams`.
    pub const CU_GRAPH_HOST_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphHostNodeSetParams
            as u32,
    );

    /// Callback identifier for `cuGraphGetNodes`.
    pub const CU_GRAPH_GET_NODES: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphGetNodes as u32);

    /// Callback identifier for `cuGraphGetEdges`.
    pub const CU_GRAPH_GET_EDGES: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphGetEdges as u32);

    /// Callback identifier for `cuStreamGetCaptureInfo`.
    pub const CU_STREAM_GET_CAPTURE_INFO: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo
            as u32,
    );

    /// Callback identifier for `cuStreamGetCaptureInfo_ptsz`.
    pub const CU_STREAM_GET_CAPTURE_INFO_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphExecKernelNodeSetParams`.
    pub const CU_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams as u32);

    /// Callback identifier for `cuStreamBeginCapture_v2`.
    pub const CU_STREAM_BEGIN_CAPTURE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2
            as u32,
    );

    /// Callback identifier for `cuStreamBeginCapture_v2_ptsz`.
    pub const CU_STREAM_BEGIN_CAPTURE_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCapture_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuThreadExchangeStreamCaptureMode`.
    pub const CU_THREAD_EXCHANGE_STREAM_CAPTURE_MODE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuThreadExchangeStreamCaptureMode as u32);

    /// Callback identifier for `cuDeviceGetNvSciSyncAttributes`.
    pub const CU_DEVICE_GET_NV_SCI_SYNC_ATTRIBUTES: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetNvSciSyncAttributes as u32);

    /// Callback identifier for `cuOccupancyAvailableDynamicSMemPerBlock`.
    pub const CU_OCCUPANCY_AVAILABLE_DYNAMIC_S_MEM_PER_BLOCK: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyAvailableDynamicSMemPerBlock as u32);

    /// Callback identifier for `cuDevicePrimaryCtxRelease_v2`.
    pub const CU_DEVICE_PRIMARY_CTX_RELEASE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease_v2
            as u32,
    );

    /// Callback identifier for `cuDevicePrimaryCtxReset_v2`.
    pub const CU_DEVICE_PRIMARY_CTX_RESET_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset_v2
            as u32,
    );

    /// Callback identifier for `cuDevicePrimaryCtxSetFlags_v2`.
    pub const CU_DEVICE_PRIMARY_CTX_SET_FLAGS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxSetFlags_v2
            as u32,
    );

    /// Callback identifier for `cuMemAddressReserve`.
    pub const CU_MEM_ADDRESS_RESERVE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAddressReserve as u32,
    );

    /// Callback identifier for `cuMemAddressFree`.
    pub const CU_MEM_ADDRESS_FREE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAddressFree as u32,
    );

    /// Callback identifier for `cuMemCreate`.
    pub const CU_MEM_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemCreate as u32);

    /// Callback identifier for `cuMemRelease`.
    pub const CU_MEM_RELEASE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemRelease as u32);

    /// Callback identifier for `cuMemMap`.
    pub const CU_MEM_MAP: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemMap as u32);

    /// Callback identifier for `cuMemUnmap`.
    pub const CU_MEM_UNMAP: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemUnmap as u32);

    /// Callback identifier for `cuMemSetAccess`.
    pub const CU_MEM_SET_ACCESS: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemSetAccess as u32);

    /// Callback identifier for `cuMemExportToShareableHandle`.
    pub const CU_MEM_EXPORT_TO_SHAREABLE_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemExportToShareableHandle
            as u32,
    );

    /// Callback identifier for `cuMemImportFromShareableHandle`.
    pub const CU_MEM_IMPORT_FROM_SHAREABLE_HANDLE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle as u32);

    /// Callback identifier for `cuMemGetAllocationGranularity`.
    pub const CU_MEM_GET_ALLOCATION_GRANULARITY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetAllocationGranularity
            as u32,
    );

    /// Callback identifier for `cuMemGetAllocationPropertiesFromHandle`.
    pub const CU_MEM_GET_ALLOCATION_PROPERTIES_FROM_HANDLE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetAllocationPropertiesFromHandle as u32);

    /// Callback identifier for `cuMemGetAccess`.
    pub const CU_MEM_GET_ACCESS: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetAccess as u32);

    /// Callback identifier for `cuStreamSetFlags`.
    pub const CU_STREAM_SET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamSetFlags as u32,
    );

    /// Callback identifier for `cuStreamSetFlags_ptsz`.
    pub const CU_STREAM_SET_FLAGS_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamSetFlags_ptsz as u32,
    );

    /// Callback identifier for `cuGraphExecUpdate`.
    pub const CU_GRAPH_EXEC_UPDATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate as u32,
    );

    /// Callback identifier for `cuGraphExecMemcpyNodeSetParams`.
    pub const CU_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecMemcpyNodeSetParams as u32);

    /// Callback identifier for `cuGraphExecMemsetNodeSetParams`.
    pub const CU_GRAPH_EXEC_MEMSET_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecMemsetNodeSetParams as u32);

    /// Callback identifier for `cuGraphExecHostNodeSetParams`.
    pub const CU_GRAPH_EXEC_HOST_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecHostNodeSetParams
            as u32,
    );

    /// Callback identifier for `cuMemRetainAllocationHandle`.
    pub const CU_MEM_RETAIN_ALLOCATION_HANDLE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemRetainAllocationHandle
            as u32,
    );

    /// Callback identifier for `cuFuncGetModule`.
    pub const CU_FUNC_GET_MODULE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncGetModule as u32);

    /// Callback identifier for `cuIpcOpenMemHandle_v2`.
    pub const CU_IPC_OPEN_MEM_HANDLE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuIpcOpenMemHandle_v2 as u32,
    );

    /// Callback identifier for `cuCtxResetPersistingL2Cache`.
    pub const CU_CTX_RESET_PERSISTING_L2_CACHE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxResetPersistingL2Cache
            as u32,
    );

    /// Callback identifier for `cuGraphKernelNodeCopyAttributes`.
    pub const CU_GRAPH_KERNEL_NODE_COPY_ATTRIBUTES: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeCopyAttributes as u32);

    /// Callback identifier for `cuGraphKernelNodeGetAttribute`.
    pub const CU_GRAPH_KERNEL_NODE_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetAttribute
            as u32,
    );

    /// Callback identifier for `cuGraphKernelNodeSetAttribute`.
    pub const CU_GRAPH_KERNEL_NODE_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetAttribute
            as u32,
    );

    /// Callback identifier for `cuStreamCopyAttributes`.
    pub const CU_STREAM_COPY_ATTRIBUTES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamCopyAttributes
            as u32,
    );

    /// Callback identifier for `cuStreamCopyAttributes_ptsz`.
    pub const CU_STREAM_COPY_ATTRIBUTES_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamCopyAttributes_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamGetAttribute`.
    pub const CU_STREAM_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetAttribute as u32,
    );

    /// Callback identifier for `cuStreamGetAttribute_ptsz`.
    pub const CU_STREAM_GET_ATTRIBUTE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetAttribute_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamSetAttribute`.
    pub const CU_STREAM_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamSetAttribute as u32,
    );

    /// Callback identifier for `cuStreamSetAttribute_ptsz`.
    pub const CU_STREAM_SET_ATTRIBUTE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamSetAttribute_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphInstantiate_v2`.
    pub const CU_GRAPH_INSTANTIATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiate_v2 as u32,
    );

    /// Callback identifier for `cuDeviceGetTexture1DLinearMaxWidth`.
    pub const CU_DEVICE_GET_TEXTURE1_D_LINEAR_MAX_WIDTH: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetTexture1DLinearMaxWidth as u32);

    /// Callback identifier for `cuGraphUpload`.
    pub const CU_GRAPH_UPLOAD: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphUpload as u32);

    /// Callback identifier for `cuGraphUpload_ptsz`.
    pub const CU_GRAPH_UPLOAD_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphUpload_ptsz as u32,
    );

    /// Callback identifier for `cuArrayGetSparseProperties`.
    pub const CU_ARRAY_GET_SPARSE_PROPERTIES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayGetSparseProperties
            as u32,
    );

    /// Callback identifier for `cuMipmappedArrayGetSparseProperties`.
    pub const CU_MIPMAPPED_ARRAY_GET_SPARSE_PROPERTIES: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetSparseProperties as u32);

    /// Callback identifier for `cuMemMapArrayAsync`.
    pub const CU_MEM_MAP_ARRAY_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync as u32,
    );

    /// Callback identifier for `cuMemMapArrayAsync_ptsz`.
    pub const CU_MEM_MAP_ARRAY_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemMapArrayAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphExecChildGraphNodeSetParams`.
    pub const CU_GRAPH_EXEC_CHILD_GRAPH_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecChildGraphNodeSetParams as u32);

    /// Callback identifier for `cuEventRecordWithFlags`.
    pub const CU_EVENT_RECORD_WITH_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags
            as u32,
    );

    /// Callback identifier for `cuEventRecordWithFlags_ptsz`.
    pub const CU_EVENT_RECORD_WITH_FLAGS_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventRecordWithFlags_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphAddEventRecordNode`.
    pub const CU_GRAPH_ADD_EVENT_RECORD_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventRecordNode
            as u32,
    );

    /// Callback identifier for `cuGraphAddEventWaitNode`.
    pub const CU_GRAPH_ADD_EVENT_WAIT_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventWaitNode
            as u32,
    );

    /// Callback identifier for `cuGraphEventRecordNodeGetEvent`.
    pub const CU_GRAPH_EVENT_RECORD_NODE_GET_EVENT: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphEventRecordNodeGetEvent as u32);

    /// Callback identifier for `cuGraphEventWaitNodeGetEvent`.
    pub const CU_GRAPH_EVENT_WAIT_NODE_GET_EVENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphEventWaitNodeGetEvent
            as u32,
    );

    /// Callback identifier for `cuGraphEventRecordNodeSetEvent`.
    pub const CU_GRAPH_EVENT_RECORD_NODE_SET_EVENT: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphEventRecordNodeSetEvent as u32);

    /// Callback identifier for `cuGraphEventWaitNodeSetEvent`.
    pub const CU_GRAPH_EVENT_WAIT_NODE_SET_EVENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphEventWaitNodeSetEvent
            as u32,
    );

    /// Callback identifier for `cuGraphExecEventRecordNodeSetEvent`.
    pub const CU_GRAPH_EXEC_EVENT_RECORD_NODE_SET_EVENT: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecEventRecordNodeSetEvent as u32);

    /// Callback identifier for `cuGraphExecEventWaitNodeSetEvent`.
    pub const CU_GRAPH_EXEC_EVENT_WAIT_NODE_SET_EVENT: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecEventWaitNodeSetEvent as u32);

    /// Callback identifier for `cuArrayGetPlane`.
    pub const CU_ARRAY_GET_PLANE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane as u32);

    /// Callback identifier for `cuMemAllocAsync`.
    pub const CU_MEM_ALLOC_ASYNC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync as u32);

    /// Callback identifier for `cuMemAllocAsync_ptsz`.
    pub const CU_MEM_ALLOC_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync_ptsz as u32,
    );

    /// Callback identifier for `cuMemFreeAsync`.
    pub const CU_MEM_FREE_ASYNC: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync as u32);

    /// Callback identifier for `cuMemFreeAsync_ptsz`.
    pub const CU_MEM_FREE_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync_ptsz as u32,
    );

    /// Callback identifier for `cuMemPoolTrimTo`.
    pub const CU_MEM_POOL_TRIM_TO: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolTrimTo as u32);

    /// Callback identifier for `cuMemPoolSetAttribute`.
    pub const CU_MEM_POOL_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolSetAttribute as u32,
    );

    /// Callback identifier for `cuMemPoolGetAttribute`.
    pub const CU_MEM_POOL_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolGetAttribute as u32,
    );

    /// Callback identifier for `cuMemPoolSetAccess`.
    pub const CU_MEM_POOL_SET_ACCESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolSetAccess as u32,
    );

    /// Callback identifier for `cuDeviceGetDefaultMemPool`.
    pub const CU_DEVICE_GET_DEFAULT_MEM_POOL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetDefaultMemPool
            as u32,
    );

    /// Callback identifier for `cuMemPoolCreate`.
    pub const CU_MEM_POOL_CREATE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate as u32);

    /// Callback identifier for `cuMemPoolDestroy`.
    pub const CU_MEM_POOL_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolDestroy as u32,
    );

    /// Callback identifier for `cuDeviceSetMemPool`.
    pub const CU_DEVICE_SET_MEM_POOL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceSetMemPool as u32,
    );

    /// Callback identifier for `cuDeviceGetMemPool`.
    pub const CU_DEVICE_GET_MEM_POOL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetMemPool as u32,
    );

    /// Callback identifier for `cuMemAllocFromPoolAsync`.
    pub const CU_MEM_ALLOC_FROM_POOL_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync
            as u32,
    );

    /// Callback identifier for `cuMemAllocFromPoolAsync_ptsz`.
    pub const CU_MEM_ALLOC_FROM_POOL_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAllocFromPoolAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemPoolExportToShareableHandle`.
    pub const CU_MEM_POOL_EXPORT_TO_SHAREABLE_HANDLE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolExportToShareableHandle as u32);

    /// Callback identifier for `cuMemPoolImportFromShareableHandle`.
    pub const CU_MEM_POOL_IMPORT_FROM_SHAREABLE_HANDLE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportFromShareableHandle as u32);

    /// Callback identifier for `cuMemPoolExportPointer`.
    pub const CU_MEM_POOL_EXPORT_POINTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolExportPointer
            as u32,
    );

    /// Callback identifier for `cuMemPoolImportPointer`.
    pub const CU_MEM_POOL_IMPORT_POINTER: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportPointer
            as u32,
    );

    /// Callback identifier for `cuMemPoolGetAccess`.
    pub const CU_MEM_POOL_GET_ACCESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPoolGetAccess as u32,
    );

    /// Callback identifier for `cuGraphAddExternalSemaphoresSignalNode`.
    pub const CU_GRAPH_ADD_EXTERNAL_SEMAPHORES_SIGNAL_NODE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddExternalSemaphoresSignalNode as u32);

    /// Callback identifier for `cuGraphExternalSemaphoresSignalNodeGetParams`.
    pub const CU_GRAPH_EXTERNAL_SEMAPHORES_SIGNAL_NODE_GET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresSignalNodeGetParams as u32);

    /// Callback identifier for `cuGraphExternalSemaphoresSignalNodeSetParams`.
    pub const CU_GRAPH_EXTERNAL_SEMAPHORES_SIGNAL_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresSignalNodeSetParams as u32);

    /// Callback identifier for `cuGraphAddExternalSemaphoresWaitNode`.
    pub const CU_GRAPH_ADD_EXTERNAL_SEMAPHORES_WAIT_NODE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddExternalSemaphoresWaitNode as u32);

    /// Callback identifier for `cuGraphExternalSemaphoresWaitNodeGetParams`.
    pub const CU_GRAPH_EXTERNAL_SEMAPHORES_WAIT_NODE_GET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresWaitNodeGetParams as u32);

    /// Callback identifier for `cuGraphExternalSemaphoresWaitNodeSetParams`.
    pub const CU_GRAPH_EXTERNAL_SEMAPHORES_WAIT_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExternalSemaphoresWaitNodeSetParams as u32);

    /// Callback identifier for `cuGraphExecExternalSemaphoresSignalNodeSetParams`.
    pub const CU_GRAPH_EXEC_EXTERNAL_SEMAPHORES_SIGNAL_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecExternalSemaphoresSignalNodeSetParams as u32);

    /// Callback identifier for `cuGraphExecExternalSemaphoresWaitNodeSetParams`.
    pub const CU_GRAPH_EXEC_EXTERNAL_SEMAPHORES_WAIT_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecExternalSemaphoresWaitNodeSetParams as u32);

    /// Callback identifier for `cuGetProcAddress`.
    pub const CU_GET_PROC_ADDRESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress as u32,
    );

    /// Callback identifier for `cuFlushGPUDirectRDMAWrites`.
    pub const CU_FLUSH_GPU_DIRECT_RDMA_WRITES: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFlushGPUDirectRDMAWrites
            as u32,
    );

    /// Callback identifier for `cuGraphDebugDotPrint`.
    pub const CU_GRAPH_DEBUG_DOT_PRINT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphDebugDotPrint as u32,
    );

    /// Callback identifier for `cuStreamGetCaptureInfo_v2`.
    pub const CU_STREAM_GET_CAPTURE_INFO_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v2
            as u32,
    );

    /// Callback identifier for `cuStreamGetCaptureInfo_v2_ptsz`.
    pub const CU_STREAM_GET_CAPTURE_INFO_V2_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v2_ptsz as u32);

    /// Callback identifier for `cuStreamUpdateCaptureDependencies`.
    pub const CU_STREAM_UPDATE_CAPTURE_DEPENDENCIES: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies as u32);

    /// Callback identifier for `cuStreamUpdateCaptureDependencies_ptsz`.
    pub const CU_STREAM_UPDATE_CAPTURE_DEPENDENCIES_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_ptsz as u32);

    /// Callback identifier for `cuUserObjectCreate`.
    pub const CU_USER_OBJECT_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuUserObjectCreate as u32,
    );

    /// Callback identifier for `cuUserObjectRetain`.
    pub const CU_USER_OBJECT_RETAIN: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuUserObjectRetain as u32,
    );

    /// Callback identifier for `cuUserObjectRelease`.
    pub const CU_USER_OBJECT_RELEASE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuUserObjectRelease as u32,
    );

    /// Callback identifier for `cuGraphRetainUserObject`.
    pub const CU_GRAPH_RETAIN_USER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphRetainUserObject
            as u32,
    );

    /// Callback identifier for `cuGraphReleaseUserObject`.
    pub const CU_GRAPH_RELEASE_USER_OBJECT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphReleaseUserObject
            as u32,
    );

    /// Callback identifier for `cuGraphAddMemAllocNode`.
    pub const CU_GRAPH_ADD_MEM_ALLOC_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemAllocNode
            as u32,
    );

    /// Callback identifier for `cuGraphAddMemFreeNode`.
    pub const CU_GRAPH_ADD_MEM_FREE_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemFreeNode as u32,
    );

    /// Callback identifier for `cuDeviceGraphMemTrim`.
    pub const CU_DEVICE_GRAPH_MEM_TRIM: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGraphMemTrim as u32,
    );

    /// Callback identifier for `cuDeviceGetGraphMemAttribute`.
    pub const CU_DEVICE_GET_GRAPH_MEM_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetGraphMemAttribute
            as u32,
    );

    /// Callback identifier for `cuDeviceSetGraphMemAttribute`.
    pub const CU_DEVICE_SET_GRAPH_MEM_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceSetGraphMemAttribute
            as u32,
    );

    /// Callback identifier for `cuGraphInstantiateWithFlags`.
    pub const CU_GRAPH_INSTANTIATE_WITH_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags
            as u32,
    );

    /// Callback identifier for `cuDeviceGetExecAffinitySupport`.
    pub const CU_DEVICE_GET_EXEC_AFFINITY_SUPPORT: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetExecAffinitySupport as u32);

    /// Callback identifier for `cuCtxCreate_v3`.
    pub const CU_CTX_CREATE_V3: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v3 as u32);

    /// Callback identifier for `cuCtxGetExecAffinity`.
    pub const CU_CTX_GET_EXEC_AFFINITY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetExecAffinity as u32,
    );

    /// Callback identifier for `cuDeviceGetUuid_v2`.
    pub const CU_DEVICE_GET_UUID_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetUuid_v2 as u32,
    );

    /// Callback identifier for `cuGraphMemAllocNodeGetParams`.
    pub const CU_GRAPH_MEM_ALLOC_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphMemAllocNodeGetParams
            as u32,
    );

    /// Callback identifier for `cuGraphMemFreeNodeGetParams`.
    pub const CU_GRAPH_MEM_FREE_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphMemFreeNodeGetParams
            as u32,
    );

    /// Callback identifier for `cuGraphNodeSetEnabled`.
    pub const CU_GRAPH_NODE_SET_ENABLED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetEnabled as u32,
    );

    /// Callback identifier for `cuGraphNodeGetEnabled`.
    pub const CU_GRAPH_NODE_GET_ENABLED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetEnabled as u32,
    );

    /// Callback identifier for `cuLaunchKernelEx`.
    pub const CU_LAUNCH_KERNEL_EX: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx as u32,
    );

    /// Callback identifier for `cuLaunchKernelEx_ptsz`.
    pub const CU_LAUNCH_KERNEL_EX_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz as u32,
    );

    /// Callback identifier for `cuArrayGetMemoryRequirements`.
    pub const CU_ARRAY_GET_MEMORY_REQUIREMENTS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuArrayGetMemoryRequirements
            as u32,
    );

    /// Callback identifier for `cuMipmappedArrayGetMemoryRequirements`.
    pub const CU_MIPMAPPED_ARRAY_GET_MEMORY_REQUIREMENTS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetMemoryRequirements as u32);

    /// Callback identifier for `cuGraphInstantiateWithParams`.
    pub const CU_GRAPH_INSTANTIATE_WITH_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams
            as u32,
    );

    /// Callback identifier for `cuGraphInstantiateWithParams_ptsz`.
    pub const CU_GRAPH_INSTANTIATE_WITH_PARAMS_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz as u32);

    /// Callback identifier for `cuGraphExecGetFlags`.
    pub const CU_GRAPH_EXEC_GET_FLAGS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecGetFlags as u32,
    );

    /// Callback identifier for `cuStreamWaitValue32_v2`.
    pub const CU_STREAM_WAIT_VALUE32_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_v2
            as u32,
    );

    /// Callback identifier for `cuStreamWaitValue32_v2_ptsz`.
    pub const CU_STREAM_WAIT_VALUE32_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue32_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamWaitValue64_v2`.
    pub const CU_STREAM_WAIT_VALUE64_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_v2
            as u32,
    );

    /// Callback identifier for `cuStreamWaitValue64_v2_ptsz`.
    pub const CU_STREAM_WAIT_VALUE64_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWaitValue64_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamWriteValue32_v2`.
    pub const CU_STREAM_WRITE_VALUE32_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_v2
            as u32,
    );

    /// Callback identifier for `cuStreamWriteValue32_v2_ptsz`.
    pub const CU_STREAM_WRITE_VALUE32_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue32_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamWriteValue64_v2`.
    pub const CU_STREAM_WRITE_VALUE64_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_v2
            as u32,
    );

    /// Callback identifier for `cuStreamWriteValue64_v2_ptsz`.
    pub const CU_STREAM_WRITE_VALUE64_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamWriteValue64_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuStreamBatchMemOp_v2`.
    pub const CU_STREAM_BATCH_MEM_OP_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_v2 as u32,
    );

    /// Callback identifier for `cuStreamBatchMemOp_v2_ptsz`.
    pub const CU_STREAM_BATCH_MEM_OP_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBatchMemOp_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphAddBatchMemOpNode`.
    pub const CU_GRAPH_ADD_BATCH_MEM_OP_NODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddBatchMemOpNode
            as u32,
    );

    /// Callback identifier for `cuGraphBatchMemOpNodeGetParams`.
    pub const CU_GRAPH_BATCH_MEM_OP_NODE_GET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphBatchMemOpNodeGetParams as u32);

    /// Callback identifier for `cuGraphBatchMemOpNodeSetParams`.
    pub const CU_GRAPH_BATCH_MEM_OP_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphBatchMemOpNodeSetParams as u32);

    /// Callback identifier for `cuGraphExecBatchMemOpNodeSetParams`.
    pub const CU_GRAPH_EXEC_BATCH_MEM_OP_NODE_SET_PARAMS: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecBatchMemOpNodeSetParams as u32);

    /// Callback identifier for `cuModuleGetLoadingMode`.
    pub const CU_MODULE_GET_LOADING_MODE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetLoadingMode
            as u32,
    );

    /// Callback identifier for `cuMemGetHandleForAddressRange`.
    pub const CU_MEM_GET_HANDLE_FOR_ADDRESS_RANGE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetHandleForAddressRange
            as u32,
    );

    /// Callback identifier for `cuOccupancyMaxPotentialClusterSize`.
    pub const CU_OCCUPANCY_MAX_POTENTIAL_CLUSTER_SIZE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxPotentialClusterSize as u32);

    /// Callback identifier for `cuOccupancyMaxActiveClusters`.
    pub const CU_OCCUPANCY_MAX_ACTIVE_CLUSTERS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuOccupancyMaxActiveClusters
            as u32,
    );

    /// Callback identifier for `cuGetProcAddress_v2`.
    pub const CU_GET_PROC_ADDRESS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGetProcAddress_v2 as u32,
    );

    /// Callback identifier for `cuLibraryLoadData`.
    pub const CU_LIBRARY_LOAD_DATA: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData as u32,
    );

    /// Callback identifier for `cuLibraryLoadFromFile`.
    pub const CU_LIBRARY_LOAD_FROM_FILE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile as u32,
    );

    /// Callback identifier for `cuLibraryUnload`.
    pub const CU_LIBRARY_UNLOAD: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryUnload as u32);

    /// Callback identifier for `cuLibraryGetKernel`.
    pub const CU_LIBRARY_GET_KERNEL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernel as u32,
    );

    /// Callback identifier for `cuLibraryGetModule`.
    pub const CU_LIBRARY_GET_MODULE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetModule as u32,
    );

    /// Callback identifier for `cuKernelGetFunction`.
    pub const CU_KERNEL_GET_FUNCTION: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelGetFunction as u32,
    );

    /// Callback identifier for `cuLibraryGetGlobal`.
    pub const CU_LIBRARY_GET_GLOBAL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetGlobal as u32,
    );

    /// Callback identifier for `cuLibraryGetManaged`.
    pub const CU_LIBRARY_GET_MANAGED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetManaged as u32,
    );

    /// Callback identifier for `cuKernelGetAttribute`.
    pub const CU_KERNEL_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelGetAttribute as u32,
    );

    /// Callback identifier for `cuKernelSetAttribute`.
    pub const CU_KERNEL_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelSetAttribute as u32,
    );

    /// Callback identifier for `cuKernelSetCacheConfig`.
    pub const CU_KERNEL_SET_CACHE_CONFIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelSetCacheConfig
            as u32,
    );

    /// Callback identifier for `cuGraphAddKernelNode_v2`.
    pub const CU_GRAPH_ADD_KERNEL_NODE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2
            as u32,
    );

    /// Callback identifier for `cuGraphKernelNodeGetParams_v2`.
    pub const CU_GRAPH_KERNEL_NODE_GET_PARAMS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeGetParams_v2
            as u32,
    );

    /// Callback identifier for `cuGraphKernelNodeSetParams_v2`.
    pub const CU_GRAPH_KERNEL_NODE_SET_PARAMS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphKernelNodeSetParams_v2
            as u32,
    );

    /// Callback identifier for `cuGraphExecKernelNodeSetParams_v2`.
    pub const CU_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecKernelNodeSetParams_v2 as u32);

    /// Callback identifier for `cuStreamGetId`.
    pub const CU_STREAM_GET_ID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetId as u32);

    /// Callback identifier for `cuStreamGetId_ptsz`.
    pub const CU_STREAM_GET_ID_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetId_ptsz as u32,
    );

    /// Callback identifier for `cuCtxGetId`.
    pub const CU_CTX_GET_ID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetId as u32);

    /// Callback identifier for `cuGraphExecUpdate_v2`.
    pub const CU_GRAPH_EXEC_UPDATE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecUpdate_v2 as u32,
    );

    /// Callback identifier for `cuTensorMapEncodeTiled`.
    pub const CU_TENSOR_MAP_ENCODE_TILED: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeTiled
            as u32,
    );

    /// Callback identifier for `cuTensorMapEncodeIm2col`.
    pub const CU_TENSOR_MAP_ENCODE_IM2COL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeIm2col
            as u32,
    );

    /// Callback identifier for `cuTensorMapReplaceAddress`.
    pub const CU_TENSOR_MAP_REPLACE_ADDRESS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTensorMapReplaceAddress
            as u32,
    );

    /// Callback identifier for `cuLibraryGetUnifiedFunction`.
    pub const CU_LIBRARY_GET_UNIFIED_FUNCTION: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetUnifiedFunction
            as u32,
    );

    /// Callback identifier for `cuCoredumpGetAttribute`.
    pub const CU_COREDUMP_GET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpGetAttribute
            as u32,
    );

    /// Callback identifier for `cuCoredumpGetAttributeGlobal`.
    pub const CU_COREDUMP_GET_ATTRIBUTE_GLOBAL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpGetAttributeGlobal
            as u32,
    );

    /// Callback identifier for `cuCoredumpSetAttribute`.
    pub const CU_COREDUMP_SET_ATTRIBUTE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpSetAttribute
            as u32,
    );

    /// Callback identifier for `cuCoredumpSetAttributeGlobal`.
    pub const CU_COREDUMP_SET_ATTRIBUTE_GLOBAL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpSetAttributeGlobal
            as u32,
    );

    /// Callback identifier for `cuCtxSetFlags`.
    pub const CU_CTX_SET_FLAGS: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSetFlags as u32);

    /// Callback identifier for `cuMulticastCreate`.
    pub const CU_MULTICAST_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastCreate as u32,
    );

    /// Callback identifier for `cuMulticastAddDevice`.
    pub const CU_MULTICAST_ADD_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastAddDevice as u32,
    );

    /// Callback identifier for `cuMulticastBindMem`.
    pub const CU_MULTICAST_BIND_MEM: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastBindMem as u32,
    );

    /// Callback identifier for `cuMulticastBindAddr`.
    pub const CU_MULTICAST_BIND_ADDR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastBindAddr as u32,
    );

    /// Callback identifier for `cuMulticastUnbind`.
    pub const CU_MULTICAST_UNBIND: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastUnbind as u32,
    );

    /// Callback identifier for `cuMulticastGetGranularity`.
    pub const CU_MULTICAST_GET_GRANULARITY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastGetGranularity
            as u32,
    );

    /// Callback identifier for `cuGraphAddNode`.
    pub const CU_GRAPH_ADD_NODE: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode as u32);

    /// Callback identifier for `cuGraphNodeSetParams`.
    pub const CU_GRAPH_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetParams as u32,
    );

    /// Callback identifier for `cuGraphExecNodeSetParams`.
    pub const CU_GRAPH_EXEC_NODE_SET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecNodeSetParams
            as u32,
    );

    /// Callback identifier for `cuMemAdvise_v2`.
    pub const CU_MEM_ADVISE_V2: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemAdvise_v2 as u32);

    /// Callback identifier for `cuMemPrefetchAsync_v2`.
    pub const CU_MEM_PREFETCH_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemPrefetchAsync_v2_ptsz`.
    pub const CU_MEM_PREFETCH_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuFuncGetName`.
    pub const CU_FUNC_GET_NAME: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncGetName as u32);

    /// Callback identifier for `cuKernelGetName`.
    pub const CU_KERNEL_GET_NAME: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelGetName as u32);

    /// Callback identifier for `cuStreamBeginCaptureToGraph`.
    pub const CU_STREAM_BEGIN_CAPTURE_TO_GRAPH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph
            as u32,
    );

    /// Callback identifier for `cuStreamBeginCaptureToGraph_ptsz`.
    pub const CU_STREAM_BEGIN_CAPTURE_TO_GRAPH_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToGraph_ptsz as u32);

    /// Callback identifier for `cuGraphConditionalHandleCreate`.
    pub const CU_GRAPH_CONDITIONAL_HANDLE_CREATE: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphConditionalHandleCreate as u32);

    /// Callback identifier for `cuGraphAddNode_v2`.
    pub const CU_GRAPH_ADD_NODE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode_v2 as u32,
    );

    /// Callback identifier for `cuGraphGetEdges_v2`.
    pub const CU_GRAPH_GET_EDGES_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphGetEdges_v2 as u32,
    );

    /// Callback identifier for `cuGraphNodeGetDependencies_v2`.
    pub const CU_GRAPH_NODE_GET_DEPENDENCIES_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependencies_v2
            as u32,
    );

    /// Callback identifier for `cuGraphNodeGetDependentNodes_v2`.
    pub const CU_GRAPH_NODE_GET_DEPENDENT_NODES_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetDependentNodes_v2 as u32);

    /// Callback identifier for `cuGraphAddDependencies_v2`.
    pub const CU_GRAPH_ADD_DEPENDENCIES_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddDependencies_v2
            as u32,
    );

    /// Callback identifier for `cuGraphRemoveDependencies_v2`.
    pub const CU_GRAPH_REMOVE_DEPENDENCIES_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphRemoveDependencies_v2
            as u32,
    );

    /// Callback identifier for `cuStreamGetCaptureInfo_v3`.
    pub const CU_STREAM_GET_CAPTURE_INFO_V3: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v3
            as u32,
    );

    /// Callback identifier for `cuStreamGetCaptureInfo_v3_ptsz`.
    pub const CU_STREAM_GET_CAPTURE_INFO_V3_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCaptureInfo_v3_ptsz as u32);

    /// Callback identifier for `cuStreamUpdateCaptureDependencies_v2`.
    pub const CU_STREAM_UPDATE_CAPTURE_DEPENDENCIES_V2: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2 as u32);

    /// Callback identifier for `cuStreamUpdateCaptureDependencies_v2_ptsz`.
    pub const CU_STREAM_UPDATE_CAPTURE_DEPENDENCIES_V2_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamUpdateCaptureDependencies_v2_ptsz as u32);

    /// Callback identifier for `cuFuncGetParamInfo`.
    pub const CU_FUNC_GET_PARAM_INFO: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamInfo as u32,
    );

    /// Callback identifier for `cuKernelGetParamInfo`.
    pub const CU_KERNEL_GET_PARAM_INFO: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelGetParamInfo as u32,
    );

    /// Callback identifier for `cuDeviceRegisterAsyncNotification`.
    pub const CU_DEVICE_REGISTER_ASYNC_NOTIFICATION: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceRegisterAsyncNotification as u32);

    /// Callback identifier for `cuDeviceUnregisterAsyncNotification`.
    pub const CU_DEVICE_UNREGISTER_ASYNC_NOTIFICATION: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceUnregisterAsyncNotification as u32);

    /// Callback identifier for `cuModuleGetFunctionCount`.
    pub const CU_MODULE_GET_FUNCTION_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunctionCount
            as u32,
    );

    /// Callback identifier for `cuModuleEnumerateFunctions`.
    pub const CU_MODULE_ENUMERATE_FUNCTIONS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuModuleEnumerateFunctions
            as u32,
    );

    /// Callback identifier for `cuLibraryGetKernelCount`.
    pub const CU_LIBRARY_GET_KERNEL_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryGetKernelCount
            as u32,
    );

    /// Callback identifier for `cuLibraryEnumerateKernels`.
    pub const CU_LIBRARY_ENUMERATE_KERNELS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLibraryEnumerateKernels
            as u32,
    );

    /// Callback identifier for `cuFuncIsLoaded`.
    pub const CU_FUNC_IS_LOADED: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncIsLoaded as u32);

    /// Callback identifier for `cuFuncLoad`.
    pub const CU_FUNC_LOAD: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncLoad as u32);

    /// Callback identifier for `cuGreenCtxCreate`.
    pub const CU_GREEN_CTX_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxCreate as u32,
    );

    /// Callback identifier for `cuGreenCtxDestroy`.
    pub const CU_GREEN_CTX_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxDestroy as u32,
    );

    /// Callback identifier for `cuDeviceGetDevResource`.
    pub const CU_DEVICE_GET_DEV_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetDevResource
            as u32,
    );

    /// Callback identifier for `cuCtxGetDevResource`.
    pub const CU_CTX_GET_DEV_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevResource as u32,
    );

    /// Callback identifier for `cuGreenCtxGetDevResource`.
    pub const CU_GREEN_CTX_GET_DEV_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxGetDevResource
            as u32,
    );

    /// Callback identifier for `cuDevResourceGenerateDesc`.
    pub const CU_DEV_RESOURCE_GENERATE_DESC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevResourceGenerateDesc
            as u32,
    );

    /// Callback identifier for `cuGreenCtxRecordEvent`.
    pub const CU_GREEN_CTX_RECORD_EVENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxRecordEvent as u32,
    );

    /// Callback identifier for `cuGreenCtxWaitEvent`.
    pub const CU_GREEN_CTX_WAIT_EVENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxWaitEvent as u32,
    );

    /// Callback identifier for `cuDevSmResourceSplitByCount`.
    pub const CU_DEV_SM_RESOURCE_SPLIT_BY_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevSmResourceSplitByCount
            as u32,
    );

    /// Callback identifier for `cuStreamGetGreenCtx`.
    pub const CU_STREAM_GET_GREEN_CTX: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetGreenCtx as u32,
    );

    /// Callback identifier for `cuCtxFromGreenCtx`.
    pub const CU_CTX_FROM_GREEN_CTX: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxFromGreenCtx as u32,
    );

    /// Callback identifier for `cuKernelGetLibrary`.
    pub const CU_KERNEL_GET_LIBRARY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelGetLibrary as u32,
    );

    /// Callback identifier for `cuCtxRecordEvent`.
    pub const CU_CTX_RECORD_EVENT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxRecordEvent as u32,
    );

    /// Callback identifier for `cuCtxWaitEvent`.
    pub const CU_CTX_WAIT_EVENT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxWaitEvent as u32);

    /// Callback identifier for `cuCtxCreate_v4`.
    pub const CU_CTX_CREATE_V4: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v4 as u32);

    /// Callback identifier for `cuGreenCtxStreamCreate`.
    pub const CU_GREEN_CTX_STREAM_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxStreamCreate
            as u32,
    );

    /// Callback identifier for `cuStreamGetCtx_v2`.
    pub const CU_STREAM_GET_CTX_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_v2 as u32,
    );

    /// Callback identifier for `cuStreamGetCtx_v2_ptsz`.
    pub const CU_STREAM_GET_CTX_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetCtx_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemBatchDecompressAsync`.
    pub const CU_MEM_BATCH_DECOMPRESS_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemBatchDecompressAsync
            as u32,
    );

    /// Callback identifier for `cuMemBatchDecompressAsync_ptsz`.
    pub const CU_MEM_BATCH_DECOMPRESS_ASYNC_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemBatchDecompressAsync_ptsz as u32);

    /// Callback identifier for `cuLogsRegisterCallback`.
    pub const CU_LOGS_REGISTER_CALLBACK: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogsRegisterCallback
            as u32,
    );

    /// Callback identifier for `cuLogsUnregisterCallback`.
    pub const CU_LOGS_UNREGISTER_CALLBACK: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogsUnregisterCallback
            as u32,
    );

    /// Callback identifier for `cuLogsCurrent`.
    pub const CU_LOGS_CURRENT: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogsCurrent as u32);

    /// Callback identifier for `cuLogsDumpToFile`.
    pub const CU_LOGS_DUMP_TO_FILE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogsDumpToFile as u32,
    );

    /// Callback identifier for `cuLogsDumpToMemory`.
    pub const CU_LOGS_DUMP_TO_MEMORY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogsDumpToMemory as u32,
    );

    /// Callback identifier for `cuCheckpointProcessGetRestoreThreadId`.
    pub const CU_CHECKPOINT_PROCESS_GET_RESTORE_THREAD_ID: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessGetRestoreThreadId as u32);

    /// Callback identifier for `cuCheckpointProcessGetState`.
    pub const CU_CHECKPOINT_PROCESS_GET_STATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessGetState
            as u32,
    );

    /// Callback identifier for `cuCheckpointProcessLock`.
    pub const CU_CHECKPOINT_PROCESS_LOCK: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessLock
            as u32,
    );

    /// Callback identifier for `cuCheckpointProcessCheckpoint`.
    pub const CU_CHECKPOINT_PROCESS_CHECKPOINT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessCheckpoint
            as u32,
    );

    /// Callback identifier for `cuCheckpointProcessRestore`.
    pub const CU_CHECKPOINT_PROCESS_RESTORE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessRestore
            as u32,
    );

    /// Callback identifier for `cuCheckpointProcessUnlock`.
    pub const CU_CHECKPOINT_PROCESS_UNLOCK: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointProcessUnlock
            as u32,
    );

    /// Callback identifier for `cuStreamGetDevice`.
    pub const CU_STREAM_GET_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetDevice as u32,
    );

    /// Callback identifier for `cuStreamGetDevice_ptsz`.
    pub const CU_STREAM_GET_DEVICE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetDevice_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpyBatchAsync`.
    pub const CU_MEMCPY_BATCH_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyBatchAsync as u32,
    );

    /// Callback identifier for `cuMemcpyBatchAsync_ptsz`.
    pub const CU_MEMCPY_BATCH_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyBatchAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpy3DBatchAsync`.
    pub const CU_MEMCPY3_D_BATCH_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DBatchAsync as u32,
    );

    /// Callback identifier for `cuMemcpy3DBatchAsync_ptsz`.
    pub const CU_MEMCPY3_D_BATCH_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DBatchAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuEventElapsedTime_v2`.
    pub const CU_EVENT_ELAPSED_TIME_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuEventElapsedTime_v2 as u32,
    );

    /// Callback identifier for `cuTensorMapEncodeIm2colWide`.
    pub const CU_TENSOR_MAP_ENCODE_IM2COL_WIDE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuTensorMapEncodeIm2colWide
            as u32,
    );

    /// Callback identifier for `cuGreenCtxGetId`.
    pub const CU_GREEN_CTX_GET_ID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGreenCtxGetId as u32);

    /// Callback identifier for `cuStreamBeginCaptureToCig`.
    pub const CU_STREAM_BEGIN_CAPTURE_TO_CIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToCig
            as u32,
    );

    /// Callback identifier for `cuMemPrefetchBatchAsync`.
    pub const CU_MEM_PREFETCH_BATCH_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchBatchAsync
            as u32,
    );

    /// Callback identifier for `cuMemPrefetchBatchAsync_ptsz`.
    pub const CU_MEM_PREFETCH_BATCH_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemPrefetchBatchAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemDiscardBatchAsync`.
    pub const CU_MEM_DISCARD_BATCH_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemDiscardBatchAsync
            as u32,
    );

    /// Callback identifier for `cuMemDiscardBatchAsync_ptsz`.
    pub const CU_MEM_DISCARD_BATCH_ASYNC_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemDiscardBatchAsync_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemDiscardAndPrefetchBatchAsync`.
    pub const CU_MEM_DISCARD_AND_PREFETCH_BATCH_ASYNC: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemDiscardAndPrefetchBatchAsync as u32);

    /// Callback identifier for `cuMemDiscardAndPrefetchBatchAsync_ptsz`.
    pub const CU_MEM_DISCARD_AND_PREFETCH_BATCH_ASYNC_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemDiscardAndPrefetchBatchAsync_ptsz as u32);

    /// Callback identifier for `cuCtxGetDevice_v2`.
    pub const CU_CTX_GET_DEVICE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxGetDevice_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyBatchAsync_v2`.
    pub const CU_MEMCPY_BATCH_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyBatchAsync_v2 as u32,
    );

    /// Callback identifier for `cuMemcpyBatchAsync_v2_ptsz`.
    pub const CU_MEMCPY_BATCH_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyBatchAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuMemcpy3DBatchAsync_v2`.
    pub const CU_MEMCPY3_D_BATCH_ASYNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DBatchAsync_v2
            as u32,
    );

    /// Callback identifier for `cuMemcpy3DBatchAsync_v2_ptsz`.
    pub const CU_MEMCPY3_D_BATCH_ASYNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DBatchAsync_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuCtxSynchronize_v2`.
    pub const CU_CTX_SYNCHRONIZE_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize_v2 as u32,
    );

    /// Callback identifier for `cuMemGetDefaultMemPool`.
    pub const CU_MEM_GET_DEFAULT_MEM_POOL: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetDefaultMemPool
            as u32,
    );

    /// Callback identifier for `cuMemGetMemPool`.
    pub const CU_MEM_GET_MEM_POOL: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetMemPool as u32);

    /// Callback identifier for `cuMemSetMemPool`.
    pub const CU_MEM_SET_MEM_POOL: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemSetMemPool as u32);

    /// Callback identifier for `cuDeviceGetP2PAtomicCapabilities`.
    pub const CU_DEVICE_GET_P2_P_ATOMIC_CAPABILITIES: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetP2PAtomicCapabilities as u32);

    /// Callback identifier for `cuDeviceGetHostAtomicCapabilities`.
    pub const CU_DEVICE_GET_HOST_ATOMIC_CAPABILITIES: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetHostAtomicCapabilities as u32);

    /// Callback identifier for `cuStreamGetDevResource`.
    pub const CU_STREAM_GET_DEV_RESOURCE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetDevResource
            as u32,
    );

    /// Callback identifier for `cuStreamGetDevResource_ptsz`.
    pub const CU_STREAM_GET_DEV_RESOURCE_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamGetDevResource_ptsz
            as u32,
    );

    /// Callback identifier for `cuGraphNodeGetContainingGraph`.
    pub const CU_GRAPH_NODE_GET_CONTAINING_GRAPH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetContainingGraph
            as u32,
    );

    /// Callback identifier for `cuGraphNodeGetLocalId`.
    pub const CU_GRAPH_NODE_GET_LOCAL_ID: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetLocalId as u32,
    );

    /// Callback identifier for `cuGraphNodeGetToolsId`.
    pub const CU_GRAPH_NODE_GET_TOOLS_ID: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetToolsId as u32,
    );

    /// Callback identifier for `cuGraphGetId`.
    pub const CU_GRAPH_GET_ID: Self =
        Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphGetId as u32);

    /// Callback identifier for `cuGraphExecGetId`.
    pub const CU_GRAPH_EXEC_GET_ID: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphExecGetId as u32,
    );

    /// Callback identifier for `cuStreamBeginCaptureToCig_ptsz`.
    pub const CU_STREAM_BEGIN_CAPTURE_TO_CIG_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginCaptureToCig_ptsz as u32);

    /// Callback identifier for `cuStreamEndCaptureToCig`.
    pub const CU_STREAM_END_CAPTURE_TO_CIG: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamEndCaptureToCig
            as u32,
    );

    /// Callback identifier for `cuStreamEndCaptureToCig_ptsz`.
    pub const CU_STREAM_END_CAPTURE_TO_CIG_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamEndCaptureToCig_ptsz
            as u32,
    );

    /// Callback identifier for `cuMulticastBindMem_v2`.
    pub const CU_MULTICAST_BIND_MEM_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastBindMem_v2 as u32,
    );

    /// Callback identifier for `cuMulticastBindAddr_v2`.
    pub const CU_MULTICAST_BIND_ADDR_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMulticastBindAddr_v2
            as u32,
    );

    /// Callback identifier for `cuDevSmResourceSplit`.
    pub const CU_DEV_SM_RESOURCE_SPLIT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDevSmResourceSplit as u32,
    );

    /// Callback identifier for `cuLaunchHostFunc_v2`.
    pub const CU_LAUNCH_HOST_FUNC_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_v2 as u32,
    );

    /// Callback identifier for `cuLaunchHostFunc_v2_ptsz`.
    pub const CU_LAUNCH_HOST_FUNC_V2_PTSZ: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_v2_ptsz
            as u32,
    );

    /// Callback identifier for `cuFuncGetParamCount`.
    pub const CU_FUNC_GET_PARAM_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamCount as u32,
    );

    /// Callback identifier for `cuKernelGetParamCount`.
    pub const CU_KERNEL_GET_PARAM_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuKernelGetParamCount as u32,
    );

    /// Callback identifier for `cuMemcpyWithAttributesAsync`.
    pub const CU_MEMCPY_WITH_ATTRIBUTES_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyWithAttributesAsync
            as u32,
    );

    /// Callback identifier for `cuMemcpyWithAttributesAsync_ptsz`.
    pub const CU_MEMCPY_WITH_ATTRIBUTES_ASYNC_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpyWithAttributesAsync_ptsz as u32);

    /// Callback identifier for `cuMemcpy3DWithAttributesAsync`.
    pub const CU_MEMCPY3_D_WITH_ATTRIBUTES_ASYNC: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DWithAttributesAsync
            as u32,
    );

    /// Callback identifier for `cuMemcpy3DWithAttributesAsync_ptsz`.
    pub const CU_MEMCPY3_D_WITH_ATTRIBUTES_ASYNC_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DWithAttributesAsync_ptsz as u32);

    /// Callback identifier for `cuGraphNodeGetParams`.
    pub const CU_GRAPH_NODE_GET_PARAMS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeGetParams as u32,
    );

    /// Callback identifier for `cuCoredumpRegisterStartCallback`.
    pub const CU_COREDUMP_REGISTER_START_CALLBACK: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpRegisterStartCallback as u32);

    /// Callback identifier for `cuCoredumpRegisterCompleteCallback`.
    pub const CU_COREDUMP_REGISTER_COMPLETE_CALLBACK: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpRegisterCompleteCallback as u32);

    /// Callback identifier for `cuCoredumpDeregisterStartCallback`.
    pub const CU_COREDUMP_DEREGISTER_START_CALLBACK: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpDeregisterStartCallback as u32);

    /// Callback identifier for `cuCoredumpDeregisterCompleteCallback`.
    pub const CU_COREDUMP_DEREGISTER_COMPLETE_CALLBACK: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCoredumpDeregisterCompleteCallback as u32);

    /// Callback identifier for `cuStreamBeginRecaptureToGraph`.
    pub const CU_STREAM_BEGIN_RECAPTURE_TO_GRAPH: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginRecaptureToGraph
            as u32,
    );

    /// Callback identifier for `cuStreamBeginRecaptureToGraph_ptsz`.
    pub const CU_STREAM_BEGIN_RECAPTURE_TO_GRAPH_PTSZ: Self = Self(sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuStreamBeginRecaptureToGraph_ptsz as u32);

    /// Callback identifier for `cuLogicalEndpointIdReserve`.
    pub const CU_LOGICAL_ENDPOINT_ID_RESERVE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointIdReserve
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointIdRelease`.
    pub const CU_LOGICAL_ENDPOINT_ID_RELEASE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointIdRelease
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointCreate`.
    pub const CU_LOGICAL_ENDPOINT_CREATE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointCreate
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointAddDevice`.
    pub const CU_LOGICAL_ENDPOINT_ADD_DEVICE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointAddDevice
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointDestroy`.
    pub const CU_LOGICAL_ENDPOINT_DESTROY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointDestroy
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointBindAddr`.
    pub const CU_LOGICAL_ENDPOINT_BIND_ADDR: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointBindAddr
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointBindMem`.
    pub const CU_LOGICAL_ENDPOINT_BIND_MEM: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointBindMem
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointUnbind`.
    pub const CU_LOGICAL_ENDPOINT_UNBIND: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointUnbind
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointExport`.
    pub const CU_LOGICAL_ENDPOINT_EXPORT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointExport
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointImport`.
    pub const CU_LOGICAL_ENDPOINT_IMPORT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointImport
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointGetLimits`.
    pub const CU_LOGICAL_ENDPOINT_GET_LIMITS: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointGetLimits
            as u32,
    );

    /// Callback identifier for `cuLogicalEndpointQuery`.
    pub const CU_LOGICAL_ENDPOINT_QUERY: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuLogicalEndpointQuery
            as u32,
    );

    /// Callback identifier for `cuMemGetLocationInfo`.
    pub const CU_MEM_GET_LOCATION_INFO: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuMemGetLocationInfo as u32,
    );

    /// Callback identifier for `cuCheckpointOperationComplete`.
    pub const CU_CHECKPOINT_OPERATION_COMPLETE: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuCheckpointOperationComplete
            as u32,
    );

    /// Callback identifier for `cuGraphAddNode_v3`.
    pub const CU_GRAPH_ADD_NODE_V3: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode_v3 as u32,
    );

    /// Callback identifier for `cuGraphNodeSetParams_v2`.
    pub const CU_GRAPH_NODE_SET_PARAMS_V2: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuGraphNodeSetParams_v2
            as u32,
    );

    /// Callback identifier for `cuDeviceGetFabricClusterUuid`.
    pub const CU_DEVICE_GET_FABRIC_CLUSTER_UUID: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetFabricClusterUuid
            as u32,
    );

    /// Callback identifier for `cuDeviceGetCliqueCount`.
    pub const CU_DEVICE_GET_CLIQUE_COUNT: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetCliqueCount
            as u32,
    );

    /// Callback identifier for `cuDeviceGetCliqueInfo`.
    pub const CU_DEVICE_GET_CLIQUE_INFO: Self = Self(
        sys::CUpti_driver_api_trace_cbid_enum::CUPTI_DRIVER_TRACE_CBID_cuDeviceGetCliqueInfo as u32,
    );
}

impl From<u32> for DriverCallbackId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<DriverCallbackId> for CallbackId {
    fn from(value: DriverCallbackId) -> Self {
        Self::from(value.get())
    }
}

/// CUDA runtime API callback IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct RuntimeCallbackId(u32);

impl RuntimeCallbackId {
    /// Returns the raw callback identifier.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Callback identifier for `cudaDriverGetVersion_v3020`.
    pub const CUDA_DRIVER_GET_VERSION_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020
            as u32,
    );

    /// Callback identifier for `cudaRuntimeGetVersion_v3020`.
    pub const CUDA_RUNTIME_GET_VERSION_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaRuntimeGetVersion_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetDeviceCount_v3020`.
    pub const CUDA_GET_DEVICE_COUNT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetDeviceProperties_v3020`.
    pub const CUDA_GET_DEVICE_PROPERTIES_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020 as u32);

    /// Callback identifier for `cudaChooseDevice_v3020`.
    pub const CUDA_CHOOSE_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaChooseDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetChannelDesc_v3020`.
    pub const CUDA_GET_CHANNEL_DESC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetChannelDesc_v3020
            as u32,
    );

    /// Callback identifier for `cudaCreateChannelDesc_v3020`.
    pub const CUDA_CREATE_CHANNEL_DESC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaCreateChannelDesc_v3020
            as u32,
    );

    /// Callback identifier for `cudaConfigureCall_v3020`.
    pub const CUDA_CONFIGURE_CALL_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020
            as u32,
    );

    /// Callback identifier for `cudaSetupArgument_v3020`.
    pub const CUDA_SETUP_ARGUMENT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetLastError_v3020`.
    pub const CUDA_GET_LAST_ERROR_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020
            as u32,
    );

    /// Callback identifier for `cudaPeekAtLastError_v3020`.
    pub const CUDA_PEEK_AT_LAST_ERROR_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaPeekAtLastError_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetErrorString_v3020`.
    pub const CUDA_GET_ERROR_STRING_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorString_v3020
            as u32,
    );

    /// Callback identifier for `cudaLaunch_v3020`.
    pub const CUDA_LAUNCH_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 as u32,
    );

    /// Callback identifier for `cudaFuncSetCacheConfig_v3020`.
    pub const CUDA_FUNC_SET_CACHE_CONFIG_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetCacheConfig_v3020 as u32);

    /// Callback identifier for `cudaFuncGetAttributes_v3020`.
    pub const CUDA_FUNC_GET_ATTRIBUTES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetAttributes_v3020
            as u32,
    );

    /// Callback identifier for `cudaSetDevice_v3020`.
    pub const CUDA_SET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020 as u32,
    );

    /// Callback identifier for `cudaGetDevice_v3020`.
    pub const CUDA_GET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020 as u32,
    );

    /// Callback identifier for `cudaSetValidDevices_v3020`.
    pub const CUDA_SET_VALID_DEVICES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSetValidDevices_v3020
            as u32,
    );

    /// Callback identifier for `cudaSetDeviceFlags_v3020`.
    pub const CUDA_SET_DEVICE_FLAGS_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSetDeviceFlags_v3020
            as u32,
    );

    /// Callback identifier for `cudaMalloc_v3020`.
    pub const CUDA_MALLOC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020 as u32,
    );

    /// Callback identifier for `cudaMallocPitch_v3020`.
    pub const CUDA_MALLOC_PITCH_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020
            as u32,
    );

    /// Callback identifier for `cudaFree_v3020`.
    pub const CUDA_FREE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020 as u32,
    );

    /// Callback identifier for `cudaMallocArray_v3020`.
    pub const CUDA_MALLOC_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020
            as u32,
    );

    /// Callback identifier for `cudaFreeArray_v3020`.
    pub const CUDA_FREE_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020 as u32,
    );

    /// Callback identifier for `cudaMallocHost_v3020`.
    pub const CUDA_MALLOC_HOST_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020
            as u32,
    );

    /// Callback identifier for `cudaFreeHost_v3020`.
    pub const CUDA_FREE_HOST_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020 as u32,
    );

    /// Callback identifier for `cudaHostAlloc_v3020`.
    pub const CUDA_HOST_ALLOC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020 as u32,
    );

    /// Callback identifier for `cudaHostGetDevicePointer_v3020`.
    pub const CUDA_HOST_GET_DEVICE_POINTER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaHostGetDevicePointer_v3020 as u32);

    /// Callback identifier for `cudaHostGetFlags_v3020`.
    pub const CUDA_HOST_GET_FLAGS_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaHostGetFlags_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemGetInfo_v3020`.
    pub const CUDA_MEM_GET_INFO_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemGetInfo_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpy_v3020`.
    pub const CUDA_MEMCPY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 as u32,
    );

    /// Callback identifier for `cudaMemcpy2D_v3020`.
    pub const CUDA_MEMCPY2_D_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020 as u32,
    );

    /// Callback identifier for `cudaMemcpyToArray_v3020`.
    pub const CUDA_MEMCPY_TO_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpy2DToArray_v3020`.
    pub const CUDA_MEMCPY2_D_TO_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpyFromArray_v3020`.
    pub const CUDA_MEMCPY_FROM_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpy2DFromArray_v3020`.
    pub const CUDA_MEMCPY2_D_FROM_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpyArrayToArray_v3020`.
    pub const CUDA_MEMCPY_ARRAY_TO_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_v3020 as u32);

    /// Callback identifier for `cudaMemcpy2DArrayToArray_v3020`.
    pub const CUDA_MEMCPY2_D_ARRAY_TO_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_v3020 as u32);

    /// Callback identifier for `cudaMemcpyToSymbol_v3020`.
    pub const CUDA_MEMCPY_TO_SYMBOL_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpyFromSymbol_v3020`.
    pub const CUDA_MEMCPY_FROM_SYMBOL_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpyAsync_v3020`.
    pub const CUDA_MEMCPY_ASYNC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpyToArrayAsync_v3020`.
    pub const CUDA_MEMCPY_TO_ARRAY_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_v3020 as u32);

    /// Callback identifier for `cudaMemcpyFromArrayAsync_v3020`.
    pub const CUDA_MEMCPY_FROM_ARRAY_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_v3020 as u32);

    /// Callback identifier for `cudaMemcpy2DAsync_v3020`.
    pub const CUDA_MEMCPY2_D_ASYNC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpy2DToArrayAsync_v3020`.
    pub const CUDA_MEMCPY2_D_TO_ARRAY_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_v3020 as u32);

    /// Callback identifier for `cudaMemcpy2DFromArrayAsync_v3020`.
    pub const CUDA_MEMCPY2_D_FROM_ARRAY_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_v3020 as u32);

    /// Callback identifier for `cudaMemcpyToSymbolAsync_v3020`.
    pub const CUDA_MEMCPY_TO_SYMBOL_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020 as u32);

    /// Callback identifier for `cudaMemcpyFromSymbolAsync_v3020`.
    pub const CUDA_MEMCPY_FROM_SYMBOL_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020 as u32);

    /// Callback identifier for `cudaMemset_v3020`.
    pub const CUDA_MEMSET_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020 as u32,
    );

    /// Callback identifier for `cudaMemset2D_v3020`.
    pub const CUDA_MEMSET2_D_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_v3020 as u32,
    );

    /// Callback identifier for `cudaMemsetAsync_v3020`.
    pub const CUDA_MEMSET_ASYNC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemset2DAsync_v3020`.
    pub const CUDA_MEMSET2_D_ASYNC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetSymbolAddress_v3020`.
    pub const CUDA_GET_SYMBOL_ADDRESS_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolAddress_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetSymbolSize_v3020`.
    pub const CUDA_GET_SYMBOL_SIZE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolSize_v3020
            as u32,
    );

    /// Callback identifier for `cudaBindTexture_v3020`.
    pub const CUDA_BIND_TEXTURE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaBindTexture_v3020
            as u32,
    );

    /// Callback identifier for `cudaBindTexture2D_v3020`.
    pub const CUDA_BIND_TEXTURE2_D_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaBindTexture2D_v3020
            as u32,
    );

    /// Callback identifier for `cudaBindTextureToArray_v3020`.
    pub const CUDA_BIND_TEXTURE_TO_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaBindTextureToArray_v3020 as u32);

    /// Callback identifier for `cudaUnbindTexture_v3020`.
    pub const CUDA_UNBIND_TEXTURE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaUnbindTexture_v3020
            as u32,
    );

    /// Callback identifier for `cudaGetTextureAlignmentOffset_v3020`.
    pub const CUDA_GET_TEXTURE_ALIGNMENT_OFFSET_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureAlignmentOffset_v3020 as u32);

    /// Callback identifier for `cudaGetTextureReference_v3020`.
    pub const CUDA_GET_TEXTURE_REFERENCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureReference_v3020 as u32);

    /// Callback identifier for `cudaBindSurfaceToArray_v3020`.
    pub const CUDA_BIND_SURFACE_TO_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaBindSurfaceToArray_v3020 as u32);

    /// Callback identifier for `cudaGetSurfaceReference_v3020`.
    pub const CUDA_GET_SURFACE_REFERENCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceReference_v3020 as u32);

    /// Callback identifier for `cudaGLSetGLDevice_v3020`.
    pub const CUDA_GL_SET_GL_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLSetGLDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaGLRegisterBufferObject_v3020`.
    pub const CUDA_GL_REGISTER_BUFFER_OBJECT_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLRegisterBufferObject_v3020 as u32);

    /// Callback identifier for `cudaGLMapBufferObject_v3020`.
    pub const CUDA_GL_MAP_BUFFER_OBJECT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLMapBufferObject_v3020
            as u32,
    );

    /// Callback identifier for `cudaGLUnmapBufferObject_v3020`.
    pub const CUDA_GL_UNMAP_BUFFER_OBJECT_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLUnmapBufferObject_v3020 as u32);

    /// Callback identifier for `cudaGLUnregisterBufferObject_v3020`.
    pub const CUDA_GL_UNREGISTER_BUFFER_OBJECT_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLUnregisterBufferObject_v3020 as u32);

    /// Callback identifier for `cudaGLSetBufferObjectMapFlags_v3020`.
    pub const CUDA_GL_SET_BUFFER_OBJECT_MAP_FLAGS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLSetBufferObjectMapFlags_v3020 as u32);

    /// Callback identifier for `cudaGLMapBufferObjectAsync_v3020`.
    pub const CUDA_GL_MAP_BUFFER_OBJECT_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLMapBufferObjectAsync_v3020 as u32);

    /// Callback identifier for `cudaGLUnmapBufferObjectAsync_v3020`.
    pub const CUDA_GL_UNMAP_BUFFER_OBJECT_ASYNC_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLUnmapBufferObjectAsync_v3020 as u32);

    /// Callback identifier for `cudaWGLGetDevice_v3020`.
    pub const CUDA_WGL_GET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaWGLGetDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaGraphicsGLRegisterImage_v3020`.
    pub const CUDA_GRAPHICS_GL_REGISTER_IMAGE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsGLRegisterImage_v3020 as u32);

    /// Callback identifier for `cudaGraphicsGLRegisterBuffer_v3020`.
    pub const CUDA_GRAPHICS_GL_REGISTER_BUFFER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsGLRegisterBuffer_v3020 as u32);

    /// Callback identifier for `cudaGraphicsUnregisterResource_v3020`.
    pub const CUDA_GRAPHICS_UNREGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsUnregisterResource_v3020 as u32);

    /// Callback identifier for `cudaGraphicsResourceSetMapFlags_v3020`.
    pub const CUDA_GRAPHICS_RESOURCE_SET_MAP_FLAGS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceSetMapFlags_v3020 as u32);

    /// Callback identifier for `cudaGraphicsMapResources_v3020`.
    pub const CUDA_GRAPHICS_MAP_RESOURCES_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsMapResources_v3020 as u32);

    /// Callback identifier for `cudaGraphicsUnmapResources_v3020`.
    pub const CUDA_GRAPHICS_UNMAP_RESOURCES_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsUnmapResources_v3020 as u32);

    /// Callback identifier for `cudaGraphicsResourceGetMappedPointer_v3020`.
    pub const CUDA_GRAPHICS_RESOURCE_GET_MAPPED_POINTER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedPointer_v3020 as u32);

    /// Callback identifier for `cudaGraphicsSubResourceGetMappedArray_v3020`.
    pub const CUDA_GRAPHICS_SUB_RESOURCE_GET_MAPPED_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsSubResourceGetMappedArray_v3020 as u32);

    /// Callback identifier for `cudaVDPAUGetDevice_v3020`.
    pub const CUDA_VDPAU_GET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaVDPAUGetDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaVDPAUSetVDPAUDevice_v3020`.
    pub const CUDA_VDPAU_SET_VDPAU_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaVDPAUSetVDPAUDevice_v3020 as u32);

    /// Callback identifier for `cudaGraphicsVDPAURegisterVideoSurface_v3020`.
    pub const CUDA_GRAPHICS_VDPAU_REGISTER_VIDEO_SURFACE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsVDPAURegisterVideoSurface_v3020 as u32);

    /// Callback identifier for `cudaGraphicsVDPAURegisterOutputSurface_v3020`.
    pub const CUDA_GRAPHICS_VDPAU_REGISTER_OUTPUT_SURFACE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsVDPAURegisterOutputSurface_v3020 as u32);

    /// Callback identifier for `cudaD3D11GetDevice_v3020`.
    pub const CUDA_D3_D11_GET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D11GetDevices_v3020`.
    pub const CUDA_D3_D11_GET_DEVICES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDevices_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D11SetDirect3DDevice_v3020`.
    pub const CUDA_D3_D11_SET_DIRECT3_D_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D11SetDirect3DDevice_v3020 as u32);

    /// Callback identifier for `cudaGraphicsD3D11RegisterResource_v3020`.
    pub const CUDA_GRAPHICS_D3_D11_REGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D11RegisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D10GetDevice_v3020`.
    pub const CUDA_D3_D10_GET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D10GetDevices_v3020`.
    pub const CUDA_D3_D10_GET_DEVICES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDevices_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D10SetDirect3DDevice_v3020`.
    pub const CUDA_D3_D10_SET_DIRECT3_D_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10SetDirect3DDevice_v3020 as u32);

    /// Callback identifier for `cudaGraphicsD3D10RegisterResource_v3020`.
    pub const CUDA_GRAPHICS_D3_D10_REGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D10RegisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D10RegisterResource_v3020`.
    pub const CUDA_D3_D10_REGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10RegisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D10UnregisterResource_v3020`.
    pub const CUDA_D3_D10_UNREGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10UnregisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D10MapResources_v3020`.
    pub const CUDA_D3_D10_MAP_RESOURCES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10MapResources_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D10UnmapResources_v3020`.
    pub const CUDA_D3_D10_UNMAP_RESOURCES_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10UnmapResources_v3020 as u32);

    /// Callback identifier for `cudaD3D10ResourceSetMapFlags_v3020`.
    pub const CUDA_D3_D10_RESOURCE_SET_MAP_FLAGS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceSetMapFlags_v3020 as u32);

    /// Callback identifier for `cudaD3D10ResourceGetSurfaceDimensions_v3020`.
    pub const CUDA_D3_D10_RESOURCE_GET_SURFACE_DIMENSIONS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetSurfaceDimensions_v3020 as u32);

    /// Callback identifier for `cudaD3D10ResourceGetMappedArray_v3020`.
    pub const CUDA_D3_D10_RESOURCE_GET_MAPPED_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedArray_v3020 as u32);

    /// Callback identifier for `cudaD3D10ResourceGetMappedPointer_v3020`.
    pub const CUDA_D3_D10_RESOURCE_GET_MAPPED_POINTER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedPointer_v3020 as u32);

    /// Callback identifier for `cudaD3D10ResourceGetMappedSize_v3020`.
    pub const CUDA_D3_D10_RESOURCE_GET_MAPPED_SIZE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedSize_v3020 as u32);

    /// Callback identifier for `cudaD3D10ResourceGetMappedPitch_v3020`.
    pub const CUDA_D3_D10_RESOURCE_GET_MAPPED_PITCH_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10ResourceGetMappedPitch_v3020 as u32);

    /// Callback identifier for `cudaD3D9GetDevice_v3020`.
    pub const CUDA_D3_D9_GET_DEVICE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDevice_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D9GetDevices_v3020`.
    pub const CUDA_D3_D9_GET_DEVICES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDevices_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D9SetDirect3DDevice_v3020`.
    pub const CUDA_D3_D9_SET_DIRECT3_D_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9SetDirect3DDevice_v3020 as u32);

    /// Callback identifier for `cudaD3D9GetDirect3DDevice_v3020`.
    pub const CUDA_D3_D9_GET_DIRECT3_D_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9GetDirect3DDevice_v3020 as u32);

    /// Callback identifier for `cudaGraphicsD3D9RegisterResource_v3020`.
    pub const CUDA_GRAPHICS_D3_D9_REGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsD3D9RegisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D9RegisterResource_v3020`.
    pub const CUDA_D3_D9_REGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9RegisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D9UnregisterResource_v3020`.
    pub const CUDA_D3_D9_UNREGISTER_RESOURCE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnregisterResource_v3020 as u32);

    /// Callback identifier for `cudaD3D9MapResources_v3020`.
    pub const CUDA_D3_D9_MAP_RESOURCES_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9MapResources_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D9UnmapResources_v3020`.
    pub const CUDA_D3_D9_UNMAP_RESOURCES_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnmapResources_v3020 as u32);

    /// Callback identifier for `cudaD3D9ResourceSetMapFlags_v3020`.
    pub const CUDA_D3_D9_RESOURCE_SET_MAP_FLAGS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceSetMapFlags_v3020 as u32);

    /// Callback identifier for `cudaD3D9ResourceGetSurfaceDimensions_v3020`.
    pub const CUDA_D3_D9_RESOURCE_GET_SURFACE_DIMENSIONS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetSurfaceDimensions_v3020 as u32);

    /// Callback identifier for `cudaD3D9ResourceGetMappedArray_v3020`.
    pub const CUDA_D3_D9_RESOURCE_GET_MAPPED_ARRAY_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedArray_v3020 as u32);

    /// Callback identifier for `cudaD3D9ResourceGetMappedPointer_v3020`.
    pub const CUDA_D3_D9_RESOURCE_GET_MAPPED_POINTER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedPointer_v3020 as u32);

    /// Callback identifier for `cudaD3D9ResourceGetMappedSize_v3020`.
    pub const CUDA_D3_D9_RESOURCE_GET_MAPPED_SIZE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedSize_v3020 as u32);

    /// Callback identifier for `cudaD3D9ResourceGetMappedPitch_v3020`.
    pub const CUDA_D3_D9_RESOURCE_GET_MAPPED_PITCH_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9ResourceGetMappedPitch_v3020 as u32);

    /// Callback identifier for `cudaD3D9Begin_v3020`.
    pub const CUDA_D3_D9_BEGIN_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9Begin_v3020 as u32,
    );

    /// Callback identifier for `cudaD3D9End_v3020`.
    pub const CUDA_D3_D9_END_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9End_v3020 as u32,
    );

    /// Callback identifier for `cudaD3D9RegisterVertexBuffer_v3020`.
    pub const CUDA_D3_D9_REGISTER_VERTEX_BUFFER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9RegisterVertexBuffer_v3020 as u32);

    /// Callback identifier for `cudaD3D9UnregisterVertexBuffer_v3020`.
    pub const CUDA_D3_D9_UNREGISTER_VERTEX_BUFFER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnregisterVertexBuffer_v3020 as u32);

    /// Callback identifier for `cudaD3D9MapVertexBuffer_v3020`.
    pub const CUDA_D3_D9_MAP_VERTEX_BUFFER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9MapVertexBuffer_v3020 as u32);

    /// Callback identifier for `cudaD3D9UnmapVertexBuffer_v3020`.
    pub const CUDA_D3_D9_UNMAP_VERTEX_BUFFER_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D9UnmapVertexBuffer_v3020 as u32);

    /// Callback identifier for `cudaThreadExit_v3020`.
    pub const CUDA_THREAD_EXIT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020
            as u32,
    );

    /// Callback identifier for `cudaSetDoubleForDevice_v3020`.
    pub const CUDA_SET_DOUBLE_FOR_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSetDoubleForDevice_v3020 as u32);

    /// Callback identifier for `cudaSetDoubleForHost_v3020`.
    pub const CUDA_SET_DOUBLE_FOR_HOST_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSetDoubleForHost_v3020
            as u32,
    );

    /// Callback identifier for `cudaThreadSynchronize_v3020`.
    pub const CUDA_THREAD_SYNCHRONIZE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020
            as u32,
    );

    /// Callback identifier for `cudaThreadGetLimit_v3020`.
    pub const CUDA_THREAD_GET_LIMIT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadGetLimit_v3020
            as u32,
    );

    /// Callback identifier for `cudaThreadSetLimit_v3020`.
    pub const CUDA_THREAD_SET_LIMIT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadSetLimit_v3020
            as u32,
    );

    /// Callback identifier for `cudaStreamCreate_v3020`.
    pub const CUDA_STREAM_CREATE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020
            as u32,
    );

    /// Callback identifier for `cudaStreamDestroy_v3020`.
    pub const CUDA_STREAM_DESTROY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020
            as u32,
    );

    /// Callback identifier for `cudaStreamSynchronize_v3020`.
    pub const CUDA_STREAM_SYNCHRONIZE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020
            as u32,
    );

    /// Callback identifier for `cudaStreamQuery_v3020`.
    pub const CUDA_STREAM_QUERY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_v3020
            as u32,
    );

    /// Callback identifier for `cudaEventCreate_v3020`.
    pub const CUDA_EVENT_CREATE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventCreate_v3020
            as u32,
    );

    /// Callback identifier for `cudaEventCreateWithFlags_v3020`.
    pub const CUDA_EVENT_CREATE_WITH_FLAGS_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020 as u32);

    /// Callback identifier for `cudaEventRecord_v3020`.
    pub const CUDA_EVENT_RECORD_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020
            as u32,
    );

    /// Callback identifier for `cudaEventDestroy_v3020`.
    pub const CUDA_EVENT_DESTROY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020
            as u32,
    );

    /// Callback identifier for `cudaEventSynchronize_v3020`.
    pub const CUDA_EVENT_SYNCHRONIZE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020
            as u32,
    );

    /// Callback identifier for `cudaEventQuery_v3020`.
    pub const CUDA_EVENT_QUERY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020
            as u32,
    );

    /// Callback identifier for `cudaEventElapsedTime_v3020`.
    pub const CUDA_EVENT_ELAPSED_TIME_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020
            as u32,
    );

    /// Callback identifier for `cudaMalloc3D_v3020`.
    pub const CUDA_MALLOC3_D_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020 as u32,
    );

    /// Callback identifier for `cudaMalloc3DArray_v3020`.
    pub const CUDA_MALLOC3_D_ARRAY_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemset3D_v3020`.
    pub const CUDA_MEMSET3_D_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_v3020 as u32,
    );

    /// Callback identifier for `cudaMemset3DAsync_v3020`.
    pub const CUDA_MEMSET3_D_ASYNC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_v3020
            as u32,
    );

    /// Callback identifier for `cudaMemcpy3D_v3020`.
    pub const CUDA_MEMCPY3_D_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_v3020 as u32,
    );

    /// Callback identifier for `cudaMemcpy3DAsync_v3020`.
    pub const CUDA_MEMCPY3_D_ASYNC_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020
            as u32,
    );

    /// Callback identifier for `cudaThreadSetCacheConfig_v3020`.
    pub const CUDA_THREAD_SET_CACHE_CONFIG_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadSetCacheConfig_v3020 as u32);

    /// Callback identifier for `cudaStreamWaitEvent_v3020`.
    pub const CUDA_STREAM_WAIT_EVENT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020
            as u32,
    );

    /// Callback identifier for `cudaD3D11GetDirect3DDevice_v3020`.
    pub const CUDA_D3_D11_GET_DIRECT3_D_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D11GetDirect3DDevice_v3020 as u32);

    /// Callback identifier for `cudaD3D10GetDirect3DDevice_v3020`.
    pub const CUDA_D3_D10_GET_DIRECT3_D_DEVICE_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaD3D10GetDirect3DDevice_v3020 as u32);

    /// Callback identifier for `cudaThreadGetCacheConfig_v3020`.
    pub const CUDA_THREAD_GET_CACHE_CONFIG_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadGetCacheConfig_v3020 as u32);

    /// Callback identifier for `cudaPointerGetAttributes_v4000`.
    pub const CUDA_POINTER_GET_ATTRIBUTES_V4000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaPointerGetAttributes_v4000 as u32);

    /// Callback identifier for `cudaHostRegister_v4000`.
    pub const CUDA_HOST_REGISTER_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaHostRegister_v4000
            as u32,
    );

    /// Callback identifier for `cudaHostUnregister_v4000`.
    pub const CUDA_HOST_UNREGISTER_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaHostUnregister_v4000
            as u32,
    );

    /// Callback identifier for `cudaDeviceCanAccessPeer_v4000`.
    pub const CUDA_DEVICE_CAN_ACCESS_PEER_V4000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceCanAccessPeer_v4000 as u32);

    /// Callback identifier for `cudaDeviceEnablePeerAccess_v4000`.
    pub const CUDA_DEVICE_ENABLE_PEER_ACCESS_V4000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceEnablePeerAccess_v4000 as u32);

    /// Callback identifier for `cudaDeviceDisablePeerAccess_v4000`.
    pub const CUDA_DEVICE_DISABLE_PEER_ACCESS_V4000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceDisablePeerAccess_v4000 as u32);

    /// Callback identifier for `cudaPeerRegister_v4000`.
    pub const CUDA_PEER_REGISTER_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaPeerRegister_v4000
            as u32,
    );

    /// Callback identifier for `cudaPeerUnregister_v4000`.
    pub const CUDA_PEER_UNREGISTER_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaPeerUnregister_v4000
            as u32,
    );

    /// Callback identifier for `cudaPeerGetDevicePointer_v4000`.
    pub const CUDA_PEER_GET_DEVICE_POINTER_V4000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaPeerGetDevicePointer_v4000 as u32);

    /// Callback identifier for `cudaMemcpyPeer_v4000`.
    pub const CUDA_MEMCPY_PEER_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeer_v4000
            as u32,
    );

    /// Callback identifier for `cudaMemcpyPeerAsync_v4000`.
    pub const CUDA_MEMCPY_PEER_ASYNC_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000
            as u32,
    );

    /// Callback identifier for `cudaMemcpy3DPeer_v4000`.
    pub const CUDA_MEMCPY3_D_PEER_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_v4000
            as u32,
    );

    /// Callback identifier for `cudaMemcpy3DPeerAsync_v4000`.
    pub const CUDA_MEMCPY3_D_PEER_ASYNC_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_v4000
            as u32,
    );

    /// Callback identifier for `cudaDeviceReset_v3020`.
    pub const CUDA_DEVICE_RESET_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020
            as u32,
    );

    /// Callback identifier for `cudaDeviceSynchronize_v3020`.
    pub const CUDA_DEVICE_SYNCHRONIZE_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020
            as u32,
    );

    /// Callback identifier for `cudaDeviceGetLimit_v3020`.
    pub const CUDA_DEVICE_GET_LIMIT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetLimit_v3020
            as u32,
    );

    /// Callback identifier for `cudaDeviceSetLimit_v3020`.
    pub const CUDA_DEVICE_SET_LIMIT_V3020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetLimit_v3020
            as u32,
    );

    /// Callback identifier for `cudaDeviceGetCacheConfig_v3020`.
    pub const CUDA_DEVICE_GET_CACHE_CONFIG_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetCacheConfig_v3020 as u32);

    /// Callback identifier for `cudaDeviceSetCacheConfig_v3020`.
    pub const CUDA_DEVICE_SET_CACHE_CONFIG_V3020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetCacheConfig_v3020 as u32);

    /// Callback identifier for `cudaProfilerInitialize_v4000`.
    pub const CUDA_PROFILER_INITIALIZE_V4000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaProfilerInitialize_v4000 as u32);

    /// Callback identifier for `cudaProfilerStart_v4000`.
    pub const CUDA_PROFILER_START_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000
            as u32,
    );

    /// Callback identifier for `cudaProfilerStop_v4000`.
    pub const CUDA_PROFILER_STOP_V4000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000
            as u32,
    );

    /// Callback identifier for `cudaDeviceGetByPCIBusId_v4010`.
    pub const CUDA_DEVICE_GET_BY_PCI_BUS_ID_V4010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetByPCIBusId_v4010 as u32);

    /// Callback identifier for `cudaDeviceGetPCIBusId_v4010`.
    pub const CUDA_DEVICE_GET_PCI_BUS_ID_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetPCIBusId_v4010
            as u32,
    );

    /// Callback identifier for `cudaGLGetDevices_v4010`.
    pub const CUDA_GL_GET_DEVICES_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGLGetDevices_v4010
            as u32,
    );

    /// Callback identifier for `cudaIpcGetEventHandle_v4010`.
    pub const CUDA_IPC_GET_EVENT_HANDLE_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetEventHandle_v4010
            as u32,
    );

    /// Callback identifier for `cudaIpcOpenEventHandle_v4010`.
    pub const CUDA_IPC_OPEN_EVENT_HANDLE_V4010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenEventHandle_v4010 as u32);

    /// Callback identifier for `cudaIpcGetMemHandle_v4010`.
    pub const CUDA_IPC_GET_MEM_HANDLE_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaIpcGetMemHandle_v4010
            as u32,
    );

    /// Callback identifier for `cudaIpcOpenMemHandle_v4010`.
    pub const CUDA_IPC_OPEN_MEM_HANDLE_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaIpcOpenMemHandle_v4010
            as u32,
    );

    /// Callback identifier for `cudaIpcCloseMemHandle_v4010`.
    pub const CUDA_IPC_CLOSE_MEM_HANDLE_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaIpcCloseMemHandle_v4010
            as u32,
    );

    /// Callback identifier for `cudaArrayGetInfo_v4010`.
    pub const CUDA_ARRAY_GET_INFO_V4010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetInfo_v4010
            as u32,
    );

    /// Callback identifier for `cudaFuncSetSharedMemConfig_v4020`.
    pub const CUDA_FUNC_SET_SHARED_MEM_CONFIG_V4020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetSharedMemConfig_v4020 as u32);

    /// Callback identifier for `cudaDeviceGetSharedMemConfig_v4020`.
    pub const CUDA_DEVICE_GET_SHARED_MEM_CONFIG_V4020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetSharedMemConfig_v4020 as u32);

    /// Callback identifier for `cudaDeviceSetSharedMemConfig_v4020`.
    pub const CUDA_DEVICE_SET_SHARED_MEM_CONFIG_V4020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetSharedMemConfig_v4020 as u32);

    /// Callback identifier for `cudaCreateTextureObject_v5000`.
    pub const CUDA_CREATE_TEXTURE_OBJECT_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaCreateTextureObject_v5000 as u32);

    /// Callback identifier for `cudaDestroyTextureObject_v5000`.
    pub const CUDA_DESTROY_TEXTURE_OBJECT_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDestroyTextureObject_v5000 as u32);

    /// Callback identifier for `cudaGetTextureObjectResourceDesc_v5000`.
    pub const CUDA_GET_TEXTURE_OBJECT_RESOURCE_DESC_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectResourceDesc_v5000 as u32);

    /// Callback identifier for `cudaGetTextureObjectTextureDesc_v5000`.
    pub const CUDA_GET_TEXTURE_OBJECT_TEXTURE_DESC_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectTextureDesc_v5000 as u32);

    /// Callback identifier for `cudaCreateSurfaceObject_v5000`.
    pub const CUDA_CREATE_SURFACE_OBJECT_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaCreateSurfaceObject_v5000 as u32);

    /// Callback identifier for `cudaDestroySurfaceObject_v5000`.
    pub const CUDA_DESTROY_SURFACE_OBJECT_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDestroySurfaceObject_v5000 as u32);

    /// Callback identifier for `cudaGetSurfaceObjectResourceDesc_v5000`.
    pub const CUDA_GET_SURFACE_OBJECT_RESOURCE_DESC_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceObjectResourceDesc_v5000 as u32);

    /// Callback identifier for `cudaMallocMipmappedArray_v5000`.
    pub const CUDA_MALLOC_MIPMAPPED_ARRAY_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000 as u32);

    /// Callback identifier for `cudaGetMipmappedArrayLevel_v5000`.
    pub const CUDA_GET_MIPMAPPED_ARRAY_LEVEL_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetMipmappedArrayLevel_v5000 as u32);

    /// Callback identifier for `cudaFreeMipmappedArray_v5000`.
    pub const CUDA_FREE_MIPMAPPED_ARRAY_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFreeMipmappedArray_v5000 as u32);

    /// Callback identifier for `cudaBindTextureToMipmappedArray_v5000`.
    pub const CUDA_BIND_TEXTURE_TO_MIPMAPPED_ARRAY_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaBindTextureToMipmappedArray_v5000 as u32);

    /// Callback identifier for `cudaGraphicsResourceGetMappedMipmappedArray_v5000`.
    pub const CUDA_GRAPHICS_RESOURCE_GET_MAPPED_MIPMAPPED_ARRAY_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedMipmappedArray_v5000 as u32);

    /// Callback identifier for `cudaStreamAddCallback_v5000`.
    pub const CUDA_STREAM_ADD_CALLBACK_V5000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamAddCallback_v5000
            as u32,
    );

    /// Callback identifier for `cudaStreamCreateWithFlags_v5000`.
    pub const CUDA_STREAM_CREATE_WITH_FLAGS_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000 as u32);

    /// Callback identifier for `cudaGetTextureObjectResourceViewDesc_v5000`.
    pub const CUDA_GET_TEXTURE_OBJECT_RESOURCE_VIEW_DESC_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectResourceViewDesc_v5000 as u32);

    /// Callback identifier for `cudaDeviceGetAttribute_v5000`.
    pub const CUDA_DEVICE_GET_ATTRIBUTE_V5000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetAttribute_v5000 as u32);

    /// Callback identifier for `cudaStreamDestroy_v5050`.
    pub const CUDA_STREAM_DESTROY_V5050: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050
            as u32,
    );

    /// Callback identifier for `cudaStreamCreateWithPriority_v5050`.
    pub const CUDA_STREAM_CREATE_WITH_PRIORITY_V5050: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050 as u32);

    /// Callback identifier for `cudaStreamGetPriority_v5050`.
    pub const CUDA_STREAM_GET_PRIORITY_V5050: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetPriority_v5050
            as u32,
    );

    /// Callback identifier for `cudaStreamGetFlags_v5050`.
    pub const CUDA_STREAM_GET_FLAGS_V5050: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetFlags_v5050
            as u32,
    );

    /// Callback identifier for `cudaDeviceGetStreamPriorityRange_v5050`.
    pub const CUDA_DEVICE_GET_STREAM_PRIORITY_RANGE_V5050: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetStreamPriorityRange_v5050 as u32);

    /// Callback identifier for `cudaMallocManaged_v6000`.
    pub const CUDA_MALLOC_MANAGED_V6000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000
            as u32,
    );

    /// Callback identifier for `cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000`.
    pub const CUDA_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_V6000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000 as u32);

    /// Callback identifier for `cudaStreamAttachMemAsync_v6000`.
    pub const CUDA_STREAM_ATTACH_MEM_ASYNC_V6000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamAttachMemAsync_v6000 as u32);

    /// Callback identifier for `cudaGetErrorName_v6050`.
    pub const CUDA_GET_ERROR_NAME_V6050: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorName_v6050
            as u32,
    );

    /// Callback identifier for `cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050`.
    pub const CUDA_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_V6050: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050 as u32);

    /// Callback identifier for `cudaLaunchKernel_v7000`.
    pub const CUDA_LAUNCH_KERNEL_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
            as u32,
    );

    /// Callback identifier for `cudaGetDeviceFlags_v7000`.
    pub const CUDA_GET_DEVICE_FLAGS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceFlags_v7000
            as u32,
    );

    /// Callback identifier for `cudaLaunch_ptsz_v7000`.
    pub const CUDA_LAUNCH_PTSZ_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000
            as u32,
    );

    /// Callback identifier for `cudaLaunchKernel_ptsz_v7000`.
    pub const CUDA_LAUNCH_KERNEL_PTSZ_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemcpy_ptds_v7000`.
    pub const CUDA_MEMCPY_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemcpy2D_ptds_v7000`.
    pub const CUDA_MEMCPY2_D_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemcpyToArray_ptds_v7000`.
    pub const CUDA_MEMCPY_TO_ARRAY_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpy2DToArray_ptds_v7000`.
    pub const CUDA_MEMCPY2_D_TO_ARRAY_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArray_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpyFromArray_ptds_v7000`.
    pub const CUDA_MEMCPY_FROM_ARRAY_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpy2DFromArray_ptds_v7000`.
    pub const CUDA_MEMCPY2_D_FROM_ARRAY_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArray_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpyArrayToArray_ptds_v7000`.
    pub const CUDA_MEMCPY_ARRAY_TO_ARRAY_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpy2DArrayToArray_ptds_v7000`.
    pub const CUDA_MEMCPY2_D_ARRAY_TO_ARRAY_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DArrayToArray_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpyToSymbol_ptds_v7000`.
    pub const CUDA_MEMCPY_TO_SYMBOL_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpyFromSymbol_ptds_v7000`.
    pub const CUDA_MEMCPY_FROM_SYMBOL_PTDS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_ptds_v7000 as u32);

    /// Callback identifier for `cudaMemcpyAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY_ASYNC_PTSZ_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemcpyToArrayAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY_TO_ARRAY_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpyFromArrayAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY_FROM_ARRAY_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpy2DAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY2_D_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpy2DToArrayAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY2_D_TO_ARRAY_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DToArrayAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpy2DFromArrayAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY2_D_FROM_ARRAY_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DFromArrayAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpyToSymbolAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY_TO_SYMBOL_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpyFromSymbolAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY_FROM_SYMBOL_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemset_ptds_v7000`.
    pub const CUDA_MEMSET_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemset2D_ptds_v7000`.
    pub const CUDA_MEMSET2_D_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemsetAsync_ptsz_v7000`.
    pub const CUDA_MEMSET_ASYNC_PTSZ_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemset2DAsync_ptsz_v7000`.
    pub const CUDA_MEMSET2_D_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaStreamGetPriority_ptsz_v7000`.
    pub const CUDA_STREAM_GET_PRIORITY_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetPriority_ptsz_v7000 as u32);

    /// Callback identifier for `cudaStreamGetFlags_ptsz_v7000`.
    pub const CUDA_STREAM_GET_FLAGS_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetFlags_ptsz_v7000 as u32);

    /// Callback identifier for `cudaStreamSynchronize_ptsz_v7000`.
    pub const CUDA_STREAM_SYNCHRONIZE_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000 as u32);

    /// Callback identifier for `cudaStreamQuery_ptsz_v7000`.
    pub const CUDA_STREAM_QUERY_PTSZ_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamQuery_ptsz_v7000
            as u32,
    );

    /// Callback identifier for `cudaStreamAttachMemAsync_ptsz_v7000`.
    pub const CUDA_STREAM_ATTACH_MEM_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamAttachMemAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaEventRecord_ptsz_v7000`.
    pub const CUDA_EVENT_RECORD_PTSZ_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_ptsz_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemset3D_ptds_v7000`.
    pub const CUDA_MEMSET3_D_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemset3DAsync_ptsz_v7000`.
    pub const CUDA_MEMSET3_D_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpy3D_ptds_v7000`.
    pub const CUDA_MEMCPY3_D_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3D_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemcpy3DAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY3_D_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaStreamWaitEvent_ptsz_v7000`.
    pub const CUDA_STREAM_WAIT_EVENT_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_ptsz_v7000 as u32);

    /// Callback identifier for `cudaStreamAddCallback_ptsz_v7000`.
    pub const CUDA_STREAM_ADD_CALLBACK_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamAddCallback_ptsz_v7000 as u32);

    /// Callback identifier for `cudaMemcpy3DPeer_ptds_v7000`.
    pub const CUDA_MEMCPY3_D_PEER_PTDS_V7000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeer_ptds_v7000
            as u32,
    );

    /// Callback identifier for `cudaMemcpy3DPeerAsync_ptsz_v7000`.
    pub const CUDA_MEMCPY3_D_PEER_ASYNC_PTSZ_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DPeerAsync_ptsz_v7000 as u32);

    /// Callback identifier for `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000`.
    pub const CUDA_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_MULTIPROCESSOR_WITH_FLAGS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000 as u32);

    /// Callback identifier for `cudaMemPrefetchAsync_v8000`.
    pub const CUDA_MEM_PREFETCH_ASYNC_V8000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v8000
            as u32,
    );

    /// Callback identifier for `cudaMemPrefetchAsync_ptsz_v8000`.
    pub const CUDA_MEM_PREFETCH_ASYNC_PTSZ_V8000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_ptsz_v8000 as u32);

    /// Callback identifier for `cudaMemAdvise_v8000`.
    pub const CUDA_MEM_ADVISE_V8000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemAdvise_v8000 as u32,
    );

    /// Callback identifier for `cudaDeviceGetP2PAttribute_v8000`.
    pub const CUDA_DEVICE_GET_P2_P_ATTRIBUTE_V8000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetP2PAttribute_v8000 as u32);

    /// Callback identifier for `cudaGraphicsEGLRegisterImage_v7000`.
    pub const CUDA_GRAPHICS_EGL_REGISTER_IMAGE_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsEGLRegisterImage_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamConsumerConnect_v7000`.
    pub const CUDA_EGL_STREAM_CONSUMER_CONNECT_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerConnect_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamConsumerDisconnect_v7000`.
    pub const CUDA_EGL_STREAM_CONSUMER_DISCONNECT_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerDisconnect_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamConsumerAcquireFrame_v7000`.
    pub const CUDA_EGL_STREAM_CONSUMER_ACQUIRE_FRAME_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerAcquireFrame_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamConsumerReleaseFrame_v7000`.
    pub const CUDA_EGL_STREAM_CONSUMER_RELEASE_FRAME_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerReleaseFrame_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamProducerConnect_v7000`.
    pub const CUDA_EGL_STREAM_PRODUCER_CONNECT_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerConnect_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamProducerDisconnect_v7000`.
    pub const CUDA_EGL_STREAM_PRODUCER_DISCONNECT_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerDisconnect_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamProducerPresentFrame_v7000`.
    pub const CUDA_EGL_STREAM_PRODUCER_PRESENT_FRAME_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerPresentFrame_v7000 as u32);

    /// Callback identifier for `cudaEGLStreamProducerReturnFrame_v7000`.
    pub const CUDA_EGL_STREAM_PRODUCER_RETURN_FRAME_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamProducerReturnFrame_v7000 as u32);

    /// Callback identifier for `cudaGraphicsResourceGetMappedEglFrame_v7000`.
    pub const CUDA_GRAPHICS_RESOURCE_GET_MAPPED_EGL_FRAME_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphicsResourceGetMappedEglFrame_v7000 as u32);

    /// Callback identifier for `cudaMemRangeGetAttribute_v8000`.
    pub const CUDA_MEM_RANGE_GET_ATTRIBUTE_V8000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemRangeGetAttribute_v8000 as u32);

    /// Callback identifier for `cudaMemRangeGetAttributes_v8000`.
    pub const CUDA_MEM_RANGE_GET_ATTRIBUTES_V8000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemRangeGetAttributes_v8000 as u32);

    /// Callback identifier for `cudaEGLStreamConsumerConnectWithFlags_v7000`.
    pub const CUDA_EGL_STREAM_CONSUMER_CONNECT_WITH_FLAGS_V7000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEGLStreamConsumerConnectWithFlags_v7000 as u32);

    /// Callback identifier for `cudaLaunchCooperativeKernel_v9000`.
    pub const CUDA_LAUNCH_COOPERATIVE_KERNEL_V9000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 as u32);

    /// Callback identifier for `cudaLaunchCooperativeKernel_ptsz_v9000`.
    pub const CUDA_LAUNCH_COOPERATIVE_KERNEL_PTSZ_V9000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000 as u32);

    /// Callback identifier for `cudaEventCreateFromEGLSync_v9000`.
    pub const CUDA_EVENT_CREATE_FROM_EGL_SYNC_V9000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateFromEGLSync_v9000 as u32);

    /// Callback identifier for `cudaLaunchCooperativeKernelMultiDevice_v9000`.
    pub const CUDA_LAUNCH_COOPERATIVE_KERNEL_MULTI_DEVICE_V9000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000 as u32);

    /// Callback identifier for `cudaFuncSetAttribute_v9000`.
    pub const CUDA_FUNC_SET_ATTRIBUTE_V9000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetAttribute_v9000
            as u32,
    );

    /// Callback identifier for `cudaImportExternalMemory_v10000`.
    pub const CUDA_IMPORT_EXTERNAL_MEMORY_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaImportExternalMemory_v10000 as u32);

    /// Callback identifier for `cudaExternalMemoryGetMappedBuffer_v10000`.
    pub const CUDA_EXTERNAL_MEMORY_GET_MAPPED_BUFFER_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExternalMemoryGetMappedBuffer_v10000 as u32);

    /// Callback identifier for `cudaExternalMemoryGetMappedMipmappedArray_v10000`.
    pub const CUDA_EXTERNAL_MEMORY_GET_MAPPED_MIPMAPPED_ARRAY_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExternalMemoryGetMappedMipmappedArray_v10000 as u32);

    /// Callback identifier for `cudaDestroyExternalMemory_v10000`.
    pub const CUDA_DESTROY_EXTERNAL_MEMORY_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDestroyExternalMemory_v10000 as u32);

    /// Callback identifier for `cudaImportExternalSemaphore_v10000`.
    pub const CUDA_IMPORT_EXTERNAL_SEMAPHORE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaImportExternalSemaphore_v10000 as u32);

    /// Callback identifier for `cudaSignalExternalSemaphoresAsync_v10000`.
    pub const CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_v10000 as u32);

    /// Callback identifier for `cudaSignalExternalSemaphoresAsync_ptsz_v10000`.
    pub const CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC_PTSZ_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_ptsz_v10000 as u32);

    /// Callback identifier for `cudaWaitExternalSemaphoresAsync_v10000`.
    pub const CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_v10000 as u32);

    /// Callback identifier for `cudaWaitExternalSemaphoresAsync_ptsz_v10000`.
    pub const CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC_PTSZ_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_ptsz_v10000 as u32);

    /// Callback identifier for `cudaDestroyExternalSemaphore_v10000`.
    pub const CUDA_DESTROY_EXTERNAL_SEMAPHORE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDestroyExternalSemaphore_v10000 as u32);

    /// Callback identifier for `cudaLaunchHostFunc_v10000`.
    pub const CUDA_LAUNCH_HOST_FUNC_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000
            as u32,
    );

    /// Callback identifier for `cudaLaunchHostFunc_ptsz_v10000`.
    pub const CUDA_LAUNCH_HOST_FUNC_PTSZ_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000 as u32);

    /// Callback identifier for `cudaGraphCreate_v10000`.
    pub const CUDA_GRAPH_CREATE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphCreate_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphKernelNodeGetParams_v10000`.
    pub const CUDA_GRAPH_KERNEL_NODE_GET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeGetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphKernelNodeSetParams_v10000`.
    pub const CUDA_GRAPH_KERNEL_NODE_SET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphAddKernelNode_v10000`.
    pub const CUDA_GRAPH_ADD_KERNEL_NODE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddKernelNode_v10000 as u32);

    /// Callback identifier for `cudaGraphAddMemcpyNode_v10000`.
    pub const CUDA_GRAPH_ADD_MEMCPY_NODE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNode_v10000 as u32);

    /// Callback identifier for `cudaGraphMemcpyNodeGetParams_v10000`.
    pub const CUDA_GRAPH_MEMCPY_NODE_GET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeGetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphMemcpyNodeSetParams_v10000`.
    pub const CUDA_GRAPH_MEMCPY_NODE_SET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphAddMemsetNode_v10000`.
    pub const CUDA_GRAPH_ADD_MEMSET_NODE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemsetNode_v10000 as u32);

    /// Callback identifier for `cudaGraphMemsetNodeGetParams_v10000`.
    pub const CUDA_GRAPH_MEMSET_NODE_GET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemsetNodeGetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphMemsetNodeSetParams_v10000`.
    pub const CUDA_GRAPH_MEMSET_NODE_SET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemsetNodeSetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphAddHostNode_v10000`.
    pub const CUDA_GRAPH_ADD_HOST_NODE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddHostNode_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphHostNodeGetParams_v10000`.
    pub const CUDA_GRAPH_HOST_NODE_GET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphHostNodeGetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphAddChildGraphNode_v10000`.
    pub const CUDA_GRAPH_ADD_CHILD_GRAPH_NODE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddChildGraphNode_v10000 as u32);

    /// Callback identifier for `cudaGraphChildGraphNodeGetGraph_v10000`.
    pub const CUDA_GRAPH_CHILD_GRAPH_NODE_GET_GRAPH_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphChildGraphNodeGetGraph_v10000 as u32);

    /// Callback identifier for `cudaGraphAddEmptyNode_v10000`.
    pub const CUDA_GRAPH_ADD_EMPTY_NODE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddEmptyNode_v10000 as u32);

    /// Callback identifier for `cudaGraphClone_v10000`.
    pub const CUDA_GRAPH_CLONE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphClone_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphNodeFindInClone_v10000`.
    pub const CUDA_GRAPH_NODE_FIND_IN_CLONE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeFindInClone_v10000 as u32);

    /// Callback identifier for `cudaGraphNodeGetType_v10000`.
    pub const CUDA_GRAPH_NODE_GET_TYPE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetType_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphGetRootNodes_v10000`.
    pub const CUDA_GRAPH_GET_ROOT_NODES_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetRootNodes_v10000 as u32);

    /// Callback identifier for `cudaGraphNodeGetDependencies_v10000`.
    pub const CUDA_GRAPH_NODE_GET_DEPENDENCIES_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependencies_v10000 as u32);

    /// Callback identifier for `cudaGraphNodeGetDependentNodes_v10000`.
    pub const CUDA_GRAPH_NODE_GET_DEPENDENT_NODES_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependentNodes_v10000 as u32);

    /// Callback identifier for `cudaGraphAddDependencies_v10000`.
    pub const CUDA_GRAPH_ADD_DEPENDENCIES_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddDependencies_v10000 as u32);

    /// Callback identifier for `cudaGraphRemoveDependencies_v10000`.
    pub const CUDA_GRAPH_REMOVE_DEPENDENCIES_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphRemoveDependencies_v10000 as u32);

    /// Callback identifier for `cudaGraphDestroyNode_v10000`.
    pub const CUDA_GRAPH_DESTROY_NODE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphDestroyNode_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphInstantiate_v10000`.
    pub const CUDA_GRAPH_INSTANTIATE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphLaunch_v10000`.
    pub const CUDA_GRAPH_LAUNCH_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphLaunch_ptsz_v10000`.
    pub const CUDA_GRAPH_LAUNCH_PTSZ_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphExecDestroy_v10000`.
    pub const CUDA_GRAPH_EXEC_DESTROY_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecDestroy_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphDestroy_v10000`.
    pub const CUDA_GRAPH_DESTROY_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphDestroy_v10000
            as u32,
    );

    /// Callback identifier for `cudaStreamBeginCapture_v10000`.
    pub const CUDA_STREAM_BEGIN_CAPTURE_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_v10000 as u32);

    /// Callback identifier for `cudaStreamBeginCapture_ptsz_v10000`.
    pub const CUDA_STREAM_BEGIN_CAPTURE_PTSZ_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_ptsz_v10000 as u32);

    /// Callback identifier for `cudaStreamIsCapturing_v10000`.
    pub const CUDA_STREAM_IS_CAPTURING_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamIsCapturing_v10000 as u32);

    /// Callback identifier for `cudaStreamIsCapturing_ptsz_v10000`.
    pub const CUDA_STREAM_IS_CAPTURING_PTSZ_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamIsCapturing_ptsz_v10000 as u32);

    /// Callback identifier for `cudaStreamEndCapture_v10000`.
    pub const CUDA_STREAM_END_CAPTURE_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_v10000
            as u32,
    );

    /// Callback identifier for `cudaStreamEndCapture_ptsz_v10000`.
    pub const CUDA_STREAM_END_CAPTURE_PTSZ_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamEndCapture_ptsz_v10000 as u32);

    /// Callback identifier for `cudaGraphHostNodeSetParams_v10000`.
    pub const CUDA_GRAPH_HOST_NODE_SET_PARAMS_V10000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphHostNodeSetParams_v10000 as u32);

    /// Callback identifier for `cudaGraphGetNodes_v10000`.
    pub const CUDA_GRAPH_GET_NODES_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetNodes_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphGetEdges_v10000`.
    pub const CUDA_GRAPH_GET_EDGES_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetEdges_v10000
            as u32,
    );

    /// Callback identifier for `cudaStreamGetCaptureInfo_v10010`.
    pub const CUDA_STREAM_GET_CAPTURE_INFO_V10010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v10010 as u32);

    /// Callback identifier for `cudaStreamGetCaptureInfo_ptsz_v10010`.
    pub const CUDA_STREAM_GET_CAPTURE_INFO_PTSZ_V10010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_ptsz_v10010 as u32);

    /// Callback identifier for `cudaGraphExecKernelNodeSetParams_v10010`.
    pub const CUDA_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS_V10010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecKernelNodeSetParams_v10010 as u32);

    /// Callback identifier for `cudaThreadExchangeStreamCaptureMode_v10010`.
    pub const CUDA_THREAD_EXCHANGE_STREAM_CAPTURE_MODE_V10010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaThreadExchangeStreamCaptureMode_v10010 as u32);

    /// Callback identifier for `cudaDeviceGetNvSciSyncAttributes_v10020`.
    pub const CUDA_DEVICE_GET_NV_SCI_SYNC_ATTRIBUTES_V10020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetNvSciSyncAttributes_v10020 as u32);

    /// Callback identifier for `cudaOccupancyAvailableDynamicSMemPerBlock_v10200`.
    pub const CUDA_OCCUPANCY_AVAILABLE_DYNAMIC_S_MEM_PER_BLOCK_V10200: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyAvailableDynamicSMemPerBlock_v10200 as u32);

    /// Callback identifier for `cudaStreamSetFlags_v10200`.
    pub const CUDA_STREAM_SET_FLAGS_V10200: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetFlags_v10200
            as u32,
    );

    /// Callback identifier for `cudaStreamSetFlags_ptsz_v10200`.
    pub const CUDA_STREAM_SET_FLAGS_PTSZ_V10200: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetFlags_ptsz_v10200 as u32);

    /// Callback identifier for `cudaGraphExecMemcpyNodeSetParams_v10020`.
    pub const CUDA_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_V10020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParams_v10020 as u32);

    /// Callback identifier for `cudaGraphExecMemsetNodeSetParams_v10020`.
    pub const CUDA_GRAPH_EXEC_MEMSET_NODE_SET_PARAMS_V10020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemsetNodeSetParams_v10020 as u32);

    /// Callback identifier for `cudaGraphExecHostNodeSetParams_v10020`.
    pub const CUDA_GRAPH_EXEC_HOST_NODE_SET_PARAMS_V10020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecHostNodeSetParams_v10020 as u32);

    /// Callback identifier for `cudaGraphExecUpdate_v10020`.
    pub const CUDA_GRAPH_EXEC_UPDATE_V10020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecUpdate_v10020
            as u32,
    );

    /// Callback identifier for `cudaGetFuncBySymbol_v11000`.
    pub const CUDA_GET_FUNC_BY_SYMBOL_V11000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetFuncBySymbol_v11000
            as u32,
    );

    /// Callback identifier for `cudaCtxResetPersistingL2Cache_v11000`.
    pub const CUDA_CTX_RESET_PERSISTING_L2_CACHE_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaCtxResetPersistingL2Cache_v11000 as u32);

    /// Callback identifier for `cudaGraphKernelNodeCopyAttributes_v11000`.
    pub const CUDA_GRAPH_KERNEL_NODE_COPY_ATTRIBUTES_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeCopyAttributes_v11000 as u32);

    /// Callback identifier for `cudaGraphKernelNodeGetAttribute_v11000`.
    pub const CUDA_GRAPH_KERNEL_NODE_GET_ATTRIBUTE_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeGetAttribute_v11000 as u32);

    /// Callback identifier for `cudaGraphKernelNodeSetAttribute_v11000`.
    pub const CUDA_GRAPH_KERNEL_NODE_SET_ATTRIBUTE_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphKernelNodeSetAttribute_v11000 as u32);

    /// Callback identifier for `cudaStreamCopyAttributes_v11000`.
    pub const CUDA_STREAM_COPY_ATTRIBUTES_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamCopyAttributes_v11000 as u32);

    /// Callback identifier for `cudaStreamCopyAttributes_ptsz_v11000`.
    pub const CUDA_STREAM_COPY_ATTRIBUTES_PTSZ_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamCopyAttributes_ptsz_v11000 as u32);

    /// Callback identifier for `cudaStreamGetAttribute_v11000`.
    pub const CUDA_STREAM_GET_ATTRIBUTE_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetAttribute_v11000 as u32);

    /// Callback identifier for `cudaStreamGetAttribute_ptsz_v11000`.
    pub const CUDA_STREAM_GET_ATTRIBUTE_PTSZ_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetAttribute_ptsz_v11000 as u32);

    /// Callback identifier for `cudaStreamSetAttribute_v11000`.
    pub const CUDA_STREAM_SET_ATTRIBUTE_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetAttribute_v11000 as u32);

    /// Callback identifier for `cudaStreamSetAttribute_ptsz_v11000`.
    pub const CUDA_STREAM_SET_ATTRIBUTE_PTSZ_V11000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamSetAttribute_ptsz_v11000 as u32);

    /// Callback identifier for `cudaDeviceGetTexture1DLinearMaxWidth_v11010`.
    pub const CUDA_DEVICE_GET_TEXTURE1_D_LINEAR_MAX_WIDTH_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetTexture1DLinearMaxWidth_v11010 as u32);

    /// Callback identifier for `cudaGraphUpload_v10000`.
    pub const CUDA_GRAPH_UPLOAD_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphUpload_ptsz_v10000`.
    pub const CUDA_GRAPH_UPLOAD_PTSZ_V10000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphUpload_ptsz_v10000
            as u32,
    );

    /// Callback identifier for `cudaGraphAddMemcpyNodeToSymbol_v11010`.
    pub const CUDA_GRAPH_ADD_MEMCPY_NODE_TO_SYMBOL_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNodeToSymbol_v11010 as u32);

    /// Callback identifier for `cudaGraphAddMemcpyNodeFromSymbol_v11010`.
    pub const CUDA_GRAPH_ADD_MEMCPY_NODE_FROM_SYMBOL_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNodeFromSymbol_v11010 as u32);

    /// Callback identifier for `cudaGraphAddMemcpyNode1D_v11010`.
    pub const CUDA_GRAPH_ADD_MEMCPY_NODE1_D_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemcpyNode1D_v11010 as u32);

    /// Callback identifier for `cudaGraphMemcpyNodeSetParamsToSymbol_v11010`.
    pub const CUDA_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_SYMBOL_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParamsToSymbol_v11010 as u32);

    /// Callback identifier for `cudaGraphMemcpyNodeSetParamsFromSymbol_v11010`.
    pub const CUDA_GRAPH_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParamsFromSymbol_v11010 as u32);

    /// Callback identifier for `cudaGraphMemcpyNodeSetParams1D_v11010`.
    pub const CUDA_GRAPH_MEMCPY_NODE_SET_PARAMS1_D_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemcpyNodeSetParams1D_v11010 as u32);

    /// Callback identifier for `cudaGraphExecMemcpyNodeSetParamsToSymbol_v11010`.
    pub const CUDA_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_TO_SYMBOL_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParamsToSymbol_v11010 as u32);

    /// Callback identifier for `cudaGraphExecMemcpyNodeSetParamsFromSymbol_v11010`.
    pub const CUDA_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS_FROM_SYMBOL_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParamsFromSymbol_v11010 as u32);

    /// Callback identifier for `cudaGraphExecMemcpyNodeSetParams1D_v11010`.
    pub const CUDA_GRAPH_EXEC_MEMCPY_NODE_SET_PARAMS1_D_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecMemcpyNodeSetParams1D_v11010 as u32);

    /// Callback identifier for `cudaArrayGetSparseProperties_v11010`.
    pub const CUDA_ARRAY_GET_SPARSE_PROPERTIES_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetSparseProperties_v11010 as u32);

    /// Callback identifier for `cudaMipmappedArrayGetSparseProperties_v11010`.
    pub const CUDA_MIPMAPPED_ARRAY_GET_SPARSE_PROPERTIES_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMipmappedArrayGetSparseProperties_v11010 as u32);

    /// Callback identifier for `cudaGraphExecChildGraphNodeSetParams_v11010`.
    pub const CUDA_GRAPH_EXEC_CHILD_GRAPH_NODE_SET_PARAMS_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecChildGraphNodeSetParams_v11010 as u32);

    /// Callback identifier for `cudaGraphAddEventRecordNode_v11010`.
    pub const CUDA_GRAPH_ADD_EVENT_RECORD_NODE_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddEventRecordNode_v11010 as u32);

    /// Callback identifier for `cudaGraphEventRecordNodeGetEvent_v11010`.
    pub const CUDA_GRAPH_EVENT_RECORD_NODE_GET_EVENT_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventRecordNodeGetEvent_v11010 as u32);

    /// Callback identifier for `cudaGraphEventRecordNodeSetEvent_v11010`.
    pub const CUDA_GRAPH_EVENT_RECORD_NODE_SET_EVENT_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventRecordNodeSetEvent_v11010 as u32);

    /// Callback identifier for `cudaGraphAddEventWaitNode_v11010`.
    pub const CUDA_GRAPH_ADD_EVENT_WAIT_NODE_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddEventWaitNode_v11010 as u32);

    /// Callback identifier for `cudaGraphEventWaitNodeGetEvent_v11010`.
    pub const CUDA_GRAPH_EVENT_WAIT_NODE_GET_EVENT_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventWaitNodeGetEvent_v11010 as u32);

    /// Callback identifier for `cudaGraphEventWaitNodeSetEvent_v11010`.
    pub const CUDA_GRAPH_EVENT_WAIT_NODE_SET_EVENT_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphEventWaitNodeSetEvent_v11010 as u32);

    /// Callback identifier for `cudaGraphExecEventRecordNodeSetEvent_v11010`.
    pub const CUDA_GRAPH_EXEC_EVENT_RECORD_NODE_SET_EVENT_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecEventRecordNodeSetEvent_v11010 as u32);

    /// Callback identifier for `cudaGraphExecEventWaitNodeSetEvent_v11010`.
    pub const CUDA_GRAPH_EXEC_EVENT_WAIT_NODE_SET_EVENT_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecEventWaitNodeSetEvent_v11010 as u32);

    /// Callback identifier for `cudaEventRecordWithFlags_v11010`.
    pub const CUDA_EVENT_RECORD_WITH_FLAGS_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventRecordWithFlags_v11010 as u32);

    /// Callback identifier for `cudaEventRecordWithFlags_ptsz_v11010`.
    pub const CUDA_EVENT_RECORD_WITH_FLAGS_PTSZ_V11010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventRecordWithFlags_ptsz_v11010 as u32);

    /// Callback identifier for `cudaDeviceGetDefaultMemPool_v11020`.
    pub const CUDA_DEVICE_GET_DEFAULT_MEM_POOL_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetDefaultMemPool_v11020 as u32);

    /// Callback identifier for `cudaMallocAsync_v11020`.
    pub const CUDA_MALLOC_ASYNC_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020
            as u32,
    );

    /// Callback identifier for `cudaMallocAsync_ptsz_v11020`.
    pub const CUDA_MALLOC_ASYNC_PTSZ_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_ptsz_v11020
            as u32,
    );

    /// Callback identifier for `cudaFreeAsync_v11020`.
    pub const CUDA_FREE_ASYNC_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020
            as u32,
    );

    /// Callback identifier for `cudaFreeAsync_ptsz_v11020`.
    pub const CUDA_FREE_ASYNC_PTSZ_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_ptsz_v11020
            as u32,
    );

    /// Callback identifier for `cudaMemPoolTrimTo_v11020`.
    pub const CUDA_MEM_POOL_TRIM_TO_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolTrimTo_v11020
            as u32,
    );

    /// Callback identifier for `cudaMemPoolSetAttribute_v11020`.
    pub const CUDA_MEM_POOL_SET_ATTRIBUTE_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolSetAttribute_v11020 as u32);

    /// Callback identifier for `cudaMemPoolGetAttribute_v11020`.
    pub const CUDA_MEM_POOL_GET_ATTRIBUTE_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolGetAttribute_v11020 as u32);

    /// Callback identifier for `cudaMemPoolSetAccess_v11020`.
    pub const CUDA_MEM_POOL_SET_ACCESS_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolSetAccess_v11020
            as u32,
    );

    /// Callback identifier for `cudaArrayGetPlane_v11020`.
    pub const CUDA_ARRAY_GET_PLANE_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetPlane_v11020
            as u32,
    );

    /// Callback identifier for `cudaMemPoolGetAccess_v11020`.
    pub const CUDA_MEM_POOL_GET_ACCESS_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolGetAccess_v11020
            as u32,
    );

    /// Callback identifier for `cudaMemPoolCreate_v11020`.
    pub const CUDA_MEM_POOL_CREATE_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolCreate_v11020
            as u32,
    );

    /// Callback identifier for `cudaMemPoolDestroy_v11020`.
    pub const CUDA_MEM_POOL_DESTROY_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolDestroy_v11020
            as u32,
    );

    /// Callback identifier for `cudaDeviceSetMemPool_v11020`.
    pub const CUDA_DEVICE_SET_MEM_POOL_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetMemPool_v11020
            as u32,
    );

    /// Callback identifier for `cudaDeviceGetMemPool_v11020`.
    pub const CUDA_DEVICE_GET_MEM_POOL_V11020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetMemPool_v11020
            as u32,
    );

    /// Callback identifier for `cudaMemPoolExportToShareableHandle_v11020`.
    pub const CUDA_MEM_POOL_EXPORT_TO_SHAREABLE_HANDLE_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolExportToShareableHandle_v11020 as u32);

    /// Callback identifier for `cudaMemPoolImportFromShareableHandle_v11020`.
    pub const CUDA_MEM_POOL_IMPORT_FROM_SHAREABLE_HANDLE_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolImportFromShareableHandle_v11020 as u32);

    /// Callback identifier for `cudaMemPoolExportPointer_v11020`.
    pub const CUDA_MEM_POOL_EXPORT_POINTER_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolExportPointer_v11020 as u32);

    /// Callback identifier for `cudaMemPoolImportPointer_v11020`.
    pub const CUDA_MEM_POOL_IMPORT_POINTER_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPoolImportPointer_v11020 as u32);

    /// Callback identifier for `cudaMallocFromPoolAsync_v11020`.
    pub const CUDA_MALLOC_FROM_POOL_ASYNC_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_v11020 as u32);

    /// Callback identifier for `cudaMallocFromPoolAsync_ptsz_v11020`.
    pub const CUDA_MALLOC_FROM_POOL_ASYNC_PTSZ_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMallocFromPoolAsync_ptsz_v11020 as u32);

    /// Callback identifier for `cudaSignalExternalSemaphoresAsync_v11020`.
    pub const CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_v11020 as u32);

    /// Callback identifier for `cudaSignalExternalSemaphoresAsync_ptsz_v11020`.
    pub const CUDA_SIGNAL_EXTERNAL_SEMAPHORES_ASYNC_PTSZ_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaSignalExternalSemaphoresAsync_ptsz_v11020 as u32);

    /// Callback identifier for `cudaWaitExternalSemaphoresAsync_v11020`.
    pub const CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_v11020 as u32);

    /// Callback identifier for `cudaWaitExternalSemaphoresAsync_ptsz_v11020`.
    pub const CUDA_WAIT_EXTERNAL_SEMAPHORES_ASYNC_PTSZ_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_ptsz_v11020 as u32);

    /// Callback identifier for `cudaGraphAddExternalSemaphoresSignalNode_v11020`.
    pub const CUDA_GRAPH_ADD_EXTERNAL_SEMAPHORES_SIGNAL_NODE_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddExternalSemaphoresSignalNode_v11020 as u32);

    /// Callback identifier for `cudaGraphExternalSemaphoresSignalNodeGetParams_v11020`.
    pub const CUDA_GRAPH_EXTERNAL_SEMAPHORES_SIGNAL_NODE_GET_PARAMS_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresSignalNodeGetParams_v11020 as u32);

    /// Callback identifier for `cudaGraphExternalSemaphoresSignalNodeSetParams_v11020`.
    pub const CUDA_GRAPH_EXTERNAL_SEMAPHORES_SIGNAL_NODE_SET_PARAMS_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresSignalNodeSetParams_v11020 as u32);

    /// Callback identifier for `cudaGraphAddExternalSemaphoresWaitNode_v11020`.
    pub const CUDA_GRAPH_ADD_EXTERNAL_SEMAPHORES_WAIT_NODE_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddExternalSemaphoresWaitNode_v11020 as u32);

    /// Callback identifier for `cudaGraphExternalSemaphoresWaitNodeGetParams_v11020`.
    pub const CUDA_GRAPH_EXTERNAL_SEMAPHORES_WAIT_NODE_GET_PARAMS_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresWaitNodeGetParams_v11020 as u32);

    /// Callback identifier for `cudaGraphExternalSemaphoresWaitNodeSetParams_v11020`.
    pub const CUDA_GRAPH_EXTERNAL_SEMAPHORES_WAIT_NODE_SET_PARAMS_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExternalSemaphoresWaitNodeSetParams_v11020 as u32);

    /// Callback identifier for `cudaGraphExecExternalSemaphoresSignalNodeSetParams_v11020`.
    pub const CUDA_GRAPH_EXEC_EXTERNAL_SEMAPHORES_SIGNAL_NODE_SET_PARAMS_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecExternalSemaphoresSignalNodeSetParams_v11020 as u32);

    /// Callback identifier for `cudaGraphExecExternalSemaphoresWaitNodeSetParams_v11020`.
    pub const CUDA_GRAPH_EXEC_EXTERNAL_SEMAPHORES_WAIT_NODE_SET_PARAMS_V11020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecExternalSemaphoresWaitNodeSetParams_v11020 as u32);

    /// Callback identifier for `cudaDeviceFlushGPUDirectRDMAWrites_v11030`.
    pub const CUDA_DEVICE_FLUSH_GPU_DIRECT_RDMA_WRITES_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceFlushGPUDirectRDMAWrites_v11030 as u32);

    /// Callback identifier for `cudaGetDriverEntryPoint_v11030`.
    pub const CUDA_GET_DRIVER_ENTRY_POINT_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPoint_v11030 as u32);

    /// Callback identifier for `cudaGetDriverEntryPoint_ptsz_v11030`.
    pub const CUDA_GET_DRIVER_ENTRY_POINT_PTSZ_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPoint_ptsz_v11030 as u32);

    /// Callback identifier for `cudaGraphDebugDotPrint_v11030`.
    pub const CUDA_GRAPH_DEBUG_DOT_PRINT_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphDebugDotPrint_v11030 as u32);

    /// Callback identifier for `cudaStreamGetCaptureInfo_v2_v11030`.
    pub const CUDA_STREAM_GET_CAPTURE_INFO_V2_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_v11030 as u32);

    /// Callback identifier for `cudaStreamGetCaptureInfo_v2_ptsz_v11030`.
    pub const CUDA_STREAM_GET_CAPTURE_INFO_V2_PTSZ_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_ptsz_v11030 as u32);

    /// Callback identifier for `cudaStreamUpdateCaptureDependencies_v11030`.
    pub const CUDA_STREAM_UPDATE_CAPTURE_DEPENDENCIES_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v11030 as u32);

    /// Callback identifier for `cudaStreamUpdateCaptureDependencies_ptsz_v11030`.
    pub const CUDA_STREAM_UPDATE_CAPTURE_DEPENDENCIES_PTSZ_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_ptsz_v11030 as u32);

    /// Callback identifier for `cudaUserObjectCreate_v11030`.
    pub const CUDA_USER_OBJECT_CREATE_V11030: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaUserObjectCreate_v11030
            as u32,
    );

    /// Callback identifier for `cudaUserObjectRetain_v11030`.
    pub const CUDA_USER_OBJECT_RETAIN_V11030: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaUserObjectRetain_v11030
            as u32,
    );

    /// Callback identifier for `cudaUserObjectRelease_v11030`.
    pub const CUDA_USER_OBJECT_RELEASE_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaUserObjectRelease_v11030 as u32);

    /// Callback identifier for `cudaGraphRetainUserObject_v11030`.
    pub const CUDA_GRAPH_RETAIN_USER_OBJECT_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphRetainUserObject_v11030 as u32);

    /// Callback identifier for `cudaGraphReleaseUserObject_v11030`.
    pub const CUDA_GRAPH_RELEASE_USER_OBJECT_V11030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphReleaseUserObject_v11030 as u32);

    /// Callback identifier for `cudaGraphInstantiateWithFlags_v11040`.
    pub const CUDA_GRAPH_INSTANTIATE_WITH_FLAGS_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithFlags_v11040 as u32);

    /// Callback identifier for `cudaGraphAddMemAllocNode_v11040`.
    pub const CUDA_GRAPH_ADD_MEM_ALLOC_NODE_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemAllocNode_v11040 as u32);

    /// Callback identifier for `cudaGraphMemAllocNodeGetParams_v11040`.
    pub const CUDA_GRAPH_MEM_ALLOC_NODE_GET_PARAMS_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemAllocNodeGetParams_v11040 as u32);

    /// Callback identifier for `cudaGraphAddMemFreeNode_v11040`.
    pub const CUDA_GRAPH_ADD_MEM_FREE_NODE_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddMemFreeNode_v11040 as u32);

    /// Callback identifier for `cudaGraphMemFreeNodeGetParams_v11040`.
    pub const CUDA_GRAPH_MEM_FREE_NODE_GET_PARAMS_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphMemFreeNodeGetParams_v11040 as u32);

    /// Callback identifier for `cudaDeviceGraphMemTrim_v11040`.
    pub const CUDA_DEVICE_GRAPH_MEM_TRIM_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGraphMemTrim_v11040 as u32);

    /// Callback identifier for `cudaDeviceGetGraphMemAttribute_v11040`.
    pub const CUDA_DEVICE_GET_GRAPH_MEM_ATTRIBUTE_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetGraphMemAttribute_v11040 as u32);

    /// Callback identifier for `cudaDeviceSetGraphMemAttribute_v11040`.
    pub const CUDA_DEVICE_SET_GRAPH_MEM_ATTRIBUTE_V11040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetGraphMemAttribute_v11040 as u32);

    /// Callback identifier for `cudaGraphNodeSetEnabled_v11060`.
    pub const CUDA_GRAPH_NODE_SET_ENABLED_V11060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetEnabled_v11060 as u32);

    /// Callback identifier for `cudaGraphNodeGetEnabled_v11060`.
    pub const CUDA_GRAPH_NODE_GET_ENABLED_V11060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetEnabled_v11060 as u32);

    /// Callback identifier for `cudaArrayGetMemoryRequirements_v11060`.
    pub const CUDA_ARRAY_GET_MEMORY_REQUIREMENTS_V11060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaArrayGetMemoryRequirements_v11060 as u32);

    /// Callback identifier for `cudaMipmappedArrayGetMemoryRequirements_v11060`.
    pub const CUDA_MIPMAPPED_ARRAY_GET_MEMORY_REQUIREMENTS_V11060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMipmappedArrayGetMemoryRequirements_v11060 as u32);

    /// Callback identifier for `cudaLaunchKernelExC_v11060`.
    pub const CUDA_LAUNCH_KERNEL_EX_C_V11060: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060
            as u32,
    );

    /// Callback identifier for `cudaLaunchKernelExC_ptsz_v11060`.
    pub const CUDA_LAUNCH_KERNEL_EX_C_PTSZ_V11060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060 as u32);

    /// Callback identifier for `cudaOccupancyMaxPotentialClusterSize_v11070`.
    pub const CUDA_OCCUPANCY_MAX_POTENTIAL_CLUSTER_SIZE_V11070: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxPotentialClusterSize_v11070 as u32);

    /// Callback identifier for `cudaOccupancyMaxActiveClusters_v11070`.
    pub const CUDA_OCCUPANCY_MAX_ACTIVE_CLUSTERS_V11070: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaOccupancyMaxActiveClusters_v11070 as u32);

    /// Callback identifier for `cudaCreateTextureObject_v2_v11080`.
    pub const CUDA_CREATE_TEXTURE_OBJECT_V2_V11080: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaCreateTextureObject_v2_v11080 as u32);

    /// Callback identifier for `cudaGetTextureObjectTextureDesc_v2_v11080`.
    pub const CUDA_GET_TEXTURE_OBJECT_TEXTURE_DESC_V2_V11080: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureObjectTextureDesc_v2_v11080 as u32);

    /// Callback identifier for `cudaGraphInstantiateWithParams_v12000`.
    pub const CUDA_GRAPH_INSTANTIATE_WITH_PARAMS_V12000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_v12000 as u32);

    /// Callback identifier for `cudaGraphInstantiateWithParams_ptsz_v12000`.
    pub const CUDA_GRAPH_INSTANTIATE_WITH_PARAMS_PTSZ_V12000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiateWithParams_ptsz_v12000 as u32);

    /// Callback identifier for `cudaGraphExecGetFlags_v12000`.
    pub const CUDA_GRAPH_EXEC_GET_FLAGS_V12000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecGetFlags_v12000 as u32);

    /// Callback identifier for `cudaGetKernel_v12000`.
    pub const CUDA_GET_KERNEL_V12000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetKernel_v12000
            as u32,
    );

    /// Callback identifier for `cudaGetDeviceProperties_v12000`.
    pub const CUDA_GET_DEVICE_PROPERTIES_V12000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v12000 as u32);

    /// Callback identifier for `cudaStreamGetId_v12000`.
    pub const CUDA_STREAM_GET_ID_V12000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetId_v12000
            as u32,
    );

    /// Callback identifier for `cudaStreamGetId_ptsz_v12000`.
    pub const CUDA_STREAM_GET_ID_PTSZ_V12000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetId_ptsz_v12000
            as u32,
    );

    /// Callback identifier for `cudaGraphInstantiate_v12000`.
    pub const CUDA_GRAPH_INSTANTIATE_V12000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphInstantiate_v12000
            as u32,
    );

    /// Callback identifier for `cudaInitDevice_v12000`.
    pub const CUDA_INIT_DEVICE_V12000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaInitDevice_v12000
            as u32,
    );

    /// Callback identifier for `cudaGraphAddNode_v12020`.
    pub const CUDA_GRAPH_ADD_NODE_V12020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddNode_v12020
            as u32,
    );

    /// Callback identifier for `cudaGraphNodeSetParams_v12020`.
    pub const CUDA_GRAPH_NODE_SET_PARAMS_V12020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeSetParams_v12020 as u32);

    /// Callback identifier for `cudaGraphExecNodeSetParams_v12020`.
    pub const CUDA_GRAPH_EXEC_NODE_SET_PARAMS_V12020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecNodeSetParams_v12020 as u32);

    /// Callback identifier for `cudaMemAdvise_v12020`.
    pub const CUDA_MEM_ADVISE_V12020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemAdvise_v12020
            as u32,
    );

    /// Callback identifier for `cudaMemPrefetchAsync_v12020`.
    pub const CUDA_MEM_PREFETCH_ASYNC_V12020: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v12020
            as u32,
    );

    /// Callback identifier for `cudaMemPrefetchAsync_ptsz_v12020`.
    pub const CUDA_MEM_PREFETCH_ASYNC_PTSZ_V12020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_ptsz_v12020 as u32);

    /// Callback identifier for `cudaFuncGetName_v12030`.
    pub const CUDA_FUNC_GET_NAME_V12030: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetName_v12030
            as u32,
    );

    /// Callback identifier for `cudaStreamBeginCaptureToGraph_v12030`.
    pub const CUDA_STREAM_BEGIN_CAPTURE_TO_GRAPH_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_v12030 as u32);

    /// Callback identifier for `cudaStreamBeginCaptureToGraph_ptsz_v12030`.
    pub const CUDA_STREAM_BEGIN_CAPTURE_TO_GRAPH_PTSZ_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCaptureToGraph_ptsz_v12030 as u32);

    /// Callback identifier for `cudaGraphConditionalHandleCreate_v12030`.
    pub const CUDA_GRAPH_CONDITIONAL_HANDLE_CREATE_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphConditionalHandleCreate_v12030 as u32);

    /// Callback identifier for `cudaGraphGetEdges_v12030`.
    pub const CUDA_GRAPH_GET_EDGES_V12030: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetEdges_v12030
            as u32,
    );

    /// Callback identifier for `cudaGraphNodeGetDependencies_v12030`.
    pub const CUDA_GRAPH_NODE_GET_DEPENDENCIES_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependencies_v12030 as u32);

    /// Callback identifier for `cudaGraphNodeGetDependentNodes_v12030`.
    pub const CUDA_GRAPH_NODE_GET_DEPENDENT_NODES_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependentNodes_v12030 as u32);

    /// Callback identifier for `cudaGraphAddDependencies_v12030`.
    pub const CUDA_GRAPH_ADD_DEPENDENCIES_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddDependencies_v12030 as u32);

    /// Callback identifier for `cudaGraphRemoveDependencies_v12030`.
    pub const CUDA_GRAPH_REMOVE_DEPENDENCIES_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphRemoveDependencies_v12030 as u32);

    /// Callback identifier for `cudaGraphAddNode_v12030`.
    pub const CUDA_GRAPH_ADD_NODE_V12030: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddNode_v12030
            as u32,
    );

    /// Callback identifier for `cudaStreamGetCaptureInfo_v12030`.
    pub const CUDA_STREAM_GET_CAPTURE_INFO_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v12030 as u32);

    /// Callback identifier for `cudaStreamGetCaptureInfo_ptsz_v12030`.
    pub const CUDA_STREAM_GET_CAPTURE_INFO_PTSZ_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_ptsz_v12030 as u32);

    /// Callback identifier for `cudaStreamUpdateCaptureDependencies_v12030`.
    pub const CUDA_STREAM_UPDATE_CAPTURE_DEPENDENCIES_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v12030 as u32);

    /// Callback identifier for `cudaStreamUpdateCaptureDependencies_ptsz_v12030`.
    pub const CUDA_STREAM_UPDATE_CAPTURE_DEPENDENCIES_PTSZ_V12030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_ptsz_v12030 as u32);

    /// Callback identifier for `cudaDeviceRegisterAsyncNotification_v12040`.
    pub const CUDA_DEVICE_REGISTER_ASYNC_NOTIFICATION_V12040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceRegisterAsyncNotification_v12040 as u32);

    /// Callback identifier for `cudaDeviceUnregisterAsyncNotification_v12040`.
    pub const CUDA_DEVICE_UNREGISTER_ASYNC_NOTIFICATION_V12040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceUnregisterAsyncNotification_v12040 as u32);

    /// Callback identifier for `cudaFuncGetParamInfo_v12040`.
    pub const CUDA_FUNC_GET_PARAM_INFO_V12040: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetParamInfo_v12040
            as u32,
    );

    /// Callback identifier for `cudaGetDriverEntryPointByVersion_v12050`.
    pub const CUDA_GET_DRIVER_ENTRY_POINT_BY_VERSION_V12050: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPointByVersion_v12050 as u32);

    /// Callback identifier for `cudaGetDriverEntryPointByVersion_ptsz_v12050`.
    pub const CUDA_GET_DRIVER_ENTRY_POINT_BY_VERSION_PTSZ_V12050: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetDriverEntryPointByVersion_ptsz_v12050 as u32);

    /// Callback identifier for `cudaLibraryLoadData_v12060`.
    pub const CUDA_LIBRARY_LOAD_DATA_V12060: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryLoadData_v12060
            as u32,
    );

    /// Callback identifier for `cudaLibraryLoadFromFile_v12060`.
    pub const CUDA_LIBRARY_LOAD_FROM_FILE_V12060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryLoadFromFile_v12060 as u32);

    /// Callback identifier for `cudaLibraryUnload_v12060`.
    pub const CUDA_LIBRARY_UNLOAD_V12060: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryUnload_v12060
            as u32,
    );

    /// Callback identifier for `cudaLibraryGetKernel_v12060`.
    pub const CUDA_LIBRARY_GET_KERNEL_V12060: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetKernel_v12060
            as u32,
    );

    /// Callback identifier for `cudaLibraryGetGlobal_v12060`.
    pub const CUDA_LIBRARY_GET_GLOBAL_V12060: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetGlobal_v12060
            as u32,
    );

    /// Callback identifier for `cudaLibraryGetManaged_v12060`.
    pub const CUDA_LIBRARY_GET_MANAGED_V12060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetManaged_v12060 as u32);

    /// Callback identifier for `cudaLibraryGetUnifiedFunction_v12060`.
    pub const CUDA_LIBRARY_GET_UNIFIED_FUNCTION_V12060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetUnifiedFunction_v12060 as u32);

    /// Callback identifier for `cudaLibraryGetKernelCount_v12060`.
    pub const CUDA_LIBRARY_GET_KERNEL_COUNT_V12060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryGetKernelCount_v12060 as u32);

    /// Callback identifier for `cudaLibraryEnumerateKernels_v12060`.
    pub const CUDA_LIBRARY_ENUMERATE_KERNELS_V12060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLibraryEnumerateKernels_v12060 as u32);

    /// Callback identifier for `cudaKernelSetAttributeForDevice_v12060`.
    pub const CUDA_KERNEL_SET_ATTRIBUTE_FOR_DEVICE_V12060: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaKernelSetAttributeForDevice_v12060 as u32);

    /// Callback identifier for `cudaStreamGetDevice_v12080`.
    pub const CUDA_STREAM_GET_DEVICE_V12080: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetDevice_v12080
            as u32,
    );

    /// Callback identifier for `cudaStreamGetDevice_ptsz_v12080`.
    pub const CUDA_STREAM_GET_DEVICE_PTSZ_V12080: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetDevice_ptsz_v12080 as u32);

    /// Callback identifier for `cudaMemcpyBatchAsync_v12080`.
    pub const CUDA_MEMCPY_BATCH_ASYNC_V12080: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyBatchAsync_v12080
            as u32,
    );

    /// Callback identifier for `cudaMemcpyBatchAsync_ptsz_v12080`.
    pub const CUDA_MEMCPY_BATCH_ASYNC_PTSZ_V12080: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyBatchAsync_ptsz_v12080 as u32);

    /// Callback identifier for `cudaMemcpy3DBatchAsync_v12080`.
    pub const CUDA_MEMCPY3_D_BATCH_ASYNC_V12080: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DBatchAsync_v12080 as u32);

    /// Callback identifier for `cudaMemcpy3DBatchAsync_ptsz_v12080`.
    pub const CUDA_MEMCPY3_D_BATCH_ASYNC_PTSZ_V12080: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DBatchAsync_ptsz_v12080 as u32);

    /// Callback identifier for `cudaEventElapsedTime_v12080`.
    pub const CUDA_EVENT_ELAPSED_TIME_V12080: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v12080
            as u32,
    );

    /// Callback identifier for `cudaMemPrefetchBatchAsync_v13000`.
    pub const CUDA_MEM_PREFETCH_BATCH_ASYNC_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchBatchAsync_v13000 as u32);

    /// Callback identifier for `cudaMemPrefetchBatchAsync_ptsz_v13000`.
    pub const CUDA_MEM_PREFETCH_BATCH_ASYNC_PTSZ_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchBatchAsync_ptsz_v13000 as u32);

    /// Callback identifier for `cudaMemDiscardBatchAsync_v13000`.
    pub const CUDA_MEM_DISCARD_BATCH_ASYNC_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemDiscardBatchAsync_v13000 as u32);

    /// Callback identifier for `cudaMemDiscardBatchAsync_ptsz_v13000`.
    pub const CUDA_MEM_DISCARD_BATCH_ASYNC_PTSZ_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemDiscardBatchAsync_ptsz_v13000 as u32);

    /// Callback identifier for `cudaMemDiscardAndPrefetchBatchAsync_v13000`.
    pub const CUDA_MEM_DISCARD_AND_PREFETCH_BATCH_ASYNC_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemDiscardAndPrefetchBatchAsync_v13000 as u32);

    /// Callback identifier for `cudaMemDiscardAndPrefetchBatchAsync_ptsz_v13000`.
    pub const CUDA_MEM_DISCARD_AND_PREFETCH_BATCH_ASYNC_PTSZ_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemDiscardAndPrefetchBatchAsync_ptsz_v13000 as u32);

    /// Callback identifier for `cudaGetExportTable_v13000`.
    pub const CUDA_GET_EXPORT_TABLE_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGetExportTable_v13000
            as u32,
    );

    /// Callback identifier for `__cudaRegisterFatBinary_v13000`.
    pub const __CUDA_REGISTER_FAT_BINARY_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterFatBinary_v13000 as u32);

    /// Callback identifier for `__cudaRegisterFatBinaryEnd_v13000`.
    pub const __CUDA_REGISTER_FAT_BINARY_END_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterFatBinaryEnd_v13000 as u32);

    /// Callback identifier for `__cudaUnregisterFatBinary_v13000`.
    pub const __CUDA_UNREGISTER_FAT_BINARY_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaUnregisterFatBinary_v13000 as u32);

    /// Callback identifier for `__cudaRegisterVar_v13000`.
    pub const __CUDA_REGISTER_VAR_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterVar_v13000
            as u32,
    );

    /// Callback identifier for `__cudaRegisterHostVar_v13000`.
    pub const __CUDA_REGISTER_HOST_VAR_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterHostVar_v13000 as u32);

    /// Callback identifier for `__cudaRegisterFunction_v13000`.
    pub const __CUDA_REGISTER_FUNCTION_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterFunction_v13000 as u32);

    /// Callback identifier for `__cudaRegisterManagedVar_v13000`.
    pub const __CUDA_REGISTER_MANAGED_VAR_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterManagedVar_v13000 as u32);

    /// Callback identifier for `__cudaInitModule_v13000`.
    pub const __CUDA_INIT_MODULE_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaInitModule_v13000
            as u32,
    );

    /// Callback identifier for `__cudaPushCallConfiguration_v13000`.
    pub const __CUDA_PUSH_CALL_CONFIGURATION_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaPushCallConfiguration_v13000 as u32);

    /// Callback identifier for `__cudaPopCallConfiguration_v13000`.
    pub const __CUDA_POP_CALL_CONFIGURATION_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaPopCallConfiguration_v13000 as u32);

    /// Callback identifier for `__cudaGetKernel_v13000`.
    pub const __CUDA_GET_KERNEL_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaGetKernel_v13000
            as u32,
    );

    /// Callback identifier for `__cudaLaunchKernel_v13000`.
    pub const __CUDA_LAUNCH_KERNEL_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaLaunchKernel_v13000
            as u32,
    );

    /// Callback identifier for `__cudaLaunchKernel_ptsz_v13000`.
    pub const __CUDA_LAUNCH_KERNEL_PTSZ_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaLaunchKernel_ptsz_v13000 as u32);

    /// Callback identifier for `__cudaRegisterUnifiedTable_v13000`.
    pub const __CUDA_REGISTER_UNIFIED_TABLE_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaRegisterUnifiedTable_v13000 as u32);

    /// Callback identifier for `__cudaGetProcAddress_v13000`.
    pub const __CUDA_GET_PROC_ADDRESS_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID___cudaGetProcAddress_v13000
            as u32,
    );

    /// Callback identifier for `cudaMemcpyBatchAsync_v13000`.
    pub const CUDA_MEMCPY_BATCH_ASYNC_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyBatchAsync_v13000
            as u32,
    );

    /// Callback identifier for `cudaMemcpyBatchAsync_ptsz_v13000`.
    pub const CUDA_MEMCPY_BATCH_ASYNC_PTSZ_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyBatchAsync_ptsz_v13000 as u32);

    /// Callback identifier for `cudaMemcpy3DBatchAsync_v13000`.
    pub const CUDA_MEMCPY3_D_BATCH_ASYNC_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DBatchAsync_v13000 as u32);

    /// Callback identifier for `cudaMemcpy3DBatchAsync_ptsz_v13000`.
    pub const CUDA_MEMCPY3_D_BATCH_ASYNC_PTSZ_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DBatchAsync_ptsz_v13000 as u32);

    /// Callback identifier for `cudaLogsRegisterCallback_v13000`.
    pub const CUDA_LOGS_REGISTER_CALLBACK_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLogsRegisterCallback_v13000 as u32);

    /// Callback identifier for `cudaLogsUnregisterCallback_v13000`.
    pub const CUDA_LOGS_UNREGISTER_CALLBACK_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLogsUnregisterCallback_v13000 as u32);

    /// Callback identifier for `cudaLogsCurrent_v13000`.
    pub const CUDA_LOGS_CURRENT_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLogsCurrent_v13000
            as u32,
    );

    /// Callback identifier for `cudaLogsDumpToFile_v13000`.
    pub const CUDA_LOGS_DUMP_TO_FILE_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLogsDumpToFile_v13000
            as u32,
    );

    /// Callback identifier for `cudaLogsDumpToMemory_v13000`.
    pub const CUDA_LOGS_DUMP_TO_MEMORY_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLogsDumpToMemory_v13000
            as u32,
    );

    /// Callback identifier for `cudaMemGetDefaultMemPool_v13000`.
    pub const CUDA_MEM_GET_DEFAULT_MEM_POOL_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemGetDefaultMemPool_v13000 as u32);

    /// Callback identifier for `cudaMemGetMemPool_v13000`.
    pub const CUDA_MEM_GET_MEM_POOL_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemGetMemPool_v13000
            as u32,
    );

    /// Callback identifier for `cudaMemSetMemPool_v13000`.
    pub const CUDA_MEM_SET_MEM_POOL_V13000: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemSetMemPool_v13000
            as u32,
    );

    /// Callback identifier for `cudaDeviceGetHostAtomicCapabilities_v13000`.
    pub const CUDA_DEVICE_GET_HOST_ATOMIC_CAPABILITIES_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetHostAtomicCapabilities_v13000 as u32);

    /// Callback identifier for `cudaDeviceGetP2PAtomicCapabilities_v13000`.
    pub const CUDA_DEVICE_GET_P2_P_ATOMIC_CAPABILITIES_V13000: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetP2PAtomicCapabilities_v13000 as u32);

    /// Callback identifier for `cudaDeviceGetDevResource_v13010`.
    pub const CUDA_DEVICE_GET_DEV_RESOURCE_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetDevResource_v13010 as u32);

    /// Callback identifier for `cudaDevSmResourceSplitByCount_v13010`.
    pub const CUDA_DEV_SM_RESOURCE_SPLIT_BY_COUNT_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDevSmResourceSplitByCount_v13010 as u32);

    /// Callback identifier for `cudaDevResourceGenerateDesc_v13010`.
    pub const CUDA_DEV_RESOURCE_GENERATE_DESC_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDevResourceGenerateDesc_v13010 as u32);

    /// Callback identifier for `cudaGreenCtxCreate_v13010`.
    pub const CUDA_GREEN_CTX_CREATE_V13010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGreenCtxCreate_v13010
            as u32,
    );

    /// Callback identifier for `cudaExecutionCtxDestroy_v13010`.
    pub const CUDA_EXECUTION_CTX_DESTROY_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxDestroy_v13010 as u32);

    /// Callback identifier for `cudaExecutionCtxGetDevResource_v13010`.
    pub const CUDA_EXECUTION_CTX_GET_DEV_RESOURCE_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxGetDevResource_v13010 as u32);

    /// Callback identifier for `cudaExecutionCtxGetDevice_v13010`.
    pub const CUDA_EXECUTION_CTX_GET_DEVICE_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxGetDevice_v13010 as u32);

    /// Callback identifier for `cudaExecutionCtxGetId_v13010`.
    pub const CUDA_EXECUTION_CTX_GET_ID_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxGetId_v13010 as u32);

    /// Callback identifier for `cuda531`.
    pub const CUDA531: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda531 as u32);

    /// Callback identifier for `cuda532`.
    pub const CUDA532: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda532 as u32);

    /// Callback identifier for `cudaExecutionCtxStreamCreate_v13010`.
    pub const CUDA_EXECUTION_CTX_STREAM_CREATE_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxStreamCreate_v13010 as u32);

    /// Callback identifier for `cudaExecutionCtxSynchronize_v13010`.
    pub const CUDA_EXECUTION_CTX_SYNCHRONIZE_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxSynchronize_v13010 as u32);

    /// Callback identifier for `cudaGraphConditionalHandleCreate_v2_v13010`.
    pub const CUDA_GRAPH_CONDITIONAL_HANDLE_CREATE_V2_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphConditionalHandleCreate_v2_v13010 as u32);

    /// Callback identifier for `cudaStreamGetDevResource_v13010`.
    pub const CUDA_STREAM_GET_DEV_RESOURCE_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetDevResource_v13010 as u32);

    /// Callback identifier for `cudaStreamGetDevResource_ptsz_v13010`.
    pub const CUDA_STREAM_GET_DEV_RESOURCE_PTSZ_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetDevResource_ptsz_v13010 as u32);

    /// Callback identifier for `cudaGraphNodeGetContainingGraph_v13010`.
    pub const CUDA_GRAPH_NODE_GET_CONTAINING_GRAPH_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetContainingGraph_v13010 as u32);

    /// Callback identifier for `cudaGraphNodeGetLocalId_v13010`.
    pub const CUDA_GRAPH_NODE_GET_LOCAL_ID_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetLocalId_v13010 as u32);

    /// Callback identifier for `cudaGraphNodeGetToolsId_v13010`.
    pub const CUDA_GRAPH_NODE_GET_TOOLS_ID_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetToolsId_v13010 as u32);

    /// Callback identifier for `cudaGraphGetId_v13010`.
    pub const CUDA_GRAPH_GET_ID_V13010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetId_v13010
            as u32,
    );

    /// Callback identifier for `cudaGraphExecGetId_v13010`.
    pub const CUDA_GRAPH_EXEC_GET_ID_V13010: Self = Self(
        sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphExecGetId_v13010
            as u32,
    );

    /// Callback identifier for `cudaExecutionCtxRecordEvent_v13010`.
    pub const CUDA_EXECUTION_CTX_RECORD_EVENT_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxRecordEvent_v13010 as u32);

    /// Callback identifier for `cudaExecutionCtxWaitEvent_v13010`.
    pub const CUDA_EXECUTION_CTX_WAIT_EVENT_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaExecutionCtxWaitEvent_v13010 as u32);

    /// Callback identifier for `cuda545`.
    pub const CUDA545: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda545 as u32);

    /// Callback identifier for `cuda546`.
    pub const CUDA546: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda546 as u32);

    /// Callback identifier for `cuda547`.
    pub const CUDA547: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda547 as u32);

    /// Callback identifier for `cudaDeviceGetExecutionCtx_v13010`.
    pub const CUDA_DEVICE_GET_EXECUTION_CTX_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetExecutionCtx_v13010 as u32);

    /// Callback identifier for `cudaDevSmResourceSplit_v13010`.
    pub const CUDA_DEV_SM_RESOURCE_SPLIT_V13010: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaDevSmResourceSplit_v13010 as u32);

    /// Callback identifier for `cudaLaunchHostFunc_v2_v13020`.
    pub const CUDA_LAUNCH_HOST_FUNC_V2_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v2_v13020 as u32);

    /// Callback identifier for `cudaLaunchHostFunc_v2_ptsz_v13020`.
    pub const CUDA_LAUNCH_HOST_FUNC_V2_PTSZ_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v2_ptsz_v13020 as u32);

    /// Callback identifier for `cudaFuncGetParamCount_v13020`.
    pub const CUDA_FUNC_GET_PARAM_COUNT_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetParamCount_v13020 as u32);

    /// Callback identifier for `cuda553`.
    pub const CUDA553: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda553 as u32);

    /// Callback identifier for `cudaMemcpyWithAttributesAsync_v13020`.
    pub const CUDA_MEMCPY_WITH_ATTRIBUTES_ASYNC_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyWithAttributesAsync_v13020 as u32);

    /// Callback identifier for `cudaMemcpyWithAttributesAsync_ptsz_v13020`.
    pub const CUDA_MEMCPY_WITH_ATTRIBUTES_ASYNC_PTSZ_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyWithAttributesAsync_ptsz_v13020 as u32);

    /// Callback identifier for `cudaMemcpy3DWithAttributesAsync_v13020`.
    pub const CUDA_MEMCPY3_D_WITH_ATTRIBUTES_ASYNC_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DWithAttributesAsync_v13020 as u32);

    /// Callback identifier for `cudaMemcpy3DWithAttributesAsync_ptsz_v13020`.
    pub const CUDA_MEMCPY3_D_WITH_ATTRIBUTES_ASYNC_PTSZ_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DWithAttributesAsync_ptsz_v13020 as u32);

    /// Callback identifier for `cudaGraphNodeGetParams_v13020`.
    pub const CUDA_GRAPH_NODE_GET_PARAMS_V13020: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetParams_v13020 as u32);

    /// Callback identifier for `cuda559`.
    pub const CUDA559: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda559 as u32);

    /// Callback identifier for `cuda560`.
    pub const CUDA560: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda560 as u32);

    /// Callback identifier for `cuda561`.
    pub const CUDA561: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda561 as u32);

    /// Callback identifier for `cudaStreamBeginRecaptureToGraph_v13030`.
    pub const CUDA_STREAM_BEGIN_RECAPTURE_TO_GRAPH_V13030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginRecaptureToGraph_v13030 as u32);

    /// Callback identifier for `cudaStreamBeginRecaptureToGraph_ptsz_v13030`.
    pub const CUDA_STREAM_BEGIN_RECAPTURE_TO_GRAPH_PTSZ_V13030: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginRecaptureToGraph_ptsz_v13030 as u32);

    /// Callback identifier for `cuda564`.
    pub const CUDA564: Self =
        Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cuda564 as u32);

    /// Callback identifier for `cudaMemGetLocationInfo_v13040`.
    pub const CUDA_MEM_GET_LOCATION_INFO_V13040: Self = Self(sys::CUpti_runtime_api_trace_cbid_enum::CUPTI_RUNTIME_TRACE_CBID_cudaMemGetLocationInfo_v13040 as u32);
}

impl From<u32> for RuntimeCallbackId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<RuntimeCallbackId> for CallbackId {
    fn from(value: RuntimeCallbackId) -> Self {
        Self::from(value.get())
    }
}
