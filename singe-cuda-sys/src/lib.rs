//! Raw FFI bindings for CUDA Driver, Runtime, NVRTC, NVVM, NVTX, and library types.
//!
//! Prefer the safe `singe-cuda` crate unless direct NVIDIA ABI access is required.

#![allow(deprecated, warnings, unused_qualifications, clippy::all)]

#[cfg(feature = "driver")]
pub mod driver {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    #[cfg(feature = "driver_13_2")]
    include!("driver_sys_13020.rs");

    #[cfg(feature = "driver_13_2")]
    pub use self::{
        cuArray3DCreate_v2 as cuArray3DCreate, cuArray3DGetDescriptor_v2 as cuArray3DGetDescriptor,
        cuArrayCreate_v2 as cuArrayCreate, cuArrayGetDescriptor_v2 as cuArrayGetDescriptor,
        cuCtxCreate_v4 as cuCtxCreate, cuCtxDestroy_v2 as cuCtxDestroy,
        cuCtxPopCurrent_v2 as cuCtxPopCurrent, cuCtxPushCurrent_v2 as cuCtxPushCurrent,
        cuDeviceGetUuid_v2 as cuDeviceGetUuid,
        cuDevicePrimaryCtxRelease_v2 as cuDevicePrimaryCtxRelease,
        cuDevicePrimaryCtxReset_v2 as cuDevicePrimaryCtxReset,
        cuDevicePrimaryCtxSetFlags_v2 as cuDevicePrimaryCtxSetFlags,
        cuDeviceTotalMem_v2 as cuDeviceTotalMem, cuEventDestroy_v2 as cuEventDestroy,
        cuEventElapsedTime_v2 as cuEventElapsedTime, cuGetProcAddress_v2 as cuGetProcAddress,
        cuGraphAddDependencies_v2 as cuGraphAddDependencies,
        cuGraphAddKernelNode_v2 as cuGraphAddKernelNode, cuGraphAddNode_v2 as cuGraphAddNode,
        cuGraphExecKernelNodeSetParams_v2 as cuGraphExecKernelNodeSetParams,
        cuGraphExecUpdate_v2 as cuGraphExecUpdate, cuGraphGetEdges_v2 as cuGraphGetEdges,
        cuGraphKernelNodeGetParams_v2 as cuGraphKernelNodeGetParams,
        cuGraphKernelNodeSetParams_v2 as cuGraphKernelNodeSetParams,
        cuGraphNodeGetDependencies_v2 as cuGraphNodeGetDependencies,
        cuGraphNodeGetDependentNodes_v2 as cuGraphNodeGetDependentNodes,
        cuGraphRemoveDependencies_v2 as cuGraphRemoveDependencies,
        cuGraphicsResourceGetMappedPointer_v2 as cuGraphicsResourceGetMappedPointer,
        cuGraphicsResourceSetMapFlags_v2 as cuGraphicsResourceSetMapFlags,
        cuIpcOpenMemHandle_v2 as cuIpcOpenMemHandle, cuLinkAddData_v2 as cuLinkAddData,
        cuLinkAddFile_v2 as cuLinkAddFile, cuLinkCreate_v2 as cuLinkCreate,
        cuMemAdvise_v2 as cuMemAdvise, cuMemAlloc_v2 as cuMemAlloc,
        cuMemAllocHost_v2 as cuMemAllocHost, cuMemAllocPitch_v2 as cuMemAllocPitch,
        cuMemFree_v2 as cuMemFree, cuMemGetAddressRange_v2 as cuMemGetAddressRange,
        cuMemGetInfo_v2 as cuMemGetInfo, cuMemHostGetDevicePointer_v2 as cuMemHostGetDevicePointer,
        cuMemHostRegister_v2 as cuMemHostRegister, cuMemPrefetchAsync_v2 as cuMemPrefetchAsync,
        cuMemcpy2D_v2 as cuMemcpy2D, cuMemcpy2DAsync_v2 as cuMemcpy2DAsync,
        cuMemcpy2DUnaligned_v2 as cuMemcpy2DUnaligned, cuMemcpy3D_v2 as cuMemcpy3D,
        cuMemcpy3DAsync_v2 as cuMemcpy3DAsync, cuMemcpy3DBatchAsync_v2 as cuMemcpy3DBatchAsync,
        cuMemcpyAtoA_v2 as cuMemcpyAtoA, cuMemcpyAtoD_v2 as cuMemcpyAtoD,
        cuMemcpyAtoH_v2 as cuMemcpyAtoH, cuMemcpyAtoHAsync_v2 as cuMemcpyAtoHAsync,
        cuMemcpyBatchAsync_v2 as cuMemcpyBatchAsync, cuMemcpyDtoA_v2 as cuMemcpyDtoA,
        cuMemcpyDtoD_v2 as cuMemcpyDtoD, cuMemcpyDtoDAsync_v2 as cuMemcpyDtoDAsync,
        cuMemcpyDtoH_v2 as cuMemcpyDtoH, cuMemcpyDtoHAsync_v2 as cuMemcpyDtoHAsync,
        cuMemcpyHtoA_v2 as cuMemcpyHtoA, cuMemcpyHtoAAsync_v2 as cuMemcpyHtoAAsync,
        cuMemcpyHtoD_v2 as cuMemcpyHtoD, cuMemcpyHtoDAsync_v2 as cuMemcpyHtoDAsync,
        cuMemsetD2D8_v2 as cuMemsetD2D8, cuMemsetD2D16_v2 as cuMemsetD2D16,
        cuMemsetD2D32_v2 as cuMemsetD2D32, cuMemsetD8_v2 as cuMemsetD8,
        cuMemsetD16_v2 as cuMemsetD16, cuMemsetD32_v2 as cuMemsetD32,
        cuModuleGetGlobal_v2 as cuModuleGetGlobal, cuStreamBatchMemOp_v2 as cuStreamBatchMemOp,
        cuStreamBeginCapture_v2 as cuStreamBeginCapture, cuStreamDestroy_v2 as cuStreamDestroy,
        cuStreamGetCaptureInfo_v3 as cuStreamGetCaptureInfo,
        cuStreamUpdateCaptureDependencies_v2 as cuStreamUpdateCaptureDependencies,
        cuStreamWaitValue32_v2 as cuStreamWaitValue32,
        cuStreamWaitValue64_v2 as cuStreamWaitValue64,
        cuStreamWriteValue32_v2 as cuStreamWriteValue32,
        cuStreamWriteValue64_v2 as cuStreamWriteValue64,
        cuTexRefSetAddress2D_v3 as cuTexRefSetAddress2D,
    };
}

#[cfg(feature = "driver")]
pub mod profiler {
    #[cfg(feature = "driver_13_2")]
    include!("profiler_sys_13020.rs");
}

pub mod library_types {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    #[cfg(feature = "runtime_13_2")]
    include!("library_types_sys_13020.rs");
}

#[cfg(feature = "nvrtc")]
pub mod nvrtc {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    #[cfg(feature = "nvrtc_13_2")]
    include!("nvrtc_sys_13020.rs");
}

#[cfg(feature = "nvvm")]
pub mod nvvm {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    #[cfg(feature = "nvvm_13_2")]
    include!("nvvm_sys_13020.rs");
}

#[cfg(feature = "nvtx")]
pub mod nvtx {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    #[cfg(feature = "nvtx_3")]
    include!("nvtx_sys_3.rs");
}

#[cfg(feature = "runtime")]
pub mod runtime {
    use num_enum::{IntoPrimitive, TryFromPrimitive};

    // These types have type definitions in `driver_types.h`.
    #[cfg(feature = "driver_13_2")]
    #[rustfmt::skip]
    pub use crate::driver::{
        // --- Core handles & pointers ---
        cudaError_enum as cudaError_t,
        CUevent as cudaEvent_t,
        CUfunction as cudaFunction_t,
        CUstream as cudaStream_t,
        CUuuid as cudaUUID_t,
        CUuserObject as cudaUserObject_t,

        // --- Array & memory Types ---
        CUarray as cudaArray_t,
        CUmipmappedArray as cudaMipmappedArray_t,
        CUmemoryPool as cudaMemPool_t,

        // --- Texture & surface types ---
        CUgraphicsResource as cudaGraphicsResource_t,

        // --- Memory/semaphore types ---
        CUexternalMemory as cudaExternalMemory_t,
        CUexternalSemaphore as cudaExternalSemaphore_t,

        // --- Graph types ---
        CUgraph as cudaGraph_t,
        CUgraphNode as cudaGraphNode_t,
        CUgraphExec as cudaGraphExec_t,

        CUkernel as cudaKernel_t,
        CUlibrary as cudaLibrary_t,
        CUgraphConditionalHandle as cudaGraphConditionalHandle,
        CUasyncCallbackHandle as cudaAsyncCallbackHandle_t,
    };

    // These types are defined both in `driver_types.h` and `cuda.h` and might not be interchangeable.
    // Types of different layouts should not be imported here.
    #[cfg(feature = "driver_13_2")]
    #[rustfmt::skip]
    pub use crate::driver::{
        // --- Core Handles & Pointers ---
        CUflushGPUDirectRDMAWritesScope as cudaFlushGPUDirectRDMAWritesScope,
        CUflushGPUDirectRDMAWritesTarget as cudaFlushGPUDirectRDMAWritesTarget,

        // --- Array & memory Types ---
        CUDA_MEMSET_NODE_PARAMS as cudaMemsetParams,
        CUDA_HOST_NODE_PARAMS as cudaHostNodeParams,
        CUDA_EXT_SEM_SIGNAL_NODE_PARAMS as cudaExternalSemaphoreSignalNodeParams,
        CUDA_EXT_SEM_WAIT_NODE_PARAMS as cudaExternalSemaphoreWaitNodeParams,
        CUDA_MEM_ALLOC_NODE_PARAMS as cudaMemAllocNodeParams,
        CUDA_ARRAY_MEMORY_REQUIREMENTS as cudaArrayMemoryRequirements,
        CUDA_ARRAY_SPARSE_PROPERTIES as cudaArraySparseProperties,
        CUDA_MEMCPY3D_BATCH_OP as cudaMemcpy3DBatchOp,

        CUmemcpyAttributes as cudaMemcpyAttributes,
        CUmemPoolProps as cudaMemPoolProps,
        CUmemPoolPtrExportData as cudaMemPoolPtrExportData,
        CUmemAllocationHandleType as cudaMemAllocationHandleType,
        CUmemPool_attribute as cudaMemPoolAttr,
        CUsharedconfig as cudaSharedMemConfig,
        CUmemLocation as cudaMemLocation,
        CUmemAccessDesc as cudaMemAccessDesc,
        CUmem_range_attribute as cudaMemRangeAttribute,
        CUmem_advise as cudaMemoryAdvise,
        CUmemAccess_flags as cudaMemAccessFlags,

        // --- IPC Types ---
        CUipcEventHandle as cudaIpcEventHandle_t,
        CUipcMemHandle as cudaIpcMemHandle_t,

        // --- Graph Types ---
        CUgraphExecUpdateResult as cudaGraphExecUpdateResult,
        CUgraphMem_attribute as cudaGraphMemAttributeType,
        CUgraphNodeType as cudaGraphNodeType,
        CUgraphEdgeData as cudaGraphEdgeData,

        CUgraphNodeParams as cudaGraphNodeParams,
        CUgraphExecUpdateResultInfo as cudaGraphExecUpdateResultInfo,
        CUDA_GRAPH_INSTANTIATE_PARAMS as cudaGraphInstantiateParams,

        // --- Function Pointer Types ---
        CUhostFn as cudaHostFn_t,
        CUasyncCallback as cudaAsyncCallback,

        // --- Enums ---
        CUdevice_attribute as cudaDeviceAttr,
        CUfunction_attribute as cudaFuncAttribute,
        CUjit_option as cudaJitOption,
        CUdevice_P2PAttribute as cudaDeviceP2PAttr,
        CUlibraryOption as cudaLibraryOption,
        CUdriverProcAddressQueryResult as cudaDriverEntryPointQueryResult,

        // --- Other types ---
        CUfunc_cache as cudaFuncCache,
        CUstreamCaptureMode as cudaStreamCaptureMode,
        CUstreamCaptureStatus as cudaStreamCaptureStatus,
        // CUextent3D as cudaExtent,
    };

    pub type cudaArray_const_t = *const crate::driver::CUarray;
    pub type cudaMipmappedArray_const_t = *const crate::driver::CUmipmappedArray_st;

    #[cfg(feature = "runtime_13_2")]
    include!("driver_types_sys_13020.rs");
    #[cfg(feature = "runtime_13_2")]
    include!("runtime_sys_13020.rs");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        #[cfg(feature = "driver")]
        {
            let mut version = 0;
            unsafe {
                driver::cuDriverGetVersion(&mut version);
            }
            assert_ne!(version, 0);
            println!("CUDA driver version: {}", version);
        }

        #[cfg(feature = "runtime")]
        {
            let mut version = 0;
            unsafe {
                runtime::cudaRuntimeGetVersion(&mut version);
            }
            assert_ne!(version, 0);
            println!("CUDA runtime version: {}", version);
        }

        #[cfg(feature = "nvrtc")]
        {
            let mut major = 0;
            let mut minor = 0;
            unsafe {
                nvrtc::nvrtcVersion(&mut major, &mut minor);
            }
            assert_ne!(major, 0);
            println!("NVRTC version: {}.{}", major, minor);
        }

        #[cfg(feature = "nvtx")]
        {
            assert_eq!(nvtx::NVTX_VERSION, 3);
            println!("NVTX version: {}", nvtx::NVTX_VERSION);
        }
    }
}
