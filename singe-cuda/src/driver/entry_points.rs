use std::{ffi, ffi::CStr, mem, ptr};

use singe_core::CudaVersion;
use singe_cuda_sys::driver::{
    CUcheckpointOperationHandle, CUcliqueInfo, CUdevice, CUdeviceptr,
    CUdriverProcAddressQueryResult, CUgraph, CUgraphEdgeData, CUgraphNode, CUgraphNodeParams,
    CUlogicalEndpointId, CUlogicalEndpointIpcHandleType, CUlogicalEndpointProp,
    CUmemGenericAllocationHandle, CUmemLocation, CUresult, CUuuid, cuGetProcAddress, cuuint32_t,
    cuuint64_t, size_t,
};

use crate::{
    driver::version as driver_version,
    error::{Error, Result, Status},
    try_ffi,
};

// Entry points introduced in CUDA 13.3 and 13.4 may be absent in the installed driver.
// Resolve their exact ABI; no older entry point is substituted.

pub(crate) fn ensure_supported() -> Result<()> {
    ensure_version(13040)
}

fn ensure_version(version: i32) -> Result<()> {
    let required =
        CudaVersion::from_raw(version).ok_or_else(|| Error::out_of_range("cuda driver version"))?;
    if driver_version()? < required {
        return Err(Status::CallRequiresNewerDriver.into());
    }
    Ok(())
}

fn address(name: &CStr, version: i32) -> Result<*mut ffi::c_void> {
    ensure_version(version)?;
    let mut pointer = ptr::null_mut();
    let mut status = CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS;
    // SAFETY: `name` is NUL-terminated, and the outputs are writable locals.
    unsafe {
        try_ffi!(cuGetProcAddress(
            name.as_ptr(),
            &raw mut pointer,
            version,
            0,
            &raw mut status
        ))?;
    }

    if status != CUdriverProcAddressQueryResult::CU_GET_PROC_ADDRESS_SUCCESS || pointer.is_null() {
        return Err(Status::NotSupported.into());
    }
    Ok(pointer)
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuDeviceGetFabricClusterUuid(uuid: *mut CUuuid, dev: CUdevice) -> Result<()> {
    let pointer = address(c"cuDeviceGetFabricClusterUuid", 13040)?;
    // SAFETY: `pointer` is the driver's `cuDeviceGetFabricClusterUuid` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<*mut ffi::c_void, unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult>(
            pointer,
        )
    };
    // SAFETY: the caller passes `uuid` valid for `cuDeviceGetFabricClusterUuid` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(uuid, dev)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuDeviceGetCliqueCount(count: *mut size_t, dev: CUdevice) -> Result<()> {
    let pointer = address(c"cuDeviceGetCliqueCount", 13040)?;
    // SAFETY: `pointer` is the driver's `cuDeviceGetCliqueCount` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<*mut ffi::c_void, unsafe extern "C" fn(*mut size_t, CUdevice) -> CUresult>(
            pointer,
        )
    };
    // SAFETY: the caller passes `count` valid for `cuDeviceGetCliqueCount` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(count, dev)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuDeviceGetCliqueInfo(
    cliqueInfo: *mut CUcliqueInfo,
    count: *mut size_t,
    dev: CUdevice,
) -> Result<()> {
    let pointer = address(c"cuDeviceGetCliqueInfo", 13040)?;
    // SAFETY: `pointer` is the driver's `cuDeviceGetCliqueInfo` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(*mut CUcliqueInfo, *mut size_t, CUdevice) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `cliqueInfo`, `count` valid for `cuDeviceGetCliqueInfo` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(cliqueInfo, count, dev)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuMemGetLocationInfo(
    ptr: CUdeviceptr,
    size: size_t,
    summaryGranularity: size_t,
    samplingGranularity: size_t,
    location_out: *mut CUmemLocation,
) -> Result<()> {
    let pointer = address(c"cuMemGetLocationInfo", 13040)?;
    // SAFETY: `pointer` is the driver's `cuMemGetLocationInfo` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                CUdeviceptr,
                size_t,
                size_t,
                size_t,
                *mut CUmemLocation,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `location_out` valid for `cuMemGetLocationInfo` to read or write, as this function's call sites state.
    unsafe {
        try_ffi!(function(
            ptr,
            size,
            summaryGranularity,
            samplingGranularity,
            location_out
        ))
    }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuGraphAddNode_v3(
    phGraphNode: *mut CUgraphNode,
    hGraph: CUgraph,
    dependencies: *const CUgraphNode,
    dependencyData: *const CUgraphEdgeData,
    numDependencies: size_t,
    nodeParams: *mut CUgraphNodeParams,
) -> Result<()> {
    let pointer = address(c"cuGraphAddNode", 13040)?;
    // SAFETY: `pointer` is the driver's `cuGraphAddNode` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                *mut CUgraphNode,
                CUgraph,
                *const CUgraphNode,
                *const CUgraphEdgeData,
                size_t,
                *mut CUgraphNodeParams,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `phGraphNode`, `dependencies`, `dependencyData`, `nodeParams` valid for `cuGraphAddNode` to read or write, as this function's call sites state.
    unsafe {
        try_ffi!(function(
            phGraphNode,
            hGraph,
            dependencies,
            dependencyData,
            numDependencies,
            nodeParams
        ))
    }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuGraphNodeSetParams_v2(
    hNode: CUgraphNode,
    nodeParams: *mut CUgraphNodeParams,
) -> Result<()> {
    let pointer = address(c"cuGraphNodeSetParams", 13040)?;
    // SAFETY: `pointer` is the driver's `cuGraphNodeSetParams` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUgraphNode, *mut CUgraphNodeParams) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `nodeParams` valid for `cuGraphNodeSetParams` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(hNode, nodeParams)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuCheckpointOperationComplete(
    handle: CUcheckpointOperationHandle,
) -> Result<()> {
    let pointer = address(c"cuCheckpointOperationComplete", 13040)?;
    // SAFETY: `pointer` is the driver's `cuCheckpointOperationComplete` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUcheckpointOperationHandle) -> CUresult,
        >(pointer)
    };
    // SAFETY: `cuCheckpointOperationComplete` receives only values, and the caller passes identifiers and handles that are live for the call.
    unsafe { try_ffi!(function(handle)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointIdReserve(
    baseLeId: *mut CUlogicalEndpointId,
    count: cuuint32_t,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointIdReserve", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointIdReserve` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(*mut CUlogicalEndpointId, cuuint32_t) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `baseLeId` valid for `cuLogicalEndpointIdReserve` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(baseLeId, count)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointIdRelease(
    baseLeId: CUlogicalEndpointId,
    count: cuuint32_t,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointIdRelease", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointIdRelease` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUlogicalEndpointId, cuuint32_t) -> CUresult,
        >(pointer)
    };
    // SAFETY: `cuLogicalEndpointIdRelease` receives only values, and the caller passes identifiers and handles that are live for the call.
    unsafe { try_ffi!(function(baseLeId, count)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointCreate(
    leId: CUlogicalEndpointId,
    prop: *const CUlogicalEndpointProp,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointCreate", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointCreate` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUlogicalEndpointId, *const CUlogicalEndpointProp) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `prop` valid for `cuLogicalEndpointCreate` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(leId, prop)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointAddDevice(
    leId: CUlogicalEndpointId,
    dev: CUdevice,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointAddDevice", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointAddDevice` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUlogicalEndpointId, CUdevice) -> CUresult,
        >(pointer)
    };
    // SAFETY: `cuLogicalEndpointAddDevice` receives only values, and the caller passes identifiers and handles that are live for the call.
    unsafe { try_ffi!(function(leId, dev)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointDestroy(leId: CUlogicalEndpointId) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointDestroy", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointDestroy` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<*mut ffi::c_void, unsafe extern "C" fn(CUlogicalEndpointId) -> CUresult>(
            pointer,
        )
    };
    // SAFETY: `cuLogicalEndpointDestroy` receives only values, and the caller passes identifiers and handles that are live for the call.
    unsafe { try_ffi!(function(leId)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointBindAddr(
    leId: CUlogicalEndpointId,
    dev: CUdevice,
    offset: cuuint64_t,
    ptr: *mut ffi::c_void,
    size: cuuint64_t,
    flags: ffi::c_ulonglong,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointBindAddr", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointBindAddr` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                CUlogicalEndpointId,
                CUdevice,
                cuuint64_t,
                *mut ffi::c_void,
                cuuint64_t,
                ffi::c_ulonglong,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `ptr` valid for `cuLogicalEndpointBindAddr` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(leId, dev, offset, ptr, size, flags)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointBindMem(
    leId: CUlogicalEndpointId,
    dev: CUdevice,
    offset: cuuint64_t,
    memHandle: CUmemGenericAllocationHandle,
    memOffset: cuuint64_t,
    size: cuuint64_t,
    flags: ffi::c_ulonglong,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointBindMem", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointBindMem` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                CUlogicalEndpointId,
                CUdevice,
                cuuint64_t,
                CUmemGenericAllocationHandle,
                cuuint64_t,
                cuuint64_t,
                ffi::c_ulonglong,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: `cuLogicalEndpointBindMem` receives only values, and the caller passes identifiers and handles that are live for the call.
    unsafe {
        try_ffi!(function(
            leId, dev, offset, memHandle, memOffset, size, flags
        ))
    }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointUnbind(
    leId: CUlogicalEndpointId,
    dev: CUdevice,
    offset: cuuint64_t,
    size: cuuint64_t,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointUnbind", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointUnbind` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUlogicalEndpointId, CUdevice, cuuint64_t, cuuint64_t) -> CUresult,
        >(pointer)
    };
    // SAFETY: `cuLogicalEndpointUnbind` receives only values, and the caller passes identifiers and handles that are live for the call.
    unsafe { try_ffi!(function(leId, dev, offset, size)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointExport(
    handle: *mut ffi::c_void,
    leId: CUlogicalEndpointId,
    handleType: CUlogicalEndpointIpcHandleType,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointExport", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointExport` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                *mut ffi::c_void,
                CUlogicalEndpointId,
                CUlogicalEndpointIpcHandleType,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `handle` valid for `cuLogicalEndpointExport` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(handle, leId, handleType)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointImport(
    leId: CUlogicalEndpointId,
    handle: *const ffi::c_void,
    handleType: CUlogicalEndpointIpcHandleType,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointImport", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointImport` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                CUlogicalEndpointId,
                *const ffi::c_void,
                CUlogicalEndpointIpcHandleType,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `handle` valid for `cuLogicalEndpointImport` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(leId, handle, handleType)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointGetLimits(
    bindAlignment: *mut cuuint64_t,
    maxSize: *mut cuuint64_t,
    prop: *const CUlogicalEndpointProp,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointGetLimits", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointGetLimits` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(
                *mut cuuint64_t,
                *mut cuuint64_t,
                *const CUlogicalEndpointProp,
            ) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `bindAlignment`, `maxSize`, `prop` valid for `cuLogicalEndpointGetLimits` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(bindAlignment, maxSize, prop)) }
}

#[expect(non_snake_case, reason = "names mirror the upstream C API")]
pub(crate) unsafe fn cuLogicalEndpointQuery(
    leId: CUlogicalEndpointId,
    count: cuuint32_t,
    queryStatus: *mut ffi::c_int,
) -> Result<()> {
    let pointer = address(c"cuLogicalEndpointQuery", 13030)?;
    // SAFETY: `pointer` is the driver's `cuLogicalEndpointQuery` entry point resolved for this exact ABI version, so it has this function type.
    let function = unsafe {
        mem::transmute::<
            *mut ffi::c_void,
            unsafe extern "C" fn(CUlogicalEndpointId, cuuint32_t, *mut ffi::c_int) -> CUresult,
        >(pointer)
    };
    // SAFETY: the caller passes `queryStatus` valid for `cuLogicalEndpointQuery` to read or write, as this function's call sites state.
    unsafe { try_ffi!(function(leId, count, queryStatus)) }
}
