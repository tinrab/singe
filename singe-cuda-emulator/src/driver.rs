use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_ulonglong, c_void},
    ptr, slice,
};

use singe_cuda_sys::driver::{
    CUcontext, CUctxCreateParams, CUdevice, CUdevice_attribute, CUdeviceptr, CUfunction,
    CUjit_option, CUlaunchConfig, CUmemAccess_flags, CUmemAccessDesc,
    CUmemAllocationGranularity_flags, CUmemAllocationProp, CUmemGenericAllocationHandle,
    CUmemLocation, CUmemLocationType, CUmodule, CUresult, CUstream, size_t,
};
use singe_ptx::emulator::{Kernel, KernelArg, LaunchConfig, ParsedModule};

use crate::machine::{
    Physical, Status, current_context, index, is_default_stream, machine, provide_symbols,
    set_current_context, size, status, store, submit_status,
};

/// Granularity of physical memory and reserved ranges, as on current GPUs.
pub(crate) const GRANULARITY: u64 = 1 << 21;

/// The CUDA version the emulated driver and runtime report, 13.4.
pub(crate) const VERSION: c_int = 13040;

pub(crate) fn provide(name: &CStr) -> Option<*const ()> {
    provide_symbols!(
        name,
        cuInit => init,
        cuDriverGetVersion => driver_version,
        cuDeviceGet => device_get,
        cuDeviceGetCount => device_count,
        cuDeviceGetName => device_name,
        cuDeviceGetAttribute => device_attribute,
        cuDeviceTotalMem_v2 => device_total_memory,
        cuDevicePrimaryCtxRetain => primary_retain,
        cuDevicePrimaryCtxRelease_v2 => primary_release,
        cuCtxGetCurrent => context_current,
        cuCtxSetCurrent => context_set_current,
        cuCtxGetDevice => context_device,
        cuCtxCreate_v4 => context_create,
        cuCtxDestroy_v2 => context_destroy,
        cuCtxSynchronize => context_synchronize,
        cuCtxEnablePeerAccess => enable_peer_access,
        cuCtxDisablePeerAccess => disable_peer_access,
        cuModuleLoadData => module_load,
        cuModuleLoadDataEx => module_load_with_options,
        cuModuleGetFunction => module_function,
        cuModuleUnload => module_unload,
        cuFuncGetParamCount => function_parameter_count,
        cuFuncGetParamInfo => function_parameter_info,
        cuLaunchKernelEx => launch_with_config,
        cuStreamSynchronize => stream_synchronize,
        cuMemAlloc_v2 => memory_allocate,
        cuMemFree_v2 => memory_free,
        cuMemcpy => memory_copy,
        cuMemcpyAsync => memory_copy_async,
        cuMemcpyHtoD_v2 => copy_to_device,
        cuMemcpyDtoH_v2 => copy_to_host,
        cuMemGetAllocationGranularity => allocation_granularity,
        cuMemCreate => physical_create,
        cuMemRelease => physical_release,
        cuMemAddressReserve => address_reserve,
        cuMemAddressFree => address_free,
        cuMemMap => map,
        cuMemUnmap => unmap,
        cuMemSetAccess => set_access,
        cuMemGetAccess => get_access,
        cuGetErrorString => error_string,
        cuGetErrorName => error_name,
    )
}

unsafe extern "C" fn init(_flags: c_uint) -> CUresult {
    status(|| machine().map(|_| ()))
}

unsafe extern "C" fn driver_version(version: *mut c_int) -> CUresult {
    // SAFETY: `version` is the caller's output slot.
    status(|| unsafe { store(version, VERSION) })
}

unsafe extern "C" fn device_get(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    status(|| {
        machine()?.device(index(ordinal)?)?;
        // SAFETY: `device` is the caller's output slot.
        unsafe { store(device, ordinal) }
    })
}

unsafe extern "C" fn device_count(count: *mut c_int) -> CUresult {
    status(|| {
        let value =
            c_int::try_from(machine()?.devices.len()).map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
        // SAFETY: `count` is the caller's output slot.
        unsafe { store(count, value) }
    })
}

unsafe extern "C" fn device_name(name: *mut c_char, len: c_int, device: CUdevice) -> CUresult {
    status(|| {
        machine()?.device(index(device)?)?;
        let text = format!("Singe emulated device {device}");
        let capacity = usize::try_from(len).map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
        if name.is_null() || capacity == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        let written = text.len().min(capacity - 1);
        // SAFETY: `name` is the caller's buffer of `len` bytes, and `written + 1` fits in it.
        unsafe {
            ptr::copy_nonoverlapping(text.as_ptr().cast(), name, written);
            name.add(written).write(0);
        }
        Ok(())
    })
}

unsafe extern "C" fn device_attribute(
    value: *mut c_int,
    attribute: CUdevice_attribute,
    device: CUdevice,
) -> CUresult {
    status(|| {
        machine()?.device(index(device)?)?;
        let result = match attribute {
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR => 1024,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z => 64,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X => c_int::MAX,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z => 65_535,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN => 49_152,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK => 65_536,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE => 32,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            | CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED => 1,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR => 9,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID => device,
            _ => 0,
        };
        // SAFETY: `value` is the caller's output slot.
        unsafe { store(value, result) }
    })
}

unsafe extern "C" fn device_total_memory(bytes: *mut size_t, device: CUdevice) -> CUresult {
    status(|| {
        let machine = machine()?;
        machine.device(index(device)?)?;
        // SAFETY: `bytes` is the caller's output slot.
        unsafe { store(bytes, machine.device_bytes as size_t) }
    })
}

unsafe extern "C" fn primary_retain(context: *mut CUcontext, device: CUdevice) -> CUresult {
    status(|| {
        let machine = machine()?;
        let device = index(device)?;
        let handle = machine.primary_context(device)?;
        machine.state().primary_references[device] += 1;
        // SAFETY: `context` is the caller's output slot.
        unsafe { store(context, ptr::without_provenance_mut(handle)) }
    })
}

unsafe extern "C" fn primary_release(device: CUdevice) -> CUresult {
    status(|| {
        let machine = machine()?;
        let device = index(device)?;
        machine.device(device)?;
        let mut state = machine.state();
        let references = &mut state.primary_references[device];
        *references = references
            .checked_sub(1)
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
        drop(state);
        Ok(())
    })
}

unsafe extern "C" fn context_current(context: *mut CUcontext) -> CUresult {
    // SAFETY: `context` is the caller's output slot.
    status(|| unsafe { store(context, ptr::without_provenance_mut(current_context())) })
}

unsafe extern "C" fn context_set_current(context: CUcontext) -> CUresult {
    status(|| {
        if !context.is_null() {
            machine()?.context_device(context.addr())?;
        }

        set_current_context(context.addr());
        Ok(())
    })
}

unsafe extern "C" fn context_device(device: *mut CUdevice) -> CUresult {
    status(|| {
        let current = machine()?.current_device()?;
        let current = CUdevice::try_from(current).map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
        // SAFETY: `device` is the caller's output slot.
        unsafe { store(device, current) }
    })
}

unsafe extern "C" fn context_create(
    context: *mut CUcontext,
    _parameters: *mut CUctxCreateParams,
    _flags: c_uint,
    device: CUdevice,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        let device = index(device)?;
        machine.device(device)?;
        let handle = {
            let mut state = machine.state();
            let handle = state.handle();
            state.contexts.insert(handle, device);
            handle
        };
        set_current_context(handle);
        // SAFETY: `context` is the caller's output slot.
        unsafe { store(context, ptr::without_provenance_mut(handle)) }
    })
}

unsafe extern "C" fn context_destroy(context: CUcontext) -> CUresult {
    status(|| {
        machine()?
            .state()
            .contexts
            .remove(&context.addr())
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)?;
        if current_context() == context.addr() {
            set_current_context(0);
        }
        Ok(())
    })
}

unsafe extern "C" fn context_synchronize() -> CUresult {
    status(|| {
        let machine = machine()?;
        machine.synchronize_device(machine.current_device()?)
    })
}

pub(crate) fn peer_access(accessor: usize, owner: usize, enable: bool) -> Status {
    let machine = machine()?;
    let (accessor_device, owner_device) = (machine.device(accessor)?, machine.device(owner)?);
    if accessor == owner {
        return Err(CUresult::CUDA_ERROR_INVALID_DEVICE);
    }

    if !machine.topology.are_peers(accessor, owner) {
        return Err(CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
    }

    let enabled = machine.state().peers.contains(&(accessor, owner));
    match (enable, enabled) {
        (true, true) => Err(CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED),
        (false, false) => Err(CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED),
        (true, false) => {
            machine.wait(accessor_device.enable_peer_access(owner_device))?;
            machine.state().peers.insert((accessor, owner));
            Ok(())
        }
        (false, true) => {
            machine.wait(accessor_device.disable_peer_access(owner_device))?;
            machine.state().peers.remove(&(accessor, owner));
            Ok(())
        }
    }
}

unsafe extern "C" fn enable_peer_access(peer: CUcontext, _flags: c_uint) -> CUresult {
    status(|| {
        let machine = machine()?;
        peer_access(
            machine.current_device()?,
            machine.context_device(peer.addr())?,
            true,
        )
    })
}

unsafe extern "C" fn disable_peer_access(peer: CUcontext) -> CUresult {
    status(|| {
        let machine = machine()?;
        peer_access(
            machine.current_device()?,
            machine.context_device(peer.addr())?,
            false,
        )
    })
}

unsafe extern "C" fn module_load(module: *mut CUmodule, image: *const c_void) -> CUresult {
    status(|| {
        let machine = machine()?;
        if image.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        // SAFETY: CUDA takes PTX images as nul-terminated text, which singe-cuda appends.
        let text = unsafe { CStr::from_ptr(image.cast()) }
            .to_str()
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_IMAGE)?;
        let parsed = ParsedModule::parse(text).map_err(|_| CUresult::CUDA_ERROR_INVALID_PTX)?;
        let device = machine.current_device()?;
        let loaded = machine
            .device(device)?
            .load(parsed)
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_PTX)?
            .wait()
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_IMAGE)?;
        let handle = {
            let mut state = machine.state();
            let handle = state.handle();
            state.modules.insert(handle, loaded);
            handle
        };
        // SAFETY: `module` is the caller's output slot.
        unsafe { store(module, ptr::without_provenance_mut(handle)) }
    })
}

unsafe extern "C" fn module_load_with_options(
    module: *mut CUmodule,
    image: *const c_void,
    _count: c_uint,
    _options: *mut CUjit_option,
    _values: *mut *mut c_void,
) -> CUresult {
    // SAFETY: the caller's arguments meet `cuModuleLoadData`'s requirements, and JIT options have no effect on the emulator.
    unsafe { module_load(module, image) }
}

unsafe extern "C" fn module_function(
    function: *mut CUfunction,
    module: CUmodule,
    name: *const c_char,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        if name.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        // SAFETY: the caller passes a nul-terminated kernel name.
        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| CUresult::CUDA_ERROR_NOT_FOUND)?;
        let loaded = machine
            .state()
            .modules
            .get(&module.addr())
            .cloned()
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
        let kernel = loaded
            .kernel(name)
            .map_err(|_| CUresult::CUDA_ERROR_NOT_FOUND)?;
        let handle = {
            let mut state = machine.state();
            let handle = state.handle();
            state.functions.insert(handle, kernel);
            handle
        };
        // SAFETY: `function` is the caller's output slot.
        unsafe { store(function, ptr::without_provenance_mut(handle)) }
    })
}

unsafe extern "C" fn module_unload(module: CUmodule) -> CUresult {
    status(|| {
        machine()?
            .state()
            .modules
            .remove(&module.addr())
            .map(|_| ())
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)
    })
}

fn kernel(function: CUfunction) -> Result<Kernel, CUresult> {
    machine()?
        .state()
        .functions
        .get(&function.addr())
        .cloned()
        .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)
}

unsafe extern "C" fn function_parameter_count(
    function: CUfunction,
    count: *mut size_t,
) -> CUresult {
    status(|| {
        let kernel = kernel(function)?;
        // SAFETY: `count` is the caller's output slot.
        unsafe { store(count, kernel.parameters().len() as size_t) }
    })
}

unsafe extern "C" fn function_parameter_info(
    function: CUfunction,
    parameter: size_t,
    offset: *mut size_t,
    bytes: *mut size_t,
) -> CUresult {
    status(|| {
        let kernel = kernel(function)?;
        let layout = kernel
            .parameters()
            .get(size(parameter)?)
            .ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        // SAFETY: `offset` and `bytes` are the caller's output slots.
        unsafe {
            store(offset, layout.byte_offset())?;
            store(bytes, layout.storage().byte_len())
        }
    })
}

unsafe extern "C" fn launch_with_config(
    config: *const CUlaunchConfig,
    function: CUfunction,
    arguments: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    status(|| {
        // SAFETY: the caller passes a valid launch configuration.
        let config = unsafe { config.as_ref() }.ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        if !extra.is_null() {
            return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
        }

        let kernel = kernel(function)?;
        let stream = machine()?.stream(config.hStream, kernel.module().device())?;
        let parameters = kernel.parameters();
        if !parameters.is_empty() && arguments.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        let mut values = Vec::with_capacity(parameters.len());
        for (position, parameter) in parameters.iter().enumerate() {
            // SAFETY: the caller passes one argument pointer per kernel parameter.
            let argument = unsafe { *arguments.add(position) };
            if argument.is_null() {
                return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
            }

            let len = size(parameter.storage().byte_len())?;
            // SAFETY: the caller passes each argument pointer valid for reads of its parameter's size.
            let bytes = unsafe { slice::from_raw_parts(argument.cast::<u8>(), len) };
            values.push(KernelArg::raw(bytes.to_vec()));
        }

        let launch = LaunchConfig::create(
            [config.gridDimX, config.gridDimY, config.gridDimZ],
            [config.blockDimX, config.blockDimY, config.blockDimZ],
            config.sharedMemBytes as usize,
        )
        .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
        drop(
            kernel
                .launch_on(&stream, launch, values)
                .map_err(|error| submit_status(&error))?,
        );
        Ok(())
    })
}

pub(crate) fn synchronize_stream(handle: CUstream) -> Status {
    let machine = machine()?;
    if is_default_stream(handle) {
        return machine.synchronize_legacy(machine.runtime_device()?);
    }

    machine.wait(machine.stream(handle, 0)?.synchronize())
}

unsafe extern "C" fn stream_synchronize(stream: CUstream) -> CUresult {
    status(|| synchronize_stream(stream))
}

unsafe extern "C" fn memory_allocate(address: *mut CUdeviceptr, bytes: size_t) -> CUresult {
    status(|| {
        let machine = machine()?;
        let allocated = machine.allocate(machine.current_device()?, bytes)?;
        // SAFETY: `address` is the caller's output slot.
        unsafe { store(address, allocated) }
    })
}

unsafe extern "C" fn memory_free(address: CUdeviceptr) -> CUresult {
    status(|| machine()?.free(address))
}

/// Copies `count` bytes between unified addresses in `stream`'s order on the current device.
///
/// # Safety
///
/// Host ranges must be valid for `count` bytes until the copy completes.
unsafe fn copy(destination: u64, source: u64, count: u64, stream: CUstream) -> Status {
    let machine = machine()?;
    let device = machine.current_device()?;
    let (to, from) = (machine.side(destination), machine.side(source));
    // SAFETY: the caller guarantees the host ranges.
    unsafe { machine.copy((destination, to), (source, from), count, stream, device) }
}

/// Copies `count` bytes between unified addresses on the current device's legacy stream, then waits.
///
/// # Safety
///
/// Host ranges must be valid for `count` bytes.
unsafe fn copy_now(destination: u64, source: u64, count: u64) -> Status {
    // SAFETY: the caller guarantees the host ranges, and the copy completes before this returns.
    unsafe { copy(destination, source, count, ptr::null_mut())? };
    let machine = machine()?;
    machine.synchronize_legacy(machine.current_device()?)
}

unsafe extern "C" fn memory_copy(
    destination: CUdeviceptr,
    source: CUdeviceptr,
    count: size_t,
) -> CUresult {
    // SAFETY: the caller passes ranges valid for `count` bytes.
    status(|| unsafe { copy_now(destination, source, count) })
}

unsafe extern "C" fn memory_copy_async(
    destination: CUdeviceptr,
    source: CUdeviceptr,
    count: size_t,
    stream: CUstream,
) -> CUresult {
    // SAFETY: the caller passes ranges valid for `count` bytes until the copy completes.
    status(|| unsafe { copy(destination, source, count, stream) })
}

unsafe extern "C" fn copy_to_device(
    destination: CUdeviceptr,
    source: *const c_void,
    count: size_t,
) -> CUresult {
    // SAFETY: the caller passes a host source valid for `count` bytes.
    status(|| unsafe { copy_now(destination, source.addr() as u64, count) })
}

unsafe extern "C" fn copy_to_host(
    destination: *mut c_void,
    source: CUdeviceptr,
    count: size_t,
) -> CUresult {
    // SAFETY: the caller passes a host destination valid for `count` bytes.
    status(|| unsafe { copy_now(destination.addr() as u64, source, count) })
}

unsafe extern "C" fn allocation_granularity(
    granularity: *mut size_t,
    _properties: *const CUmemAllocationProp,
    _option: CUmemAllocationGranularity_flags,
) -> CUresult {
    // SAFETY: `granularity` is the caller's output slot.
    status(|| unsafe { store(granularity, GRANULARITY as size_t) })
}

fn location_device(location: CUmemLocation) -> Result<usize, CUresult> {
    if location.type_ != CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE {
        return Err(CUresult::CUDA_ERROR_NOT_SUPPORTED);
    }
    // SAFETY: a device location stores its ordinal in `id`.
    index(unsafe { location.__bindgen_anon_1.id })
}

unsafe extern "C" fn physical_create(
    handle: *mut CUmemGenericAllocationHandle,
    bytes: size_t,
    properties: *const CUmemAllocationProp,
    _flags: c_ulonglong,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        // SAFETY: the caller passes valid allocation properties.
        let properties =
            unsafe { properties.as_ref() }.ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let device = location_device(properties.location)?;
        machine.device(device)?;
        if bytes == 0 || !bytes.is_multiple_of(GRANULARITY) {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        let len = size(bytes)?;
        let physical = machine
            .host(move |memory| memory.allocate_physical(device, len))
            .map_err(|_| CUresult::CUDA_ERROR_OUT_OF_MEMORY)?;
        machine.state().physical.insert(
            physical,
            Physical {
                released: false,
                mappings: 0,
            },
        );
        // SAFETY: `handle` is the caller's output slot.
        unsafe { store(handle, physical) }
    })
}

/// Frees physical memory that is released and no longer mapped.
fn collect_physical(handle: u64) -> Status {
    let machine = machine()?;
    let unused = machine
        .state()
        .physical
        .get(&handle)
        .is_some_and(|physical| physical.released && physical.mappings == 0);
    if unused {
        machine.host(move |memory| memory.free(handle))?;
        machine.state().physical.remove(&handle);
    }
    Ok(())
}

unsafe extern "C" fn physical_release(handle: CUmemGenericAllocationHandle) -> CUresult {
    status(|| {
        machine()?
            .state()
            .physical
            .get_mut(&handle)
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?
            .released = true;
        collect_physical(handle)
    })
}

unsafe extern "C" fn address_reserve(
    address: *mut CUdeviceptr,
    bytes: size_t,
    alignment: size_t,
    _hint: CUdeviceptr,
    _flags: c_ulonglong,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        let alignment = alignment.max(GRANULARITY);
        if bytes == 0 || !alignment.is_power_of_two() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        let reserved = machine
            .host(move |memory| memory.reserve(bytes, alignment))
            .map_err(|_| CUresult::CUDA_ERROR_OUT_OF_MEMORY)?;
        machine.state().reservations.insert(reserved, bytes);
        // SAFETY: `address` is the caller's output slot.
        unsafe { store(address, reserved) }
    })
}

unsafe extern "C" fn address_free(address: CUdeviceptr, _bytes: size_t) -> CUresult {
    status(|| {
        let machine = machine()?;
        if !machine.state().reservations.contains_key(&address) {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        machine.host(move |memory| memory.free_reservation(address))?;
        machine.state().reservations.remove(&address);
        Ok(())
    })
}

unsafe extern "C" fn map(
    address: CUdeviceptr,
    bytes: size_t,
    offset: size_t,
    handle: CUmemGenericAllocationHandle,
    _flags: c_ulonglong,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        if !machine.state().physical.contains_key(&handle) {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        }

        machine
            .host(move |memory| memory.map(address, bytes, handle, offset))
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let mut state = machine.state();
        state.mappings.insert(address, (bytes, handle));
        if let Some(physical) = state.physical.get_mut(&handle) {
            physical.mappings += 1;
        }

        drop(state);
        Ok(())
    })
}

unsafe extern "C" fn unmap(address: CUdeviceptr, bytes: size_t) -> CUresult {
    status(|| {
        let machine = machine()?;
        machine
            .host(move |memory| memory.unmap(address, bytes))
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let handles: Vec<u64> = {
            let mut state = machine.state();
            let starts: Vec<u64> = state
                .mappings
                .range(address..address.saturating_add(bytes))
                .map(|(start, _)| *start)
                .collect();
            let mut handles = Vec::new();
            for start in starts {
                if let Some((_, handle)) = state.mappings.remove(&start) {
                    if let Some(physical) = state.physical.get_mut(&handle) {
                        physical.mappings -= 1;
                    }

                    handles.push(handle);
                }
            }

            drop(state);
            handles
        };
        handles.into_iter().try_for_each(collect_physical)
    })
}

unsafe extern "C" fn set_access(
    address: CUdeviceptr,
    bytes: size_t,
    descriptors: *const CUmemAccessDesc,
    count: size_t,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        if descriptors.is_null() {
            return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
        }

        // SAFETY: the caller passes `count` access descriptors.
        let descriptors = unsafe { slice::from_raw_parts(descriptors, size(count)?) };
        let mut changes = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            let device = location_device(descriptor.location)?;
            machine.device(device)?;
            changes.push((
                device,
                descriptor.flags != CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE,
            ));
        }

        machine
            .host(move |memory| {
                changes.into_iter().try_for_each(|(device, allowed)| {
                    memory.set_access(address, bytes, device, allowed)
                })
            })
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)
    })
}

unsafe extern "C" fn get_access(
    flags: *mut c_ulonglong,
    location: *const CUmemLocation,
    address: CUdeviceptr,
) -> CUresult {
    status(|| {
        let machine = machine()?;
        // SAFETY: the caller passes a valid location.
        let location = unsafe { location.as_ref() }.ok_or(CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let device = location_device(*location)?;
        let allowed = machine
            .host(move |memory| memory.has_access(address, device))
            .map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)?;
        let value = if allowed {
            CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE.0
        } else {
            CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_NONE.0
        };
        // SAFETY: `flags` is the caller's output slot.
        unsafe { store(flags, c_ulonglong::from(value)) }
    })
}

/// Returns the name and description of a CUDA status.
pub(crate) fn describe(error: CUresult) -> (&'static CStr, &'static CStr) {
    match error {
        CUresult::CUDA_SUCCESS => (c"CUDA_SUCCESS", c"no error"),
        CUresult::CUDA_ERROR_INVALID_VALUE => (c"CUDA_ERROR_INVALID_VALUE", c"invalid argument"),
        CUresult::CUDA_ERROR_OUT_OF_MEMORY => (c"CUDA_ERROR_OUT_OF_MEMORY", c"out of memory"),
        CUresult::CUDA_ERROR_NOT_INITIALIZED => (
            c"CUDA_ERROR_NOT_INITIALIZED",
            c"the emulator is not installed",
        ),
        CUresult::CUDA_ERROR_INVALID_DEVICE => {
            (c"CUDA_ERROR_INVALID_DEVICE", c"invalid device ordinal")
        }
        CUresult::CUDA_ERROR_INVALID_IMAGE => (
            c"CUDA_ERROR_INVALID_IMAGE",
            c"device kernel image is invalid",
        ),
        CUresult::CUDA_ERROR_INVALID_CONTEXT => {
            (c"CUDA_ERROR_INVALID_CONTEXT", c"invalid device context")
        }
        CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => (
            c"CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
            c"peer access is not supported between these two devices",
        ),
        CUresult::CUDA_ERROR_INVALID_PTX => {
            (c"CUDA_ERROR_INVALID_PTX", c"a PTX JIT compilation failed")
        }
        CUresult::CUDA_ERROR_INVALID_HANDLE => {
            (c"CUDA_ERROR_INVALID_HANDLE", c"invalid resource handle")
        }
        CUresult::CUDA_ERROR_NOT_FOUND => (c"CUDA_ERROR_NOT_FOUND", c"named symbol not found"),
        CUresult::CUDA_ERROR_NOT_READY => (c"CUDA_ERROR_NOT_READY", c"device not ready"),
        CUresult::CUDA_ERROR_ILLEGAL_ADDRESS => (
            c"CUDA_ERROR_ILLEGAL_ADDRESS",
            c"an illegal memory access was encountered",
        ),
        CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => (
            c"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
            c"too many resources requested for launch",
        ),
        CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => (
            c"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
            c"peer access is already enabled",
        ),
        CUresult::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => (
            c"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
            c"peer access has not been enabled",
        ),
        CUresult::CUDA_ERROR_ASSERT => (c"CUDA_ERROR_ASSERT", c"device-side assert triggered"),
        CUresult::CUDA_ERROR_LAUNCH_FAILED => {
            (c"CUDA_ERROR_LAUNCH_FAILED", c"unspecified launch failure")
        }
        CUresult::CUDA_ERROR_NOT_SUPPORTED => {
            (c"CUDA_ERROR_NOT_SUPPORTED", c"operation not supported")
        }
        _ => (c"CUDA_ERROR_UNKNOWN", c"unknown error"),
    }
}

unsafe extern "C" fn error_string(error: CUresult, text: *mut *const c_char) -> CUresult {
    // SAFETY: `text` is the caller's output slot.
    status(|| unsafe { store(text, describe(error).1.as_ptr()) })
}

unsafe extern "C" fn error_name(error: CUresult, text: *mut *const c_char) -> CUresult {
    // SAFETY: `text` is the caller's output slot.
    status(|| unsafe { store(text, describe(error).0.as_ptr()) })
}
