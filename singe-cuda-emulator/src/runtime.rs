use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    ptr,
    time::Instant,
};

use singe_cuda_sys::{
    driver::{CUdevice_P2PAttribute, CUresult, CUstreamCaptureStatus, size_t},
    runtime::{cudaError_t, cudaEvent_t, cudaMemcpyKind, cudaStream_t, cudaStreamNonBlocking},
};

use crate::{
    driver::{VERSION, describe, peer_access, synchronize_stream},
    machine::{
        Recorded, Status, index, machine, provide_symbols, set_current_context, store,
        submit_status,
    },
    memory::Side,
};

thread_local! {
    static LAST_ERROR: Cell<CUresult> = const { Cell::new(CUresult::CUDA_SUCCESS) };
}

/// Runs `body` and turns its result into a CUDA status, keeping a failure for `cudaGetLastError`.
fn status(body: impl FnOnce() -> Status) -> cudaError_t {
    match body() {
        Ok(()) => CUresult::CUDA_SUCCESS,
        Err(error) => {
            if error != CUresult::CUDA_ERROR_NOT_READY {
                LAST_ERROR.set(error);
            }
            error
        }
    }
}

pub(crate) fn provide(name: &CStr) -> Option<*const ()> {
    provide_symbols!(
        name,
        cudaGetDeviceCount => device_count,
        cudaGetDevice => get_device,
        cudaSetDevice => set_device,
        cudaDeviceSynchronize => device_synchronize,
        cudaDeviceCanAccessPeer => can_access_peer,
        cudaDeviceGetP2PAttribute => peer_attribute,
        cudaDeviceEnablePeerAccess => enable_peer_access,
        cudaDeviceDisablePeerAccess => disable_peer_access,
        cudaRuntimeGetVersion => version,
        cudaDriverGetVersion => version,
        cudaGetLastError => last_error,
        cudaPeekAtLastError => peek_error,
        cudaGetErrorString => error_string,
        cudaGetErrorName => error_name,
        cudaStreamCreate => stream_create,
        cudaStreamCreateWithFlags => stream_create_with_flags,
        cudaStreamCreateWithPriority => stream_create_with_priority,
        cudaStreamDestroy => stream_destroy,
        cudaStreamSynchronize => stream_synchronize,
        cudaStreamQuery => stream_query,
        cudaStreamIsCapturing => stream_is_capturing,
        cudaStreamWaitEvent => stream_wait_event,
        cudaEventCreate => event_create,
        cudaEventCreateWithFlags => event_create_with_flags,
        cudaEventRecord => event_record,
        cudaEventRecordWithFlags => event_record_with_flags,
        cudaEventQuery => event_query,
        cudaEventSynchronize => event_synchronize,
        cudaEventDestroy => event_destroy,
        cudaEventElapsedTime => event_elapsed_time,
        cudaMalloc => malloc,
        cudaMallocAsync => malloc_async,
        cudaFree => free,
        cudaFreeAsync => free_async,
        cudaMemcpy => memcpy,
        cudaMemcpyAsync => memcpy_async,
        cudaMemset => memset,
        cudaMemsetAsync => memset_async,
    )
}

unsafe extern "C" fn device_count(count: *mut c_int) -> cudaError_t {
    status(|| {
        let value =
            c_int::try_from(machine()?.devices.len()).map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
        // SAFETY: `count` is the caller's output slot.
        unsafe { store(count, value) }
    })
}

unsafe extern "C" fn get_device(device: *mut c_int) -> cudaError_t {
    status(|| {
        let current = machine()?.runtime_device()?;
        let current = c_int::try_from(current).map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
        // SAFETY: `device` is the caller's output slot.
        unsafe { store(device, current) }
    })
}

unsafe extern "C" fn set_device(device: c_int) -> cudaError_t {
    status(|| {
        set_current_context(machine()?.primary_context(index(device)?)?);
        Ok(())
    })
}

unsafe extern "C" fn device_synchronize() -> cudaError_t {
    status(|| {
        let machine = machine()?;
        machine.synchronize_device(machine.runtime_device()?)
    })
}

unsafe extern "C" fn can_access_peer(
    result: *mut c_int,
    device: c_int,
    peer: c_int,
) -> cudaError_t {
    status(|| {
        let machine = machine()?;
        let (device, peer) = (index(device)?, index(peer)?);
        machine.device(device)?;
        machine.device(peer)?;
        // SAFETY: `result` is the caller's output slot.
        unsafe {
            store(
                result,
                c_int::from(machine.topology.are_peers(device, peer)),
            )
        }
    })
}

unsafe extern "C" fn peer_attribute(
    value: *mut c_int,
    attribute: CUdevice_P2PAttribute,
    source: c_int,
    destination: c_int,
) -> cudaError_t {
    status(|| {
        let machine = machine()?;
        let (source, destination) = (index(source)?, index(destination)?);
        machine.device(source)?;
        machine.device(destination)?;
        let peers = machine.topology.are_peers(source, destination);
        let result = match attribute {
            CUdevice_P2PAttribute::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
            | CUdevice_P2PAttribute::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED => {
                c_int::from(peers)
            }
            _ => 0,
        };
        // SAFETY: `value` is the caller's output slot.
        unsafe { store(value, result) }
    })
}

unsafe extern "C" fn enable_peer_access(peer: c_int, _flags: c_uint) -> cudaError_t {
    status(|| peer_access(machine()?.runtime_device()?, index(peer)?, true))
}

unsafe extern "C" fn disable_peer_access(peer: c_int) -> cudaError_t {
    status(|| peer_access(machine()?.runtime_device()?, index(peer)?, false))
}

unsafe extern "C" fn version(value: *mut c_int) -> cudaError_t {
    // SAFETY: `value` is the caller's output slot.
    status(|| unsafe { store(value, VERSION) })
}

unsafe extern "C" fn last_error() -> cudaError_t {
    LAST_ERROR.replace(CUresult::CUDA_SUCCESS)
}

unsafe extern "C" fn peek_error() -> cudaError_t {
    LAST_ERROR.get()
}

unsafe extern "C" fn error_string(error: cudaError_t) -> *const c_char {
    describe(error).1.as_ptr()
}

unsafe extern "C" fn error_name(error: cudaError_t) -> *const c_char {
    describe(error).0.as_ptr()
}

fn create_stream(handle: *mut cudaStream_t, flags: c_uint) -> Status {
    let machine = machine()?;
    let stream = machine
        .device(machine.runtime_device()?)?
        .create_stream()
        .map_err(|error| submit_status(&error))?;
    let key = {
        let mut state = machine.state();
        let key = state.handle();
        if flags & cudaStreamNonBlocking != 0 {
            state.non_blocking.insert(stream.id());
        }

        state.streams.insert(key, stream);
        key
    };
    // SAFETY: `handle` is the caller's output slot.
    unsafe { store(handle, ptr::without_provenance_mut(key)) }
}

unsafe extern "C" fn stream_create(handle: *mut cudaStream_t) -> cudaError_t {
    status(|| create_stream(handle, 0))
}

unsafe extern "C" fn stream_create_with_flags(
    handle: *mut cudaStream_t,
    flags: c_uint,
) -> cudaError_t {
    status(|| create_stream(handle, flags))
}

unsafe extern "C" fn stream_create_with_priority(
    handle: *mut cudaStream_t,
    flags: c_uint,
    _priority: c_int,
) -> cudaError_t {
    status(|| create_stream(handle, flags))
}

unsafe extern "C" fn stream_destroy(handle: cudaStream_t) -> cudaError_t {
    status(|| {
        let machine = machine()?;
        let stream = {
            let mut state = machine.state();
            let stream = state
                .streams
                .remove(&handle.addr())
                .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?;
            state.queries.remove(&stream.id());
            state.non_blocking.remove(&stream.id());
            stream
        };
        machine.wait(stream.synchronize())
    })
}

unsafe extern "C" fn stream_synchronize(handle: cudaStream_t) -> cudaError_t {
    status(|| synchronize_stream(handle))
}

unsafe extern "C" fn stream_query(handle: cudaStream_t) -> cudaError_t {
    status(|| {
        let machine = machine()?;
        let device = machine.runtime_device()?;
        let id = machine.named_stream(handle, device)?.id();
        let pending = machine.state().queries.get(&id).cloned();
        let event = if let Some(event) = pending {
            event
        } else {
            let (event, submission) = machine
                .stream(handle, device)?
                .record_event()
                .map_err(|error| submit_status(&error))?;
            drop(submission);
            machine.state().queries.insert(id, event.clone());
            event
        };

        let Some(result) = event.status() else {
            return Err(CUresult::CUDA_ERROR_NOT_READY);
        };
        machine.state().queries.remove(&id);
        result.map_err(|error| machine.keep_fault(error))
    })
}

unsafe extern "C" fn stream_is_capturing(
    _handle: cudaStream_t,
    capture: *mut CUstreamCaptureStatus,
) -> cudaError_t {
    status(|| {
        machine()?;
        // SAFETY: `capture` is the caller's output slot.
        unsafe {
            store(
                capture,
                CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            )
        }
    })
}

unsafe extern "C" fn stream_wait_event(
    handle: cudaStream_t,
    event: cudaEvent_t,
    _flags: c_uint,
) -> cudaError_t {
    status(|| {
        let machine = machine()?;
        let stream = machine.stream(handle, machine.runtime_device()?)?;
        let recorded = machine
            .state()
            .events
            .get(&event.addr())
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)?
            .as_ref()
            .map(|recorded| recorded.event.clone());
        if let Some(event) = recorded {
            drop(
                stream
                    .wait_event(&event)
                    .map_err(|error| submit_status(&error))?,
            );
        }
        Ok(())
    })
}

fn create_event(handle: *mut cudaEvent_t) -> Status {
    let machine = machine()?;
    let key = {
        let mut state = machine.state();
        let key = state.handle();
        state.events.insert(key, None);
        key
    };
    // SAFETY: `handle` is the caller's output slot.
    unsafe { store(handle, ptr::without_provenance_mut(key)) }
}

unsafe extern "C" fn event_create(handle: *mut cudaEvent_t) -> cudaError_t {
    status(|| create_event(handle))
}

unsafe extern "C" fn event_create_with_flags(
    handle: *mut cudaEvent_t,
    _flags: c_uint,
) -> cudaError_t {
    status(|| create_event(handle))
}

fn record(event: cudaEvent_t, handle: cudaStream_t) -> Status {
    let machine = machine()?;
    let stream = machine.stream(handle, machine.runtime_device()?)?;
    if !machine.state().events.contains_key(&event.addr()) {
        return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
    }

    let (recorded, submission) = stream
        .record_event()
        .map_err(|error| submit_status(&error))?;
    drop(submission);
    machine.state().events.insert(
        event.addr(),
        Some(Recorded {
            event: recorded,
            time: Instant::now(),
        }),
    );
    Ok(())
}

unsafe extern "C" fn event_record(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t {
    status(|| record(event, stream))
}

unsafe extern "C" fn event_record_with_flags(
    event: cudaEvent_t,
    stream: cudaStream_t,
    _flags: c_uint,
) -> cudaError_t {
    status(|| record(event, stream))
}

fn recorded(event: cudaEvent_t) -> Result<Option<Recorded>, CUresult> {
    machine()?
        .state()
        .events
        .get(&event.addr())
        .cloned()
        .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)
}

unsafe extern "C" fn event_query(event: cudaEvent_t) -> cudaError_t {
    status(|| match recorded(event)? {
        None => Ok(()),
        Some(recorded) => match recorded.event.status() {
            None => Err(CUresult::CUDA_ERROR_NOT_READY),
            Some(result) => {
                let machine = machine()?;
                result.map_err(|error| machine.keep_fault(error))
            }
        },
    })
}

unsafe extern "C" fn event_synchronize(event: cudaEvent_t) -> cudaError_t {
    status(|| match recorded(event)? {
        None => Ok(()),
        Some(recorded) => {
            let machine = machine()?;
            recorded
                .event
                .wait()
                .map_err(|error| machine.keep_fault(error))
        }
    })
}

unsafe extern "C" fn event_destroy(event: cudaEvent_t) -> cudaError_t {
    status(|| {
        machine()?
            .state()
            .events
            .remove(&event.addr())
            .map(|_| ())
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)
    })
}

unsafe extern "C" fn event_elapsed_time(
    milliseconds: *mut f32,
    start: cudaEvent_t,
    end: cudaEvent_t,
) -> cudaError_t {
    status(|| {
        let (Some(start), Some(end)) = (recorded(start)?, recorded(end)?) else {
            return Err(CUresult::CUDA_ERROR_INVALID_HANDLE);
        };
        if start.event.status().is_none() || end.event.status().is_none() {
            return Err(CUresult::CUDA_ERROR_NOT_READY);
        }

        let elapsed = end.time.saturating_duration_since(start.time).as_secs_f32() * 1000.0;
        // SAFETY: `milliseconds` is the caller's output slot.
        unsafe { store(milliseconds, elapsed) }
    })
}

fn allocate(address: *mut *mut c_void, bytes: size_t) -> Status {
    let machine = machine()?;
    let allocated = machine.allocate(machine.runtime_device()?, bytes)?;
    let allocated = usize::try_from(allocated).map_err(|_| CUresult::CUDA_ERROR_UNKNOWN)?;
    // SAFETY: `address` is the caller's output slot.
    unsafe { store(address, ptr::without_provenance_mut(allocated)) }
}

unsafe extern "C" fn malloc(address: *mut *mut c_void, bytes: size_t) -> cudaError_t {
    status(|| allocate(address, bytes))
}

unsafe extern "C" fn malloc_async(
    address: *mut *mut c_void,
    bytes: size_t,
    _stream: cudaStream_t,
) -> cudaError_t {
    status(|| allocate(address, bytes))
}

unsafe extern "C" fn free(address: *mut c_void) -> cudaError_t {
    status(|| machine()?.free(address.addr() as u64))
}

unsafe extern "C" fn free_async(address: *mut c_void, stream: cudaStream_t) -> cudaError_t {
    status(|| {
        let machine = machine()?;
        machine.wait(
            machine
                .stream(stream, machine.runtime_device()?)?
                .synchronize(),
        )?;
        machine.free(address.addr() as u64)
    })
}

/// Returns the sides of a copy of `kind`, inferring them from the addresses for `cudaMemcpyDefault`.
fn sides(kind: cudaMemcpyKind, destination: u64, source: u64) -> Result<(Side, Side), CUresult> {
    let machine = machine()?;
    Ok(match kind {
        cudaMemcpyKind::cudaMemcpyHostToHost => (Side::Host, Side::Host),
        cudaMemcpyKind::cudaMemcpyHostToDevice => (Side::Device, Side::Host),
        cudaMemcpyKind::cudaMemcpyDeviceToHost => (Side::Host, Side::Device),
        cudaMemcpyKind::cudaMemcpyDeviceToDevice => (Side::Device, Side::Device),
        cudaMemcpyKind::cudaMemcpyDefault => (machine.side(destination), machine.side(source)),
        _ => return Err(CUresult::CUDA_ERROR_INVALID_VALUE),
    })
}

/// Copies `count` bytes in `stream`'s order.
///
/// # Safety
///
/// Host ranges must be valid for `count` bytes until the copy completes.
unsafe fn copy(
    destination: *mut c_void,
    source: *const c_void,
    count: size_t,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> Status {
    let machine = machine()?;
    let (to, from) = (destination.addr() as u64, source.addr() as u64);
    let (to_side, from_side) = sides(kind, to, from)?;
    // SAFETY: the caller guarantees the host ranges.
    unsafe {
        machine.copy(
            (to, to_side),
            (from, from_side),
            count,
            stream,
            machine.runtime_device()?,
        )
    }
}

unsafe extern "C" fn memcpy(
    destination: *mut c_void,
    source: *const c_void,
    count: size_t,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    status(|| {
        // SAFETY: the caller passes ranges valid for `count` bytes, and the copy completes before this returns.
        unsafe { copy(destination, source, count, kind, ptr::null_mut())? };
        synchronize_stream(ptr::null_mut())
    })
}

unsafe extern "C" fn memcpy_async(
    destination: *mut c_void,
    source: *const c_void,
    count: size_t,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t {
    // SAFETY: the caller passes ranges valid for `count` bytes until the copy completes.
    status(|| unsafe { copy(destination, source, count, kind, stream) })
}

fn fill(address: *mut c_void, value: c_int, count: size_t, stream: cudaStream_t) -> Status {
    let machine = machine()?;
    machine.fill(
        address.addr() as u64,
        value as u8,
        count,
        stream,
        machine.runtime_device()?,
    )
}

unsafe extern "C" fn memset(address: *mut c_void, value: c_int, count: size_t) -> cudaError_t {
    status(|| fill(address, value, count, ptr::null_mut()))
}

unsafe extern "C" fn memset_async(
    address: *mut c_void,
    value: c_int,
    count: size_t,
    stream: cudaStream_t,
) -> cudaError_t {
    status(|| fill(address, value, count, stream))
}
