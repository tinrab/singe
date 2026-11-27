use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ptr,
    sync::{Mutex, MutexGuard, OnceLock, PoisonError},
    task::Poll,
    time::Instant,
};

use singe_cuda_sys::driver::{CUresult, CUstream};
use singe_ptx::emulator::{
    Device, Emulator, Event, ExecutionError, HostMemory, Kernel, Module, Stream, Submission,
    SubmitError, Topology,
};

use crate::InstallError;

pub(crate) type Status = Result<(), CUresult>;

/// Primary context handles sit below every handle [`State::handle`] returns.
const PRIMARY_BASE: usize = 0x100;

macro_rules! provide_symbols {
    ($name:expr, $($symbol:ident => $function:path),* $(,)?) => {{
        let name = $name.to_bytes();
        $(if name == stringify!($symbol).as_bytes() {
            Some($function as *const ())
        } else)* {
            None
        }
    }};
}

pub(crate) use provide_symbols;

/// The emulated machine behind the CUDA and NCCL functions, created once by [`install`](crate::install).
pub(crate) struct Machine {
    pub(crate) emulator: Emulator,
    pub(crate) devices: Vec<Device>,
    pub(crate) topology: Topology,
    pub(crate) device_bytes: usize,
    pub(crate) state: Mutex<State>,
}

#[derive(Clone)]
pub(crate) struct Recorded {
    pub(crate) event: Event,
    pub(crate) time: Instant,
}

/// Physical memory from `cuMemCreate`, freed once it is released and no longer mapped.
pub(crate) struct Physical {
    pub(crate) released: bool,
    pub(crate) mappings: usize,
}

#[derive(Default)]
pub(crate) struct State {
    next_handle: usize,
    /// Contexts from `cuCtxCreate`, by handle, with their device.
    pub(crate) contexts: HashMap<usize, usize>,
    pub(crate) primary_references: Vec<usize>,
    pub(crate) streams: HashMap<usize, Stream>,
    /// Ids of streams created with `cudaStreamNonBlocking`, which the legacy default stream does not wait for.
    pub(crate) non_blocking: HashSet<u64>,
    pub(crate) queries: HashMap<u64, Event>,
    pub(crate) events: HashMap<usize, Option<Recorded>>,
    pub(crate) modules: HashMap<usize, Module>,
    pub(crate) functions: HashMap<usize, Kernel>,
    /// Device allocations from `cudaMalloc` and `cuMemAlloc`, by address, with their device and size.
    pub(crate) allocations: BTreeMap<u64, (usize, u64)>,
    pub(crate) physical: HashMap<u64, Physical>,
    /// Reserved address ranges, by first address, with their size.
    pub(crate) reservations: BTreeMap<u64, u64>,
    /// Mappings in reserved ranges, by address, with their size and physical handle.
    pub(crate) mappings: BTreeMap<u64, (u64, u64)>,
    /// Pairs of accessing and owning devices with peer access enabled.
    pub(crate) peers: BTreeSet<(usize, usize)>,
    pub(crate) fault: Option<ExecutionError>,
}

impl State {
    pub(crate) fn new(devices: usize) -> Self {
        Self {
            primary_references: vec![0; devices],
            ..Self::default()
        }
    }

    /// Returns a new handle value, never zero and never one of the special stream handles.
    pub(crate) fn handle(&mut self) -> usize {
        self.next_handle = self.next_handle.max(0x1000) + 0x10;
        self.next_handle
    }

    /// Returns whether `address` lies in device memory: an allocation or a reserved range.
    pub(crate) fn is_device_address(&self, address: u64) -> bool {
        let is_allocated = self
            .allocations
            .range(..=address)
            .next_back()
            .is_some_and(|(start, (_, size))| address - start < *size);
        is_allocated
            || self
                .reservations
                .range(..=address)
                .next_back()
                .is_some_and(|(start, size)| address - start < *size)
    }

    fn is_blocking(&self, stream: &Stream) -> bool {
        !self.non_blocking.contains(&stream.id())
    }
}

static MACHINE: OnceLock<Machine> = OnceLock::new();

thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

/// Installs `machine` for the process, or fails if another machine is already installed.
pub(crate) fn install(machine: Machine) -> Result<(), InstallError> {
    MACHINE.set(machine).map_err(|_| InstallError::Installed)
}

pub(crate) fn installed() -> Option<&'static Machine> {
    MACHINE.get()
}

pub(crate) fn machine() -> Result<&'static Machine, CUresult> {
    installed().ok_or(CUresult::CUDA_ERROR_NOT_INITIALIZED)
}

impl Machine {
    pub(crate) fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn device(&self, index: usize) -> Result<&Device, CUresult> {
        self.devices
            .get(index)
            .ok_or(CUresult::CUDA_ERROR_INVALID_DEVICE)
    }

    /// Returns the handle of `device`'s primary context, which the machine creates up front.
    pub(crate) fn primary_context(&self, device: usize) -> Result<usize, CUresult> {
        self.device(device)?;
        Ok(PRIMARY_BASE + device * 0x10)
    }

    /// Returns the device of the context current on this thread.
    pub(crate) fn current_device(&self) -> Result<usize, CUresult> {
        let current = CURRENT.get();
        if current == 0 {
            return Err(CUresult::CUDA_ERROR_INVALID_CONTEXT);
        }
        self.context_device(current)
    }

    /// Returns the current device as the runtime sees it, making device 0's primary context current if none is.
    pub(crate) fn runtime_device(&self) -> Result<usize, CUresult> {
        if CURRENT.get() == 0 {
            CURRENT.set(self.primary_context(0)?);
        }
        self.current_device()
    }

    pub(crate) fn context_device(&self, context: usize) -> Result<usize, CUresult> {
        if let Some(device) = context
            .checked_sub(PRIMARY_BASE)
            .map(|offset| offset / 0x10)
            .filter(|device| *device < self.devices.len() && context.is_multiple_of(0x10))
        {
            return Ok(device);
        }
        self.state()
            .contexts
            .get(&context)
            .copied()
            .ok_or(CUresult::CUDA_ERROR_INVALID_CONTEXT)
    }

    /// Returns the stream `handle` names; null and the legacy and per-thread handles name `device`'s default stream.
    pub(crate) fn named_stream(&self, handle: CUstream, device: usize) -> Result<Stream, CUresult> {
        if is_default_stream(handle) {
            return Ok(self.device(device)?.default_stream());
        }

        self.state()
            .streams
            .get(&handle.addr())
            .cloned()
            .ok_or(CUresult::CUDA_ERROR_INVALID_HANDLE)
    }

    /// Resolves a stream handle for new work, as [`named_stream`](Self::named_stream) does, and drops the stream's pending `cudaStreamQuery` event, which no longer covers all of its work.
    ///
    /// As with CUDA's legacy default stream, work on the default stream waits for the work already on the device's blocking streams, and work on a blocking stream waits for the work already on the default stream.
    pub(crate) fn stream(&self, handle: CUstream, device: usize) -> Result<Stream, CUresult> {
        let stream = self.named_stream(handle, device)?;
        let waited = if is_default_stream(handle) {
            self.blocking_streams(device)
        } else if self.state().is_blocking(&stream) {
            vec![self.device(stream.device())?.default_stream()]
        } else {
            Vec::new()
        };
        self.state().queries.remove(&stream.id());
        for other in waited {
            let (event, submission) = other
                .record_event()
                .map_err(|error| submit_status(&error))?;
            drop(submission);
            drop(
                stream
                    .wait_event(&event)
                    .map_err(|error| submit_status(&error))?,
            );
        }
        Ok(stream)
    }

    /// Returns `device`'s streams other than its default stream that were not created with `cudaStreamNonBlocking`.
    fn blocking_streams(&self, device: usize) -> Vec<Stream> {
        let state = self.state();
        let streams = state
            .streams
            .values()
            .filter(|stream| stream.device() == device && state.is_blocking(stream))
            .cloned()
            .collect();
        drop(state);
        streams
    }

    /// Waits for every command on `device`'s streams, as `cudaDeviceSynchronize` does.
    pub(crate) fn synchronize_device(&self, device: usize) -> Status {
        let mut streams = vec![self.device(device)?.default_stream()];
        streams.extend(
            self.state()
                .streams
                .values()
                .filter(|stream| stream.device() == device)
                .cloned(),
        );
        for stream in streams {
            self.wait(stream.synchronize())?;
        }
        Ok(())
    }

    /// Waits for `device`'s default stream and its blocking streams, as synchronous copies and the legacy default stream do.
    pub(crate) fn synchronize_legacy(&self, device: usize) -> Status {
        self.wait(self.stream(ptr::null_mut(), device)?.synchronize())
    }

    /// Runs `task` on the worker outside every stream and waits for its result.
    pub(crate) fn host<T: Send + 'static>(
        &self,
        task: impl FnOnce(&mut HostMemory<'_>) -> Result<T, ExecutionError> + Send + 'static,
    ) -> Result<T, CUresult> {
        let mut task = Some(task);
        self.wait(self.emulator.host_task(move |memory| {
            Poll::Ready(task.take().expect("host tasks are polled until ready")(
                memory,
            ))
        }))
    }

    /// Waits for a submission, keeping any execution failure as the machine's fault.
    pub(crate) fn wait<T>(
        &self,
        submission: Result<Submission<T>, SubmitError>,
    ) -> Result<T, CUresult> {
        submission
            .map_err(|error| submit_status(&error))?
            .wait()
            .map_err(|error| self.keep_fault(error))
    }

    /// Returns the CUDA status of `error`, keeping it as the machine's fault if it is the first.
    pub(crate) fn keep_fault(&self, error: ExecutionError) -> CUresult {
        let error = match error {
            ExecutionError::Faulted => self.emulator.fault().unwrap_or(error),
            error => error,
        };
        let status = execution_status(&error);
        let mut state = self.state();
        if state.fault.is_none() && !matches!(error, ExecutionError::Faulted) {
            state.fault = Some(error);
        }
        status
    }
}

pub(crate) fn current_context() -> usize {
    CURRENT.get()
}

pub(crate) fn set_current_context(context: usize) {
    CURRENT.set(context);
}

pub(crate) fn is_default_stream(handle: CUstream) -> bool {
    handle.addr() <= 2
}

pub(crate) fn submit_status(error: &SubmitError) -> CUresult {
    match error {
        SubmitError::Closed | SubmitError::Faulted => CUresult::CUDA_ERROR_LAUNCH_FAILED,
        SubmitError::ForeignResource => CUresult::CUDA_ERROR_INVALID_HANDLE,
        SubmitError::UnknownDevice(_) => CUresult::CUDA_ERROR_INVALID_DEVICE,
        SubmitError::QueueFull | SubmitError::StagingLimit | SubmitError::ExecutionStateLimit => {
            CUresult::CUDA_ERROR_OUT_OF_MEMORY
        }
        _ => CUresult::CUDA_ERROR_INVALID_VALUE,
    }
}

fn execution_status(error: &ExecutionError) -> CUresult {
    match error {
        ExecutionError::Launch { cause, .. } | ExecutionError::Lane { cause, .. } => {
            execution_status(cause)
        }
        ExecutionError::Memory(_) => CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
        ExecutionError::MemoryLimit | ExecutionError::AddressSpaceExhausted(_) => {
            CUresult::CUDA_ERROR_OUT_OF_MEMORY
        }
        ExecutionError::Trap => CUresult::CUDA_ERROR_ASSERT,
        ExecutionError::ExecutionStateLimit | ExecutionError::CallDepth => {
            CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
        }
        _ => CUresult::CUDA_ERROR_LAUNCH_FAILED,
    }
}

/// Runs `body` and turns its result into a CUDA status.
pub(crate) fn status(body: impl FnOnce() -> Status) -> CUresult {
    match body() {
        Ok(()) => CUresult::CUDA_SUCCESS,
        Err(status) => status,
    }
}

/// Writes `value` to the caller's output slot.
///
/// # Safety
///
/// `slot` must be null or valid for a write of `T`.
pub(crate) unsafe fn store<T>(slot: *mut T, value: T) -> Status {
    if slot.is_null() {
        return Err(CUresult::CUDA_ERROR_INVALID_VALUE);
    }

    // SAFETY: the caller guarantees a non-null `slot` is valid for a write of `T`.
    unsafe { slot.write(value) };
    Ok(())
}

pub(crate) fn index(value: i32) -> Result<usize, CUresult> {
    usize::try_from(value).map_err(|_| CUresult::CUDA_ERROR_INVALID_DEVICE)
}

pub(crate) fn size(value: u64) -> Result<usize, CUresult> {
    usize::try_from(value).map_err(|_| CUresult::CUDA_ERROR_INVALID_VALUE)
}
