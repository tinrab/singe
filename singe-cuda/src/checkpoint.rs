use std::{
    fmt::{self, Display, Formatter},
    mem,
    os::raw::c_int,
    process, ptr, slice,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cuda_sys::driver;

use crate::{
    context::Context,
    device::{Device, Uuid},
    driver::entry_points::{cuCheckpointOperationComplete, ensure_supported},
    error::{Error, Result, Status},
    memory::{HostAllocationFlags, PinnedHostMemory},
    try_ffi,
    types::DevicePtr,
};

/// CUDA process state used by the checkpoint and restore driver APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ProcessState {
    /// The process can make CUDA API calls.
    Running = driver::CUprocessState::CU_PROCESS_STATE_RUNNING as _,
    /// CUDA API locks are taken and further CUDA API calls will block.
    Locked = driver::CUprocessState::CU_PROCESS_STATE_LOCKED as _,
    /// GPU memory has been moved to host memory and device handles were released.
    Checkpointed = driver::CUprocessState::CU_PROCESS_STATE_CHECKPOINTED as _,
    /// The process entered an unrecoverable error during checkpoint or restore.
    Failed = driver::CUprocessState::CU_PROCESS_STATE_FAILED as _,
    /// GPU memory is being copied to custom storage.
    Checkpointing = driver::CUprocessState::CU_PROCESS_STATE_CHECKPOINTING as _,
    /// GPU memory is being restored from custom storage.
    Restoring = driver::CUprocessState::CU_PROCESS_STATE_RESTORING as _,
    /// Unknown process state returned by a newer CUDA driver version.
    #[num_enum(catch_all)]
    Unknown(u32),
}

impl_enum_conversion!(driver::CUprocessState, ProcessState);

impl Display for ProcessState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpointing => write!(f, "CU_PROCESS_STATE_CHECKPOINTING"),
            Self::Restoring => write!(f, "CU_PROCESS_STATE_RESTORING"),
            Self::Running => write!(f, "CU_PROCESS_STATE_RUNNING"),
            Self::Locked => write!(f, "CU_PROCESS_STATE_LOCKED"),
            Self::Checkpointed => write!(f, "CU_PROCESS_STATE_CHECKPOINTED"),
            Self::Failed => write!(f, "CU_PROCESS_STATE_FAILED"),
            Self::Unknown(value) => write!(f, "CU_PROCESS_STATE_UNKNOWN({value})"),
        }
    }
}

/// Operating-system process identifier used by checkpoint APIs.
pub type ProcessId = c_int;

/// Options for [`CheckpointProcess::lock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LockOptions {
    timeout: Option<Duration>,
}

impl LockOptions {
    /// Creates lock options without a timeout.
    #[must_use]
    pub const fn new() -> Self {
        Self { timeout: None }
    }

    /// Sets the maximum time CUDA should spend attempting to lock the process.
    ///
    /// A missing timeout passes `0` to CUDA, which means no timeout.
    /// Durations are truncated to whole milliseconds; a submillisecond timeout
    /// therefore also passes zero and disables the timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    fn to_raw(self) -> Result<driver::CUcheckpointLockArgs> {
        let timeout_ms = match self.timeout {
            Some(timeout) => timeout
                .as_millis()
                .try_into()
                .map_err(|_| Error::InvalidValue)?,
            None => 0,
        };

        Ok(driver::CUcheckpointLockArgs {
            timeoutMs: timeout_ms,
            ..Default::default()
        })
    }
}

/// Outcome of [`CheckpointProcess::try_lock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LockResult {
    /// Lock completed and the process entered [`ProcessState::Locked`].
    Locked,
    /// Lock call timed out and the process remained in [`ProcessState::Running`].
    TimedOut,
}

/// GPU UUID remapping entry used during restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuPair {
    /// UUID of the GPU that was checkpointed.
    pub old_uuid: Uuid,
    /// UUID of the GPU to restore onto.
    pub new_uuid: Uuid,
}

impl From<(Uuid, Uuid)> for GpuPair {
    fn from((old_uuid, new_uuid): (Uuid, Uuid)) -> Self {
        Self::new(old_uuid, new_uuid)
    }
}

impl GpuPair {
    /// Creates a GPU pair from the original and target GPU UUIDs.
    #[must_use]
    pub const fn new(old_uuid: Uuid, new_uuid: Uuid) -> Self {
        Self { old_uuid, new_uuid }
    }

    /// Creates a GPU pair from two CUDA devices by reading their UUIDs.
    ///
    /// # Errors
    ///
    /// Propagates CUDA device-property query errors for either the original or target device. No remapping is performed by this constructor.
    pub fn from_devices(old: Device, new: Device) -> Result<Self> {
        let old_uuid = old.properties()?.uuid;
        let new_uuid = new.properties()?.uuid;
        Ok(Self::new(old_uuid, new_uuid))
    }

    fn to_raw(self) -> driver::CUcheckpointGpuPair {
        driver::CUcheckpointGpuPair {
            oldUuid: self.old_uuid.into(),
            newUuid: self.new_uuid.into(),
        }
    }
}

/// Options for [`CheckpointProcess::checkpoint`]. Reserved for future CUDA versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointOptions;

impl CheckpointOptions {
    /// Creates default checkpoint options.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn to_raw(self) -> driver::CUcheckpointCheckpointArgs {
        let _ = self;
        driver::CUcheckpointCheckpointArgs::default()
    }
}

impl Default for CheckpointOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Options for [`CheckpointProcess::restore`], including optional GPU remap entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreOptions {
    gpu_pairs: Vec<driver::CUcheckpointGpuPair>,
}

impl RestoreOptions {
    /// Creates restore options with no GPU remapping pairs.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gpu_pairs: Vec::new(),
        }
    }

    /// Creates restore options from checkpointed/restored GPU UUID pairs.
    pub fn with_gpu_pairs(gpu_pairs: &[GpuPair]) -> Self {
        let mut options = Self::new();
        options.gpu_pairs = gpu_pairs.iter().copied().map(GpuPair::to_raw).collect();
        options
    }

    /// Adds a single GPU remapping pair.
    #[must_use]
    pub fn with_gpu_pair(mut self, pair: GpuPair) -> Self {
        self.gpu_pairs.push(pair.to_raw());
        self
    }

    /// Adds a GPU remapping pair from CUDA devices.
    ///
    /// # Errors
    ///
    /// Propagates CUDA device-property query errors while reading the original or target device UUIDs through [`GpuPair::from_devices`].
    pub fn with_device_pair(mut self, old: Device, new: Device) -> Result<Self> {
        self.push_device_pair(old, new)?;
        Ok(self)
    }

    /// Adds a single GPU remapping pair in place.
    pub fn push_gpu_pair(&mut self, pair: GpuPair) {
        self.gpu_pairs.push(pair.to_raw());
    }

    /// Adds a GPU remapping pair from CUDA devices in place.
    ///
    /// # Errors
    ///
    /// Propagates CUDA device-property query errors while reading the original or target device UUIDs through [`GpuPair::from_devices`].
    pub fn push_device_pair(&mut self, old: Device, new: Device) -> Result<()> {
        let pair = GpuPair::from_devices(old, new)?;
        self.push_gpu_pair(pair);
        Ok(())
    }

    /// Creates restore options from checkpointed/restored GPU device pairs.
    ///
    /// # Errors
    ///
    /// Propagates CUDA device-property query errors while reading the original or target device UUIDs through [`GpuPair::from_devices`].
    pub fn with_device_pairs(gpu_pairs: impl AsRef<[(Device, Device)]>) -> Result<Self> {
        let mut options = Self::new();
        for &(old, new) in gpu_pairs.as_ref() {
            options.push_device_pair(old, new)?;
        }
        Ok(options)
    }

    fn raw_args(&mut self) -> Result<driver::CUcheckpointRestoreArgs> {
        let gpu_pairs_count = self
            .gpu_pairs
            .len()
            .try_into()
            .map_err(|_| Error::InvalidValue)?;
        let gpu_pairs = if self.gpu_pairs.is_empty() {
            ptr::null_mut()
        } else {
            self.gpu_pairs.as_mut_ptr()
        };

        Ok(driver::CUcheckpointRestoreArgs {
            gpuPairs: gpu_pairs,
            gpuPairsCount: gpu_pairs_count,
            ..Default::default()
        })
    }
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&[GpuPair]> for RestoreOptions {
    fn from(gpu_pairs: &[GpuPair]) -> Self {
        Self::with_gpu_pairs(gpu_pairs)
    }
}

impl From<Vec<GpuPair>> for RestoreOptions {
    fn from(gpu_pairs: Vec<GpuPair>) -> Self {
        Self::with_gpu_pairs(&gpu_pairs)
    }
}

/// Options for [`CheckpointProcess::unlock`]. Reserved for future CUDA versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnlockOptions;

impl UnlockOptions {
    /// Creates default unlock options.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn to_raw(self) -> driver::CUcheckpointUnlockArgs {
        let _ = self;
        driver::CUcheckpointUnlockArgs::default()
    }
}

impl Default for UnlockOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// A CUDA process controlled through the driver checkpoint APIs.
///
/// These APIs are intended for an external controller process. Locking a
/// process blocks further CUDA API calls in that process until it is restored
/// and unlocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointProcess {
    pid: ProcessId,
}

impl CheckpointProcess {
    /// Creates a CUDA checkpoint target from an operating-system process ID.
    #[must_use]
    pub const fn new(pid: ProcessId) -> Self {
        Self { pid }
    }

    /// Creates a CUDA checkpoint target from an operating-system process ID.
    #[must_use]
    pub const fn from_pid(pid: ProcessId) -> Self {
        Self::new(pid)
    }

    /// Creates a CUDA checkpoint target for the current process.
    ///
    /// # Panics
    ///
    /// Panics if the current operating-system process ID does not fit [`ProcessId`].
    #[must_use]
    pub fn current() -> Self {
        Self::new(ProcessId::try_from(process::id()).expect("current process ID fits pid_t"))
    }

    /// Returns the operating-system process ID controlled by this value.
    #[must_use]
    pub const fn pid(self) -> ProcessId {
        self.pid
    }

    /// Returns the current CUDA checkpoint state of the process.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the target process's checkpoint state, for example because the process is unavailable or checkpoint control is unsupported. Also returns `Error::InvalidValue` if the native state cannot be converted to [`ProcessState`].
    pub fn state(self) -> Result<ProcessState> {
        let mut state = driver::CUprocessState::CU_PROCESS_STATE_RUNNING;
        // SAFETY: `cuCheckpointProcessGetState` receives valid output locations.
        unsafe {
            try_ffi!(driver::cuCheckpointProcessGetState(
                self.pid,
                &raw mut state,
            ))?;
        }
        ProcessState::try_from(state as u32).map_err(|_| Error::InvalidValue)
    }

    /// Returns `true` when this process currently has [`ProcessState::Running`].
    #[must_use]
    pub fn is_running(self) -> bool {
        self.state()
            .is_ok_and(|state| state == ProcessState::Running)
    }

    /// Returns `true` when this process currently has [`ProcessState::Locked`].
    #[must_use]
    pub fn is_locked(self) -> bool {
        self.state()
            .is_ok_and(|state| state == ProcessState::Locked)
    }

    /// Returns `true` when this process currently has [`ProcessState::Checkpointed`].
    #[must_use]
    pub fn is_checkpointed(self) -> bool {
        self.state()
            .is_ok_and(|state| state == ProcessState::Checkpointed)
    }

    /// Blocks until the process reaches `expected`, or returns a timeout error.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the timeout overflows the host monotonic clock, or the native timeout status if `expected` is not reached before the polling deadline. Also propagates state-query errors. The state is queried immediately and then every 25 milliseconds.
    pub fn wait_for_state(self, expected: ProcessState, timeout: Duration) -> Result<ProcessState> {
        const POLL_INTERVAL: Duration = Duration::from_millis(25);
        let end = Instant::now()
            .checked_add(timeout)
            .ok_or(Error::InvalidValue)?;

        loop {
            let state = self.state()?;
            if state == expected {
                return Ok(state);
            }
            if Instant::now() >= end {
                return Err(Error::Cuda {
                    code: Status::Timeout,
                    message: format!(
                        "timed out waiting for checkpoint process {} to reach {}",
                        self.pid, expected
                    ),
                });
            }
            thread::sleep(POLL_INTERVAL);
        }
    }

    /// Returns the CUDA restore thread ID for the process.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot query the target process's restore thread, including an unavailable target or unsupported checkpoint control.
    pub fn restore_thread_id(self) -> Result<i32> {
        let mut thread_id = 0;
        // SAFETY: `cuCheckpointProcessGetRestoreThreadId` receives valid output locations.
        unsafe {
            try_ffi!(driver::cuCheckpointProcessGetRestoreThreadId(
                self.pid,
                &raw mut thread_id,
            ))?;
        }
        Ok(thread_id)
    }

    /// Locks a running CUDA process so further CUDA API calls in that process block.
    ///
    /// On success the process enters [`ProcessState::Locked`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::try_lock`]. A timed-out lock attempt is converted from [`LockResult::TimedOut`] to the native not-ready error status.
    pub fn lock(self, options: LockOptions) -> Result<()> {
        match self.try_lock(options)? {
            LockResult::Locked => Ok(()),
            LockResult::TimedOut => Err(driver::CUresult::CUDA_ERROR_NOT_READY.into()),
        }
    }

    /// Attempts to lock a running CUDA process and reports whether the timeout was hit.
    ///
    /// On success the process usually enters [`ProcessState::Locked`]. If a timeout
    /// is set and reached, this returns [`LockResult::TimedOut`].
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the timeout in milliseconds does not fit the native timeout field. Also propagates native lock failures for invalid process state, unavailable targets, or unsupported checkpoint control. The native not-ready status is returned as [`LockResult::TimedOut`] rather than an error.
    pub fn try_lock(self, options: LockOptions) -> Result<LockResult> {
        let mut args = options.to_raw()?;
        // SAFETY: `cuCheckpointProcessLock` receives valid output locations.
        let result = unsafe { driver::cuCheckpointProcessLock(self.pid, &raw mut args) };
        match result {
            driver::CUresult::CUDA_SUCCESS => Ok(LockResult::Locked),
            driver::CUresult::CUDA_ERROR_NOT_READY => Ok(LockResult::TimedOut),
            status => Err(status.into()),
        }
    }

    /// Moves the locked process's GPU memory into host memory managed by the driver.
    ///
    /// On success the process enters [`ProcessState::Checkpointed`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::checkpoint_with_options`] using the default reserved options.
    pub fn checkpoint(self) -> Result<()> {
        self.checkpoint_with_options(CheckpointOptions::new())
    }

    /// Moves the locked process's GPU memory into host memory managed by the driver.
    ///
    /// This variant accepts explicit checkpoint options.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot checkpoint the target, including when it is not locked, the platform or process uses unsupported features, or host storage for saved GPU state cannot be provided.
    pub fn checkpoint_with_options(self, options: CheckpointOptions) -> Result<()> {
        let mut args = options.to_raw();
        // SAFETY: `cuCheckpointProcessCheckpoint` receives valid output locations.
        unsafe {
            try_ffi!(driver::cuCheckpointProcessCheckpoint(
                self.pid,
                &raw mut args
            ))
        }
    }

    /// Locks and checkpoints a running CUDA process.
    ///
    /// On success the process enters [`ProcessState::Checkpointed`].
    ///
    /// # Errors
    ///
    /// Propagates lock or checkpoint errors. A successful lock is not undone if the following checkpoint operation fails.
    pub fn suspend(self, options: LockOptions) -> Result<()> {
        self.lock(options)?;
        self.checkpoint()
    }

    /// Toggles the process between running and checkpointed states.
    ///
    /// - From `Running`, performs [`CheckpointProcess::suspend`].
    /// - From `Checkpointed`, performs [`CheckpointProcess::resume`].
    ///
    /// # Errors
    ///
    /// Propagates state-query, suspend, restore, or unlock errors. Returns the native illegal-state status if the target is neither running nor checkpointed. Successful earlier state transitions are not rolled back if a later step fails.
    pub fn toggle(
        self,
        options: LockOptions,
        gpu_pairs: impl AsRef<[GpuPair]>,
    ) -> Result<ProcessState> {
        match self.state()? {
            ProcessState::Running => {
                self.suspend(options)?;
                Ok(ProcessState::Checkpointed)
            }
            ProcessState::Checkpointed => {
                self.resume(gpu_pairs)?;
                Ok(ProcessState::Running)
            }
            _ => Err(Error::Cuda {
                code: Status::IllegalState,
                message: String::from("cannot toggle checkpoint process from current state"),
            }),
        }
    }

    /// Restores a checkpointed process, optionally remapping checkpointed GPUs.
    ///
    /// If `gpu_pairs` is not empty, CUDA requires it to contain every
    /// checkpointed GPU.
    ///
    /// On success the process enters [`ProcessState::Locked`].
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` if the remapping-pair count exceeds the native integer range. Also propagates native restore errors for invalid target state, incomplete or incompatible GPU remapping, unavailable devices, or insufficient resources.
    pub fn restore_with_options(self, mut options: RestoreOptions) -> Result<()> {
        let mut args = options.raw_args()?;
        // SAFETY: `cuCheckpointProcessRestore` receives valid output locations.
        unsafe { try_ffi!(driver::cuCheckpointProcessRestore(self.pid, &raw mut args)) }
    }

    /// Restores a checkpointed process, optionally remapping checkpointed GPUs.
    ///
    /// If `gpu_pairs` is not empty, CUDA requires it to contain every
    /// checkpointed GPU.
    ///
    /// On success the process enters [`ProcessState::Locked`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::restore_with_options`] using the supplied GPU remapping pairs.
    pub fn restore(self, gpu_pairs: impl AsRef<[GpuPair]>) -> Result<()> {
        self.restore_with_options(gpu_pairs.as_ref().into())
    }

    /// Restores and unlocks a checkpointed CUDA process.
    ///
    /// On success the process enters [`ProcessState::Running`].
    ///
    /// # Errors
    ///
    /// Propagates restore or unlock errors. A successful restore is not rolled back if the following unlock operation fails.
    pub fn resume(self, gpu_pairs: impl AsRef<[GpuPair]>) -> Result<()> {
        self.restore(gpu_pairs)?;
        self.unlock()
    }

    /// Unlocks a locked CUDA process so it can resume CUDA API calls.
    ///
    /// On success the process enters [`ProcessState::Running`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::unlock_with_options`] using the default reserved options.
    pub fn unlock(self) -> Result<()> {
        self.unlock_with_options(UnlockOptions::new())
    }

    /// Unlocks a locked CUDA process so it can resume CUDA API calls with options.
    ///
    /// On success the process enters [`ProcessState::Running`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot unlock the target, including when it is not in a state that permits unlocking or checkpoint control is unsupported.
    pub fn unlock_with_options(self, options: UnlockOptions) -> Result<()> {
        let mut args = options.to_raw();
        // SAFETY: `cuCheckpointProcessUnlock` receives valid output locations.
        unsafe { try_ffi!(driver::cuCheckpointProcessUnlock(self.pid, &raw mut args)) }
    }
}

/// Output from a CUDA custom-storage checkpoint or restore operation.
///
/// Complete the operation after copying all regions. Dropping an incomplete
/// operation leaves the target suspended and retains its primary contexts;
/// CUDA provides no cancellation operation that can safely restore missing data.
#[derive(Debug)]
#[must_use = "custom-storage operations must be explicitly completed"]
pub struct StorageOperation {
    handle: driver::CUcheckpointOperationHandle,
    regions: Vec<driver::CUcheckpointCustomStoragePerDeviceData>,
    contexts: Vec<Arc<Context>>,
    restoring: bool,
}

/// A GPU mapping and stream borrowed from a pending custom-storage operation.
#[derive(Debug)]
pub struct StorageRegion<'a> {
    raw: &'a driver::CUcheckpointCustomStoragePerDeviceData,
    contexts: &'a [Arc<Context>],
    restoring: bool,
}

impl StorageRegion<'_> {
    /// Size of the mapped GPU data in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the native mapped byte size does not fit the host address space.
    #[must_use]
    pub fn size(&self) -> usize {
        usize::try_from(self.raw.size).expect("mapped storage region fits host address space")
    }

    /// Returns the mapped device address borrowed from the pending storage operation.
    ///
    /// The address becomes invalid when the operation is successfully completed. Its contents are read for checkpointing and written for restoration.
    #[must_use]
    pub fn device_pointer(&self) -> DevicePtr {
        // SAFETY: the address is only stored; dereferencing it requires further unsafe code.
        unsafe { DevicePtr::from_raw(self.raw.devPtr as *mut ()) }
    }

    /// Borrows the stream; it must not be destroyed or put into capture mode.
    #[must_use]
    pub fn stream_handle(&self) -> driver::CUstream {
        self.raw.stream
    }

    fn context(&self) -> Result<&Arc<Context>> {
        let mut context = ptr::null_mut();
        // SAFETY: `cuStreamGetCtx` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(driver::cuStreamGetCtx(self.raw.stream, &raw mut context))?;
        }
        self.contexts
            .iter()
            .find(|owner| owner.as_raw() == context)
            .ok_or(Error::StreamContextMismatch)
    }

    /// Copies mapped GPU data to host storage and waits for the copy.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` during restoration or for a destination whose byte length differs from the mapped region. Also propagates stream-context lookup, context binding, pinned staging allocation, asynchronous copy, and stream synchronization errors.
    pub fn copy_to_host(&self, destination: &mut [u8]) -> Result<()> {
        if self.restoring || destination.len() != self.size() {
            return Err(Error::InvalidValue);
        }
        let context = self.context()?;
        context.bind()?;
        let mut pinned = PinnedHostMemory::<u8>::create_uninit(
            context,
            destination.len(),
            HostAllocationFlags::DEFAULT,
        )?;
        // SAFETY: `cuMemcpyDtoHAsync`, `cuStreamSynchronize` receive live handles owned by the wrappers and pointers that stay valid for each call.
        unsafe {
            try_ffi!(driver::cuMemcpyDtoHAsync(
                pinned.as_slice_mut().as_mut_ptr().cast(),
                self.raw.devPtr,
                self.raw.size,
                self.raw.stream
            ))?;
            try_ffi!(driver::cuStreamSynchronize(self.raw.stream))?;
        }
        // SAFETY: the successful transfer initialized the entire byte buffer and completed.
        let pinned = unsafe { pinned.assume_init() };
        destination.copy_from_slice(pinned.as_slice());
        Ok(())
    }

    /// Copies host storage into a restore mapping and waits for the copy.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidValue` during checkpointing or for a source whose byte length differs from the restore region. Also propagates stream-context lookup, context binding, pinned staging initialization, asynchronous copy, and stream synchronization errors.
    pub fn copy_from_host(&self, source: &[u8]) -> Result<()> {
        if !self.restoring || source.len() != self.size() {
            return Err(Error::InvalidValue);
        }
        let context = self.context()?;
        context.bind()?;
        let pinned = PinnedHostMemory::from_slice(context, source)?;
        // SAFETY: `cuMemcpyHtoDAsync`, `cuStreamSynchronize` receive live handles owned by the wrappers and pointers that stay valid for each call.
        unsafe {
            try_ffi!(driver::cuMemcpyHtoDAsync(
                self.raw.devPtr,
                pinned.as_slice().as_ptr().cast(),
                self.raw.size,
                self.raw.stream
            ))?;
            try_ffi!(driver::cuStreamSynchronize(self.raw.stream))
        }
    }
}

impl StorageOperation {
    /// Iterates over the device mappings and streams borrowed from this pending storage operation.
    ///
    /// The iterator is empty after successful completion.
    pub fn regions(&self) -> impl ExactSizeIterator<Item = StorageRegion<'_>> {
        self.regions.iter().map(|raw| StorageRegion {
            raw,
            contexts: &self.contexts,
            restoring: self.restoring,
        })
    }

    /// Synchronizes the returned streams, unmaps GPU memory and completes the operation.
    ///
    /// # Safety
    /// All required checkpoint or restore copies must have completed or been
    /// enqueued on the returned streams. Restored contents must match the saved
    /// checkpoint. No access to the mappings may occur after successful completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver completion entry point is unavailable or CUDA cannot synchronize and finalize the custom-storage operation. On failure, this wrapper keeps its handle, regions, and retained contexts for another completion attempt. Calling it again after success returns `Ok(())`.
    pub unsafe fn complete(&mut self) -> Result<()> {
        if self.handle.is_null() {
            return Ok(());
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            cuCheckpointOperationComplete(self.handle)?;
        }
        self.handle = ptr::null_mut();
        self.regions.clear();
        self.contexts.clear();
        Ok(())
    }
}

impl Drop for StorageOperation {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Releasing these contexts would invalidate the driver's pending mappings.
            for context in self.contexts.drain(..) {
                mem::forget(context);
            }
        }
    }
}

impl CheckpointProcess {
    /// Maps the target process's GPU memory for copying to custom storage.
    /// The target must be locked. Checkpointing the calling process is unsupported.
    ///
    /// # Errors
    ///
    /// Returns an error if the required driver support is unavailable, the target is the calling process, CUDA initialization or device enumeration fails, a primary context cannot be retained, or CUDA cannot start the requested state transition. Also rejects null native storage-info pointers. The operation can already have started when a later output-validation error is returned.
    pub fn begin_checkpoint_to_custom_storage(self) -> Result<StorageOperation> {
        self.begin_storage_operation(None)
    }

    /// Maps the target's restore memory for copying data from custom storage.
    /// The target must be checkpointed. Restoring the calling process is unsupported.
    ///
    /// # Errors
    ///
    /// Returns an error if the required driver support is unavailable, the target is the calling process, CUDA initialization or device enumeration fails, a primary context cannot be retained, or CUDA cannot start the requested state transition. Also rejects null native storage-info pointers. The operation can already have started when a later output-validation error is returned. Also returns `Error::InvalidValue` if the supplied GPU remapping-pair count exceeds the native integer range.
    pub fn begin_restore_from_custom_storage(
        self,
        options: RestoreOptions,
    ) -> Result<StorageOperation> {
        self.begin_storage_operation(Some(options))
    }

    fn begin_storage_operation(self, restore: Option<RestoreOptions>) -> Result<StorageOperation> {
        ensure_supported()?;
        if u32::try_from(self.pid) == Ok(process::id()) {
            return Err(Error::InvalidValue);
        }
        // SAFETY: `cuInit` receives only plain values.
        unsafe {
            try_ffi!(driver::cuInit(0))?;
        }
        let contexts = (0..Device::count()?)
            .map(|id| Context::retain_primary_for_device(Device::new(id)))
            .collect::<Result<Vec<_>>>()?;
        let restoring = restore.is_some();
        let mut info = ptr::null_mut();
        if let Some(mut options) = restore {
            let mut args = options.raw_args()?;
            args.customStorageInfo_out = &raw mut info;
            // SAFETY: `cuCheckpointProcessRestore` receives valid output locations.
            unsafe {
                try_ffi!(driver::cuCheckpointProcessRestore(self.pid, &raw mut args))?;
            }
        } else {
            let mut args = driver::CUcheckpointCheckpointArgs {
                customStorageInfo_out: &raw mut info,
                ..Default::default()
            };
            // SAFETY: `cuCheckpointProcessCheckpoint` receives valid output locations.
            unsafe {
                try_ffi!(driver::cuCheckpointProcessCheckpoint(
                    self.pid,
                    &raw mut args
                ))?;
            }
        }
        if info.is_null() {
            return Err(Error::NullHandle);
        }
        // SAFETY: the driver passes valid checkpoint info for the duration of this callback.
        let info = unsafe { &*info };
        let regions = if info.deviceCount == 0 {
            Vec::new()
        } else {
            if info.perDeviceData.is_null() {
                return Err(Error::NullHandle);
            }
            // SAFETY: the driver provides `deviceCount` per-device entries for the duration of this callback.
            unsafe { slice::from_raw_parts(info.perDeviceData, info.deviceCount as usize).to_vec() }
        };
        Ok(StorageOperation {
            handle: info.handle,
            regions,
            contexts,
            restoring,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() {
        let process = CheckpointProcess::current();
        match process.state() {
            Ok(state) => assert!(matches!(
                state,
                ProcessState::Running
                    | ProcessState::Locked
                    | ProcessState::Checkpointed
                    | ProcessState::Failed
                    | ProcessState::Unknown(_)
            )),
            Err(error) => assert_checkpoint_error(error),
        }

        let missing_process = CheckpointProcess::from_pid(-1);
        checkpoint_fails(missing_process.restore_thread_id());
        checkpoint_fails(
            missing_process.lock(LockOptions::new().with_timeout(Duration::from_millis(1))),
        );
        checkpoint_fails(missing_process.checkpoint());
        checkpoint_fails(missing_process.checkpoint_with_options(CheckpointOptions::new()));
        checkpoint_fails(missing_process.restore([]));
        checkpoint_fails(missing_process.restore_with_options(RestoreOptions::new()));
        checkpoint_fails(missing_process.unlock());
        checkpoint_fails(missing_process.unlock_with_options(UnlockOptions::new()));
    }

    fn checkpoint_fails<T>(result: Result<T>) {
        match result {
            Err(error) => assert_checkpoint_error(error),
            Ok(_) => panic!("checkpoint call unexpectedly succeeded"),
        }
    }

    fn assert_checkpoint_error(error: Error) {
        match error {
            Error::Cuda {
                code:
                    Status::InvalidValue
                    | Status::NotInitialized
                    | Status::NotSupported
                    | Status::IllegalState
                    | Status::OperatingSystem,
                ..
            } => {}
            error => panic!("{error:?}"),
        }
    }
}
