//! CUDA process checkpoint, restore, and checkpoint storage APIs.

use std::{
    fmt::{self, Display, Formatter},
    mem,
    os::raw::c_int,
    process, ptr, slice,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use singe_cuda_sys::driver;

use crate::{
    context::Context,
    device::{Device, Uuid},
    driver::entry_points::{cuCheckpointOperationComplete, ensure_supported},
    error::{Error, Result, Status},
    memory::PinnedVec,
    try_ffi,
    types::DevicePtr,
    utility::to_usize,
};

singe_core::native_enum! {
    /// CUDA process state used by the checkpoint and restore driver APIs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[non_exhaustive]
    pub enum ProcessState: driver::CUprocessState, unknown = Unknown(u32) {
        /// The process can make CUDA API calls.
        Running = CU_PROCESS_STATE_RUNNING,
        /// CUDA API locks are taken and further CUDA API calls will block.
        Locked = CU_PROCESS_STATE_LOCKED,
        /// GPU memory has been moved to host memory and device handles were released.
        Checkpointed = CU_PROCESS_STATE_CHECKPOINTED,
        /// The process entered an unrecoverable error during checkpoint or restore.
        Failed = CU_PROCESS_STATE_FAILED,
        /// GPU memory is being copied to custom storage.
        Checkpointing = CU_PROCESS_STATE_CHECKPOINTING,
        /// GPU memory is being restored from custom storage.
        Restoring = CU_PROCESS_STATE_RESTORING,
    }
}

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
    /// Durations are truncated to whole milliseconds; a submillisecond timeout therefore also passes zero and disables the timeout.
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
                .map_err(|_| Error::out_of_range("lock timeout"))?,
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
    /// Returns an error if CUDA cannot report either device's properties.
    /// This constructor performs no remapping.
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

/// Options for [`CheckpointProcess::checkpoint`].
/// Reserved for future CUDA versions.
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
    #[must_use]
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
    /// Returns an error if [`GpuPair::from_devices`] cannot read either device's UUID.
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
    /// Returns an error if [`GpuPair::from_devices`] cannot read either device's UUID.
    pub fn push_device_pair(&mut self, old: Device, new: Device) -> Result<()> {
        let pair = GpuPair::from_devices(old, new)?;
        self.push_gpu_pair(pair);
        Ok(())
    }

    /// Creates restore options from checkpointed/restored GPU device pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if [`GpuPair::from_devices`] cannot read a device's UUID.
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
            .map_err(|_| Error::out_of_range("gpu remapping pair count"))?;
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

/// Options for [`CheckpointProcess::unlock`].
/// Reserved for future CUDA versions.
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
/// These APIs are intended for an external controller process.
/// Locking a process blocks further CUDA API calls in that process until it is restored and unlocked.
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
    /// Returns an error if CUDA cannot query the target process's checkpoint state, for example because the process is unavailable or checkpoint control is unsupported.
    pub fn state(self) -> Result<ProcessState> {
        let mut state = driver::CUprocessState::CU_PROCESS_STATE_RUNNING;
        // SAFETY: `state` is a local output, and the driver validates the process ID.
        unsafe {
            try_ffi!(driver::cuCheckpointProcessGetState(
                self.pid,
                &raw mut state,
            ))?;
        }
        Ok(ProcessState::from(state))
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
    /// Returns an error if the timeout overflows the monotonic clock, a state query fails, or the process does not reach `expected` before the deadline, with [`Status::Timeout`].
    /// The state is queried immediately and then every 25 milliseconds.
    pub fn wait_for_state(self, expected: ProcessState, timeout: Duration) -> Result<ProcessState> {
        const POLL_INTERVAL: Duration = Duration::from_millis(25);
        let end = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| Error::out_of_range("wait timeout"))?;

        loop {
            let state = self.state()?;
            if state == expected {
                return Ok(state);
            }

            if Instant::now() >= end {
                return Err(Error::Cuda {
                    code: Status::Timeout,
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
        // SAFETY: `thread_id` is a local output, and the driver validates the process ID.
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
    /// Returns an error if [`Self::try_lock`] fails or times out, with [`Status::NotReady`] for a timeout.
    pub fn lock(self, options: LockOptions) -> Result<()> {
        match self.try_lock(options)? {
            LockResult::Locked => Ok(()),
            LockResult::TimedOut => Err(driver::CUresult::CUDA_ERROR_NOT_READY.into()),
        }
    }

    /// Attempts to lock a running CUDA process and reports whether the timeout was hit.
    ///
    /// On success the process usually enters [`ProcessState::Locked`].
    /// If a timeout is set and reached, this returns [`LockResult::TimedOut`].
    ///
    /// # Errors
    ///
    /// Returns an error if the timeout in milliseconds does not fit the native field, or CUDA cannot lock the target, such as for an invalid process state.
    /// A timeout is returned as [`LockResult::TimedOut`] rather than an error.
    pub fn try_lock(self, options: LockOptions) -> Result<LockResult> {
        let mut args = options.to_raw()?;
        // SAFETY: `args` is a local argument struct that lives through the call, and the driver validates the process ID.
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
    /// Returns an error if CUDA cannot checkpoint the target, including when it is not locked, the platform or process uses unsupported features, or host storage for saved GPU state cannot be provided.
    pub fn checkpoint(self, options: CheckpointOptions) -> Result<()> {
        let mut args = options.to_raw();
        // SAFETY: `args` is a local argument struct with no custom-storage output, and the driver validates the process ID.
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
    /// Returns an error if locking or checkpointing fails.
    /// A successful lock is not undone if the checkpoint fails.
    pub fn suspend(self, options: LockOptions) -> Result<()> {
        self.lock(options)?;
        self.checkpoint(CheckpointOptions::new())
    }

    /// Toggles the process between running and checkpointed states.
    ///
    /// - From `Running`, performs [`CheckpointProcess::suspend`].
    /// - From `Checkpointed`, performs [`CheckpointProcess::resume`].
    ///
    /// # Errors
    ///
    /// Returns an error if the state query, suspend, restore, or unlock fails, or the target is neither running nor checkpointed, with [`Status::IllegalState`].
    /// Successful earlier state transitions are not rolled back if a later step fails.
    pub fn toggle(
        self,
        lock_options: LockOptions,
        restore_options: RestoreOptions,
    ) -> Result<ProcessState> {
        match self.state()? {
            ProcessState::Running => {
                self.suspend(lock_options)?;
                Ok(ProcessState::Checkpointed)
            }
            ProcessState::Checkpointed => {
                self.resume(restore_options)?;
                Ok(ProcessState::Running)
            }
            _ => Err(Error::Cuda {
                code: Status::IllegalState,
            }),
        }
    }

    /// Restores a checkpointed process, optionally remapping checkpointed GPUs.
    ///
    /// If `options` has GPU remapping pairs, CUDA requires them to cover every checkpointed GPU.
    ///
    /// On success the process enters [`ProcessState::Locked`].
    ///
    /// # Errors
    ///
    /// Returns an error if the remapping-pair count does not fit the native type, or CUDA cannot restore the target, such as for an invalid state or incomplete GPU remapping.
    pub fn restore(self, mut options: RestoreOptions) -> Result<()> {
        let mut args = options.raw_args()?;
        // SAFETY: `args` points at the counted remapping pairs in `options`, which outlives the call, and the driver validates the process ID.
        unsafe { try_ffi!(driver::cuCheckpointProcessRestore(self.pid, &raw mut args)) }
    }

    /// Restores and unlocks a checkpointed CUDA process.
    ///
    /// On success the process enters [`ProcessState::Running`].
    ///
    /// # Errors
    ///
    /// Returns an error if the restore or unlock fails.
    /// A successful restore is not rolled back if the unlock fails.
    pub fn resume(self, options: RestoreOptions) -> Result<()> {
        self.restore(options)?;
        self.unlock(UnlockOptions::new())
    }

    /// Unlocks a locked CUDA process so it can resume CUDA API calls.
    ///
    /// On success the process enters [`ProcessState::Running`].
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot unlock the target, including when it is not in a state that permits unlocking or checkpoint control is unsupported.
    pub fn unlock(self, options: UnlockOptions) -> Result<()> {
        let mut args = options.to_raw();
        // SAFETY: `args` is a local argument struct that lives through the call, and the driver validates the process ID.
        unsafe { try_ffi!(driver::cuCheckpointProcessUnlock(self.pid, &raw mut args)) }
    }
}

/// Output from a CUDA custom-storage checkpoint or restore operation.
///
/// Complete the operation after copying all regions.
/// Dropping an incomplete operation leaves the target suspended and retains its primary contexts; CUDA provides no cancellation operation that can safely restore missing data.
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
    /// Returns the size of the mapped GPU data in bytes.
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
    /// The address becomes invalid when the operation is successfully completed.
    /// Its contents are read for checkpointing and written for restoration.
    #[must_use]
    pub fn device_pointer(&self) -> DevicePtr {
        // SAFETY: the address is the pending operation's mapping, and `DevicePtr` claims no access to it.
        unsafe { DevicePtr::from_raw(self.raw.devPtr as *mut ()) }
    }

    fn context(&self) -> Result<&Arc<Context>> {
        let mut context = ptr::null_mut();
        // SAFETY: the pending operation keeps its stream live while this region borrows it, and `context` is a local output.
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
    /// Returns an error if the operation is a restore, with [`Error::CheckpointDirectionMismatch`].
    /// Returns an error if `destination` differs in length from the region, or the context lookup, staging allocation, copy, or synchronization fails.
    pub fn copy_to_host(&self, destination: &mut [u8]) -> Result<()> {
        if self.restoring {
            return Err(Error::CheckpointDirectionMismatch {
                operation: "restore",
            });
        }

        if destination.len() != self.size() {
            return Err(Error::size_mismatch(
                "destination",
                self.size(),
                destination.len(),
            ));
        }

        let context = self.context()?;
        context.bind()?;
        let mut pinned = context
            .pinned_host()
            .with_capacity::<u8>(destination.len())?;
        // SAFETY: `pinned` has capacity for the region's `size` bytes and is held until the stream synchronizes, and the operation keeps the mapping and stream live.
        unsafe {
            try_ffi!(driver::cuMemcpyDtoHAsync(
                pinned.as_mut_ptr().cast(),
                self.raw.devPtr,
                self.raw.size,
                self.raw.stream
            ))?;
        }

        let mut pinned = self.synchronize_holding(pinned)?;

        // SAFETY: the successful transfer initialized the entire capacity and completed.
        unsafe { pinned.set_len(destination.len()) };
        destination.copy_from_slice(&pinned);
        Ok(())
    }

    /// Copies host storage into a restore mapping and waits for the copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is a checkpoint, with [`Error::CheckpointDirectionMismatch`].
    /// Returns an error if `source` differs in length from the region, or the context lookup, staging allocation, copy, or synchronization fails.
    pub fn copy_from_host(&self, source: &[u8]) -> Result<()> {
        if !self.restoring {
            return Err(Error::CheckpointDirectionMismatch {
                operation: "checkpoint",
            });
        }

        if source.len() != self.size() {
            return Err(Error::size_mismatch("source", self.size(), source.len()));
        }

        let context = self.context()?;
        context.bind()?;
        let pinned = context.pinned_host().from_slice(source)?;
        // SAFETY: `pinned` holds the region's `size` initialized bytes and is held until the stream synchronizes, and the operation keeps the mapping and stream live.
        unsafe {
            try_ffi!(driver::cuMemcpyHtoDAsync(
                self.raw.devPtr,
                pinned.as_ptr().cast(),
                self.raw.size,
                self.raw.stream
            ))?;
        }

        self.synchronize_holding(pinned).map(drop)
    }

    fn synchronize_holding(&self, staging: PinnedVec<u8>) -> Result<PinnedVec<u8>> {
        // SAFETY: the pending operation keeps its stream live while this region borrows it.
        if let Err(error) = unsafe { try_ffi!(driver::cuStreamSynchronize(self.raw.stream)) } {
            // The copy may still use the staging, so it never returns to the pool.
            mem::forget(staging);
            return Err(error);
        }
        Ok(staging)
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
    ///
    /// All required checkpoint or restore copies must have completed or been enqueued on the returned streams.
    /// Restored contents must match the saved checkpoint.
    /// No access to the mappings may occur after successful completion.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver completion entry point is unavailable or CUDA cannot synchronize and finalize the custom-storage operation.
    /// On failure, this wrapper keeps its handle, regions, and retained contexts for another completion attempt.
    /// Calling it again after success returns `Ok(())`.
    pub unsafe fn complete(&mut self) -> Result<()> {
        if self.handle.is_null() {
            return Ok(());
        }

        // SAFETY: the handle is the live pending operation, and the caller guarantees every copy is complete or enqueued and the mappings are no longer used.
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
    /// The target must be locked.
    /// Checkpointing the calling process is unsupported.
    ///
    /// # Errors
    ///
    /// Returns an error if the required driver support is unavailable, the target is the calling process, CUDA initialization or device enumeration fails, a primary context cannot be retained, or CUDA cannot start the requested state transition.
    /// Returns an error if CUDA returns null storage information.
    /// The operation can already have started when a later output-validation error is returned.
    pub fn begin_checkpoint_to_custom_storage(self) -> Result<StorageOperation> {
        self.begin_storage_operation(None)
    }

    /// Maps the target's restore memory for copying data from custom storage.
    /// The target must be checkpointed.
    /// Restoring the calling process is unsupported.
    ///
    /// # Errors
    ///
    /// Returns an error if the required driver support is unavailable, the target is the calling process, CUDA initialization or device enumeration fails, a primary context cannot be retained, or CUDA cannot start the requested state transition.
    /// Returns an error if CUDA returns null storage information or the remapping-pair count does not fit the native type.
    /// The operation can already have started when a later output-validation error is returned.
    pub fn begin_restore_from_custom_storage(
        self,
        options: RestoreOptions,
    ) -> Result<StorageOperation> {
        self.begin_storage_operation(Some(options))
    }

    fn begin_storage_operation(self, restore: Option<RestoreOptions>) -> Result<StorageOperation> {
        ensure_supported()?;
        if u32::try_from(self.pid) == Ok(process::id()) {
            return Err(Error::Cuda {
                code: Status::NotSupported,
            });
        }

        // SAFETY: `cuInit` takes only its reserved zero flags.
        unsafe {
            try_ffi!(driver::cuInit(0))?;
        }

        let contexts = (0..Device::count()?)
            .map(|id| Context::create_for_device(Device::new(id)))
            .collect::<Result<Vec<_>>>()?;
        let restoring = restore.is_some();
        let mut info = ptr::null_mut();
        if let Some(mut options) = restore {
            let mut args = options.raw_args()?;
            args.customStorageInfo_out = &raw mut info;
            // SAFETY: `args` points at the counted remapping pairs in `options` and at the local `info` output, which outlive the call.
            unsafe {
                try_ffi!(driver::cuCheckpointProcessRestore(self.pid, &raw mut args))?;
            }
        } else {
            let mut args = driver::CUcheckpointCheckpointArgs {
                customStorageInfo_out: &raw mut info,
                ..Default::default()
            };
            // SAFETY: `args` points at the local `info` output, which outlives the call.
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

        // SAFETY: CUDA returned a non-null storage-info pointer that stays valid while the operation is pending, and it is copied out below.
        let info = unsafe { &*info };
        let regions = if info.deviceCount == 0 {
            Vec::new()
        } else {
            if info.perDeviceData.is_null() {
                return Err(Error::NullHandle);
            }

            let count = to_usize(info.deviceCount, "checkpoint device count")?;
            // SAFETY: CUDA provides `deviceCount` non-null per-device entries while the operation is pending, and they are copied out here.
            unsafe { slice::from_raw_parts(info.perDeviceData, count).to_vec() }
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
        checkpoint_fails(missing_process.checkpoint(CheckpointOptions::new()));
        checkpoint_fails(missing_process.restore(RestoreOptions::new()));
        checkpoint_fails(missing_process.unlock(UnlockOptions::new()));
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
