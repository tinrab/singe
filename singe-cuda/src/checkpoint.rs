use std::{
    fmt::{self, Display, Formatter},
    os::raw::c_int,
    ptr, thread,
    time::{Duration, Instant},
};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_conversion;
use singe_cuda_sys::driver;

use crate::{
    device::{Device, Uuid},
    error::{Error, Result, Status},
    try_ffi,
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
    /// Unknown process state returned by a newer CUDA driver version.
    #[num_enum(catch_all)]
    Unknown(u32),
}

impl_enum_conversion!(driver::CUprocessState, ProcessState);

impl Display for ProcessState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
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
    pub const fn new() -> Self {
        Self { timeout: None }
    }

    /// Sets the maximum time CUDA should spend attempting to lock the process.
    ///
    /// A missing timeout passes `0` to CUDA, which means no timeout.
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
    pub const fn new(old_uuid: Uuid, new_uuid: Uuid) -> Self {
        Self { old_uuid, new_uuid }
    }

    /// Creates a GPU pair from two CUDA devices by reading their UUIDs.
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
    pub fn with_gpu_pair(mut self, pair: GpuPair) -> Self {
        self.gpu_pairs.push(pair.to_raw());
        self
    }

    /// Adds a GPU remapping pair from CUDA devices.
    pub fn with_device_pair(mut self, old: Device, new: Device) -> Result<Self> {
        self.push_device_pair(old, new)?;
        Ok(self)
    }

    /// Adds a single GPU remapping pair in place.
    pub fn push_gpu_pair(&mut self, pair: GpuPair) {
        self.gpu_pairs.push(pair.to_raw());
    }

    /// Adds a GPU remapping pair from CUDA devices in place.
    pub fn push_device_pair(&mut self, old: Device, new: Device) -> Result<()> {
        let pair = GpuPair::from_devices(old, new)?;
        self.push_gpu_pair(pair);
        Ok(())
    }

    /// Creates restore options from checkpointed/restored GPU device pairs.
    pub fn with_device_pairs(gpu_pairs: impl AsRef<[(Device, Device)]>) -> Result<Self> {
        let mut options = Self::new();
        for &(old, new) in gpu_pairs.as_ref() {
            options.push_device_pair(old, new)?;
        }
        Ok(options)
    }

    fn into_raw(mut self) -> Result<driver::CUcheckpointRestoreArgs> {
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
    pub const fn new(pid: ProcessId) -> Self {
        Self { pid }
    }

    /// Creates a CUDA checkpoint target from an operating-system process ID.
    pub const fn from_pid(pid: ProcessId) -> Self {
        Self::new(pid)
    }

    /// Creates a CUDA checkpoint target for the current process.
    pub fn current() -> Self {
        Self::new(std::process::id() as ProcessId)
    }

    /// Returns the operating-system process ID controlled by this value.
    pub const fn pid(self) -> ProcessId {
        self.pid
    }

    /// Returns the current CUDA checkpoint state of the process.
    pub fn state(self) -> Result<ProcessState> {
        let mut state = driver::CUprocessState::CU_PROCESS_STATE_RUNNING;
        unsafe {
            try_ffi!(driver::cuCheckpointProcessGetState(
                self.pid,
                &raw mut state,
            ))?;
        }
        ProcessState::try_from(state as u32).map_err(|_| Error::InvalidValue)
    }

    /// Returns `true` when this process currently has [`ProcessState::Running`].
    pub fn is_running(self) -> bool {
        self.state()
            .is_ok_and(|state| state == ProcessState::Running)
    }

    /// Returns `true` when this process currently has [`ProcessState::Locked`].
    pub fn is_locked(self) -> bool {
        self.state()
            .is_ok_and(|state| state == ProcessState::Locked)
    }

    /// Returns `true` when this process currently has [`ProcessState::Checkpointed`].
    pub fn is_checkpointed(self) -> bool {
        self.state()
            .is_ok_and(|state| state == ProcessState::Checkpointed)
    }

    /// Blocks until the process reaches `expected`, or returns a timeout error.
    pub fn wait_for_state(self, expected: ProcessState, timeout: Duration) -> Result<ProcessState> {
        let end = Instant::now()
            .checked_add(timeout)
            .ok_or(Error::InvalidValue)?;
        const POLL_INTERVAL: Duration = Duration::from_millis(25);

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
    pub fn restore_thread_id(self) -> Result<i32> {
        let mut thread_id = 0;
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
    pub fn try_lock(self, options: LockOptions) -> Result<LockResult> {
        let mut args = options.to_raw()?;
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
    pub fn checkpoint(self) -> Result<()> {
        self.checkpoint_with_options(CheckpointOptions::new())
    }

    /// Moves the locked process's GPU memory into host memory managed by the driver.
    ///
    /// This variant accepts explicit checkpoint options.
    pub fn checkpoint_with_options(self, options: CheckpointOptions) -> Result<()> {
        let mut args = options.to_raw();
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
    pub fn suspend(self, options: LockOptions) -> Result<()> {
        self.lock(options)?;
        self.checkpoint()
    }

    /// Toggles the process between running and checkpointed states.
    ///
    /// - From `Running`, performs [`CheckpointProcess::suspend`].
    /// - From `Checkpointed`, performs [`CheckpointProcess::resume`].
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
    pub fn restore_with_options(self, options: RestoreOptions) -> Result<()> {
        let mut args = options.into_raw()?;
        unsafe { try_ffi!(driver::cuCheckpointProcessRestore(self.pid, &raw mut args)) }
    }

    /// Restores a checkpointed process, optionally remapping checkpointed GPUs.
    ///
    /// If `gpu_pairs` is not empty, CUDA requires it to contain every
    /// checkpointed GPU.
    ///
    /// On success the process enters [`ProcessState::Locked`].
    pub fn restore(self, gpu_pairs: impl AsRef<[GpuPair]>) -> Result<()> {
        self.restore_with_options(gpu_pairs.as_ref().into())
    }

    /// Restores and unlocks a checkpointed CUDA process.
    ///
    /// On success the process enters [`ProcessState::Running`].
    pub fn resume(self, gpu_pairs: impl AsRef<[GpuPair]>) -> Result<()> {
        self.restore(gpu_pairs)?;
        self.unlock()
    }

    /// Unlocks a locked CUDA process so it can resume CUDA API calls.
    ///
    /// On success the process enters [`ProcessState::Running`].
    pub fn unlock(self) -> Result<()> {
        self.unlock_with_options(UnlockOptions::new())
    }

    /// Unlocks a locked CUDA process so it can resume CUDA API calls with options.
    ///
    /// On success the process enters [`ProcessState::Running`].
    pub fn unlock_with_options(self, options: UnlockOptions) -> Result<()> {
        let mut args = options.to_raw();
        unsafe { try_ffi!(driver::cuCheckpointProcessUnlock(self.pid, &raw mut args)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
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
        checkpoint_fails(missing_process.restore(&[]));
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
            Error::Cuda { code, .. }
                if matches!(
                    code,
                    Status::InvalidValue
                        | Status::NotInitialized
                        | Status::NotSupported
                        | Status::IllegalState
                        | Status::OperatingSystem
                ) => {}
            error => panic!("{error:?}"),
        }
    }
}
