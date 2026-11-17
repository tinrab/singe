//! Device locks and context setup for CUDA tests.

use std::{
    env,
    fmt::Write as _,
    fs::{File, OpenOptions},
    path::PathBuf,
    sync::Arc,
};

use singe_cuda_sys::driver;

use crate::{
    context::Context,
    device::{Device, Uuid},
    error::{Error, Result, Status},
    try_ffi,
};

/// An exclusive lock on one CUDA device, held by cooperating test processes.
///
/// The lock is an advisory `flock` on a persistent file named by the device's UUID, so two ordinals that `CUDA_VISIBLE_DEVICES` maps to the same device share one lock.
/// The lock file stays in place; the operating system releases the lock when the guard drops or the process exits.
/// The lock directory is selected by `SINGE_CUDA_TEST_LOCK_DIR`, falling back to the operating system's temporary directory.
/// The lock is not reentrant: acquiring it again for the same device while holding it blocks forever.
#[derive(Debug)]
pub struct DeviceLock {
    _file: File,
}

impl DeviceLock {
    fn acquire(device_id: i32) -> Result<Self> {
        let uuid = device_uuid(device_id)?;
        let name = uuid
            .bytes
            .iter()
            .fold(String::from("singe-cuda-device-"), |mut name, byte| {
                let _ = write!(name, "{byte:02x}");
                name
            });
        let path = env::var_os("SINGE_CUDA_TEST_LOCK_DIR")
            .map_or_else(env::temp_dir, PathBuf::from)
            .join(format!("{name}.lock"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        file.lock()?;
        Ok(Self { _file: file })
    }
}

/// Waits for and acquires the test lock for `device`.
///
/// Initializes the CUDA driver to read the device's UUID, without creating a context.
/// An existing lock held by another process is waited on without a timeout.
///
/// # Errors
///
/// Returns an error if the driver cannot initialize or report the device's UUID, or if the lock file cannot be opened or locked.
pub fn device_lock(device: Device) -> Result<DeviceLock> {
    DeviceLock::acquire(device.id())
}

/// A locked CUDA device and the context created for it.
///
/// The context drops before the device lock is released.
#[derive(Debug)]
pub struct TestDevice {
    context: Arc<Context>,
    _lock: DeviceLock,
}

impl TestDevice {
    /// Returns the context created for the locked device.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
}

/// Acquires the test lock for device zero and creates a CUDA context for it.
///
/// # Errors
///
/// Returns an error if the device lock cannot be acquired or the CUDA context cannot be created.
pub fn bootstrap() -> Result<TestDevice> {
    bootstrap_for_device(Device::new(0))
}

/// Acquires the test lock for `device` and retains its primary context.
///
/// The context becomes current on this thread.
///
/// # Errors
///
/// Returns an error if the device lock cannot be acquired or the CUDA context cannot be created.
/// The lock is released if context creation fails.
pub fn bootstrap_for_device(device: Device) -> Result<TestDevice> {
    let lock = device_lock(device)?;
    let context = Context::create_for_device(device)?;
    Ok(TestDevice {
        context,
        _lock: lock,
    })
}

/// Returns whether `error` says there is no CUDA driver to test with: a stub library, or none that loads.
#[must_use]
pub fn is_unavailable(error: &Error) -> bool {
    matches!(
        error,
        Error::Cuda { code, .. } if *code == Status::StubLibrary
    ) || matches!(error, Error::Load(_))
}

fn device_uuid(device_id: i32) -> Result<Uuid> {
    let mut device: driver::CUdevice = 0;
    let mut uuid = driver::CUuuid::default();
    // SAFETY: `device` and `uuid` are locals the driver writes through exclusive pointers, and `cuDeviceGetUuid_v2` receives the handle `cuDeviceGet` returned after `cuInit` succeeded.
    unsafe {
        try_ffi!(driver::cuInit(0))?;
        try_ffi!(driver::cuDeviceGet(&raw mut device, device_id))?;
        try_ffi!(driver::cuDeviceGetUuid_v2(&raw mut uuid, device))?;
    }
    Ok(uuid.into())
}
