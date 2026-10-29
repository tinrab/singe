use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process,
    sync::Arc,
    thread,
    time::Duration,
};

use crate::{
    context::Context,
    device::Device,
    error::{Error, Result, Status},
};

/// A file-backed guard serializing cooperating tests for one device ID.
///
/// Dropping the guard removes the lock file when possible. The lock directory is selected by `SINGE_CUDA_TEST_LOCK_DIR`, falling back to the operating system's temporary directory.
#[derive(Debug)]
pub struct DeviceLock {
    _file: File,
    path: PathBuf,
}

impl Drop for DeviceLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl DeviceLock {
    fn acquire(device_id: i32) -> Result<Self> {
        let path = lock_path(device_id);

        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    writeln!(file, "{}", process::id())?;
                    return Ok(Self { _file: file, path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    remove_stale_lock(&path)?;
                    thread::sleep(Duration::from_millis(25));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

/// Waits for and acquires the test lock for `device_id`.
///
/// Polls existing locks every 25 milliseconds. On Linux, a lock naming a process that has exited is removed. This function does not validate the device ID or initialize CUDA.
///
/// # Errors
///
/// Returns an error if the lock file cannot be created or written, or a stale lock cannot be removed. An existing lock held by another process is waited on without a timeout.
pub fn device_lock(device_id: i32) -> Result<DeviceLock> {
    DeviceLock::acquire(device_id)
}

/// Acquires the test lock for device zero and creates a CUDA context for it.
///
/// Keep the returned lock alive for the test's device work.
///
/// # Errors
///
/// Propagates errors from acquiring the device lock or creating the CUDA context.
pub fn bootstrap() -> Result<(DeviceLock, Arc<Context>)> {
    bootstrap_for_device(0)
}

/// Acquires the test lock for `device_id` and creates a CUDA context for that device.
///
/// The new context becomes current on this thread. Keep the returned lock alive for the test's device work.
///
/// # Errors
///
/// Propagates errors from acquiring the device lock or creating the CUDA context. The lock is released if context creation fails.
pub fn bootstrap_for_device(device_id: i32) -> Result<(DeviceLock, Arc<Context>)> {
    let lock = device_lock(device_id)?;
    let context = Context::create_for_device(Device::new(device_id))?;
    Ok((lock, context))
}

/// Returns whether `error` reports that a CUDA stub library was loaded.
#[must_use]
pub fn is_stub_library(error: &Error) -> bool {
    matches!(
        error,
        Error::Cuda { code, .. } if *code == Status::StubLibrary
    )
}

fn lock_path(device_id: i32) -> PathBuf {
    env::var_os("SINGE_CUDA_TEST_LOCK_DIR")
        .map_or_else(env::temp_dir, PathBuf::from)
        .join(format!("singe-cuda-device-{device_id}.lock"))
}

fn remove_stale_lock(path: &Path) -> Result<()> {
    let mut pid = String::new();
    if File::open(path)
        .and_then(|mut file| file.read_to_string(&mut pid))
        .is_err()
    {
        return Ok(());
    }

    let Ok(pid) = pid.trim().parse::<u32>() else {
        return Ok(());
    };

    if process_is_running(pid) {
        return Ok(());
    }

    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(target_os = "linux")]
fn process_is_running(pid: u32) -> bool {
    PathBuf::from(format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn process_is_running(_pid: u32) -> bool {
    true
}
