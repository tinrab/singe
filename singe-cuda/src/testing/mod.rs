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

pub fn device_lock(device_id: i32) -> Result<DeviceLock> {
    DeviceLock::acquire(device_id)
}

pub fn bootstrap() -> Result<(DeviceLock, Arc<Context>)> {
    bootstrap_for_device(0)
}

pub fn bootstrap_for_device(device_id: i32) -> Result<(DeviceLock, Arc<Context>)> {
    let lock = device_lock(device_id)?;
    let ctx = Context::create_for_device(Device::new(device_id))?;
    Ok((lock, ctx))
}

pub fn is_stub_library(error: &Error) -> bool {
    matches!(
        error,
        Error::Cuda { code, .. } if *code == Status::StubLibrary
    )
}

fn lock_path(device_id: i32) -> PathBuf {
    env::var_os("SINGE_CUDA_TEST_LOCK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir)
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
