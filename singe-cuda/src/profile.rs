//! CUDA profiler control helpers.

use singe_cuda_sys::profiler;

use crate::{error::Result, try_ffi};

#[derive(Debug)]
pub struct Profiler {
    active: bool,
}

impl Profiler {
    /// Starts profiler collection and returns a guard that stops it on drop.
    pub fn create() -> Result<Self> {
        start()?;
        Ok(Self { active: true })
    }

    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Stops profiler collection before the guard is dropped.
    pub fn stop(mut self) -> Result<()> {
        if self.active {
            stop()?;
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        if self.active {
            let _ = stop();
            self.active = false;
        }
    }
}

pub fn start() -> Result<()> {
    unsafe {
        try_ffi!(profiler::cuProfilerStart())?;
    }
    Ok(())
}

pub fn stop() -> Result<()> {
    unsafe {
        try_ffi!(profiler::cuProfilerStop())?;
    }
    Ok(())
}
