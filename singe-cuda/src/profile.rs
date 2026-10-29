//! CUDA profiler control helpers.

use singe_cuda_sys::profiler;

use crate::{error::Result, try_ffi};

/// A guard that starts profiler collection and attempts to stop it when dropped.
///
/// Collection is controlled for CUDA's current context. The guard tracks its own stop obligation; it does not isolate profiling from other guards or native profiler calls.
#[derive(Debug)]
pub struct Profiler {
    active: bool,
}

impl Profiler {
    /// Starts profiler collection and returns a guard that stops it on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot start profiler collection for the current context.
    pub fn create() -> Result<Self> {
        start()?;
        Ok(Self { active: true })
    }

    /// Returns whether this guard still owes a profiler stop call.
    ///
    /// This does not query the active profiling tool or observe profiler calls made elsewhere.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Stops profiler collection before the guard is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot stop profiler collection. On failure, dropping the consumed guard attempts the stop call again; that cleanup result is discarded.
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

/// Enables collection by the active profiling tool for the current CUDA context.
///
/// Has no effect if profiling is already enabled.
///
/// # Errors
///
/// Returns an error if CUDA cannot enable profiler collection.
pub fn start() -> Result<()> {
    // SAFETY: the profiler control functions take no arguments and only affect profiling state.
    unsafe {
        try_ffi!(profiler::cuProfilerStart())?;
    }
    Ok(())
}

/// Disables collection by the active profiling tool for the current CUDA context.
///
/// Has no effect if profiling is already disabled.
///
/// # Errors
///
/// Returns an error if CUDA cannot disable profiler collection.
pub fn stop() -> Result<()> {
    // SAFETY: the profiler control functions take no arguments and only affect profiling state.
    unsafe {
        try_ffi!(profiler::cuProfilerStop())?;
    }
    Ok(())
}
