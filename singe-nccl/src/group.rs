use crate::{error::Result, group_end, group_start, sys, try_ffi, types::SimulationInfo};

#[derive(Debug)]
pub struct Group {
    closed: bool,
}

impl Group {
    pub fn start() -> Result<Self> {
        group_start()?;
        Ok(Self { closed: false })
    }

    pub fn end(mut self) -> Result<()> {
        self.closed = true;
        group_end()
    }

    /// Simulates [`group_end`] and returns NCCL's estimated execution metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL cannot simulate the current group.
    pub fn simulate_end(mut self) -> Result<SimulationInfo> {
        let mut value = SimulationInfo::default().into_raw();
        self.closed = true;
        unsafe {
            try_ffi!(sys::ncclGroupSimulateEnd(&raw mut value))?;
        }
        Ok(value.into())
    }
}

pub fn with_group<R>(f: impl FnOnce() -> Result<R>) -> Result<R> {
    let group = Group::start()?;
    let result = f();
    let end_result = group.end();
    match (result, end_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(end_error)) => {
            #[cfg(debug_assertions)]
            eprintln!("failed to end nccl group after grouped operation error: {end_error}");
            Err(error)
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        if self.closed {
            return;
        }

        if let Err(err) = group_end() {
            #[cfg(debug_assertions)]
            eprintln!("failed to end nccl group: {err}");
        }
    }
}
