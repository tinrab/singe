use std::fmt::Debug;

use crate::emulator::{execution_error::SubmitError, target};

/// Limits for one emulator.
///
/// The storage limits count memory the emulator manages, not the whole process.
#[derive(Debug, Clone)]
pub struct EmulatorOptions {
    /// Maximum number of accepted commands that have not started. Defaults to 256.
    pub queued_commands: usize,
    /// Bytes available for allocations, including initialization tracking. Defaults to
    /// 1 GiB.
    pub device_storage_bytes: usize,
    /// Bytes available for command state, lane frames, shared variables, and fault
    /// reports. Defaults to 512 MiB.
    pub execution_storage_bytes: usize,
    /// Bytes available for upload copies and unconsumed download results. Defaults to
    /// 256 MiB.
    pub staging_storage_bytes: usize,
    /// Maximum instructions one launch issues, summed over all lanes and counting
    /// instructions whose predicate is false. Defaults to 100 million.
    pub launch_instructions: u64,
    /// Maximum call depth per lane, counting the entry function. Defaults to 64.
    pub call_depth: usize,
    /// Maximum number of launches executing at once. Must be nonzero; defaults to 32.
    pub active_grids: usize,
}

impl Default for EmulatorOptions {
    fn default() -> Self {
        Self {
            queued_commands: 256,
            device_storage_bytes: 1 << 30,
            execution_storage_bytes: 512 << 20,
            staging_storage_bytes: 256 << 20,
            launch_instructions: 100_000_000,
            call_depth: 64,
            active_grids: 32,
        }
    }
}

/// How [`Emulator::shutdown`](super::Emulator::shutdown) treats accepted commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownMode {
    /// Runs every accepted command to completion.
    Drain,
    /// Stops accepted commands at the next instruction or transfer-chunk boundary.
    Cancel,
}

/// Grid and block dimensions of a launch, and its dynamic shared memory per CTA.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) shared_bytes: usize,
}

impl LaunchConfig {
    /// Creates a launch configuration.
    ///
    /// `grid` counts CTAs, `block` counts threads per CTA, and `shared_bytes` sizes the
    /// `.extern .shared` array of each CTA.
    ///
    /// # Errors
    ///
    /// Returns [`SubmitError::InvalidArgument`] if a dimension is zero or exceeds the
    /// `sm_80` limits.
    pub fn create(
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: usize,
    ) -> Result<Self, SubmitError> {
        if !target::grid_fits(grid) {
            return Err(SubmitError::InvalidArgument("grid dimensions".into()));
        }
        if !target::block_fits(block) {
            return Err(SubmitError::InvalidArgument(
                "block dimensions exceed sm_80 limits".into(),
            ));
        }
        Ok(Self {
            grid,
            block,
            shared_bytes,
        })
    }

    /// Returns the number of CTAs along each grid axis.
    #[must_use]
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    /// Returns the number of threads along each CTA axis.
    #[must_use]
    pub fn block(&self) -> [u32; 3] {
        self.block
    }

    /// Returns the dynamic shared memory per CTA, in bytes.
    #[must_use]
    pub fn shared_bytes(&self) -> usize {
        self.shared_bytes
    }
}

/// Counts from a finished or faulted launch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionReport {
    /// Launch identifier, unique within its emulator.
    pub launch_id: u64,
    /// Stream the launch ran on.
    pub stream_id: u64,
    /// Number of instructions lanes executed, not counting those whose predicate was
    /// false.
    pub instructions: u64,
    /// Number of lanes that exited.
    pub completed_lanes: u64,
    /// Number of CTAs whose lanes all exited.
    pub completed_ctas: u64,
}
