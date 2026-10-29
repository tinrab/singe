//! NCCL communicators, raw collectives, and completed local multi-rank operations.
//!
//! This crate provides Rust ownership around NCCL communicators, unique IDs,
//! grouped calls, local multi-rank orchestration, collective operations, device
//! buffers, and NCCL memory helpers.

/// Per-collective algorithm settings, vendor extensions, and launch-completion events.
pub mod collective;
/// Communicator creation, lifecycle, registration, and queued communication.
pub mod communicator;
pub mod encryption;
/// NCCL status codes, wrapper errors, and result types.
pub mod error;
/// Thread-local NCCL operation groups and simulated execution.
pub mod group;
/// Completed collectives across a group of ranks hosted in one process.
pub mod local;
pub mod memory;
pub mod parameter;
/// Typed NCCL identifiers, operations, flags, and communicator configuration.
pub mod types;
pub(crate) mod utility;

/// Serialized CUDA test setup for NCCL.
#[cfg(feature = "testing")]
pub mod testing;

use singe_core::LibraryVersion;
use singe_nccl_sys as sys;

use crate::{error::Result, types::UniqueId, utility::to_u64};

/// Returns the version number of the currently linked NCCL library.
///
/// The returned integer uses NCCL's packed version encoding.
///
/// # Errors
///
/// Returns an error if NCCL cannot report its version.
pub fn version() -> Result<LibraryVersion> {
    let mut value = 0;
    // SAFETY: `value` is a valid output location for the version query.
    unsafe {
        try_ffi!(sys::ncclGetVersion(&raw mut value))?;
    }
    // NCCL encodes versions as `major * 10000 + minor * 100 + patch` since 2.9.
    let value = to_u64(value, "nccl version")?;
    Ok(LibraryVersion::new(
        value / 10_000,
        (value % 10_000) / 100,
        value % 100,
    ))
}

/// Generates a [`UniqueId`] for [`Communicator::create_rank`](crate::communicator::Communicator::create_rank).
///
/// Call this once for a communicator clique and distribute the returned value to
/// all ranks before creating per-rank communicators.
///
/// # Errors
///
/// Returns an error if NCCL cannot generate a unique ID.
pub fn unique_id() -> Result<UniqueId> {
    let mut value = sys::ncclUniqueId::default();
    // SAFETY: `ncclGetUniqueId` receives valid output locations.
    unsafe {
        try_ffi!(sys::ncclGetUniqueId(&raw mut value))?;
    }
    Ok(value.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it_works() -> Result<()> {
        assert_ne!(version()?, 0);
        Ok(())
    }
}
