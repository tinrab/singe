//! Safe NCCL communicator, collective, local multi-rank group, and buffer wrappers.
//!
//! This crate provides Rust ownership around NCCL communicators, unique IDs,
//! grouped calls, local multi-rank orchestration, collective operations, device
//! buffers, and NCCL memory helpers.

pub mod buffer;
pub mod communicator;
pub mod error;
pub mod group;
pub mod local;
pub mod memory;
pub mod types;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

use singe_core::string_from_c_ptr;
use singe_nccl_sys as sys;

use crate::{
    error::Result,
    group::Group,
    types::{SimulationInfo, Status, UniqueId},
};

/// Returns the version number of the currently linked NCCL library.
///
/// The returned integer uses NCCL's packed version encoding.
///
/// # Errors
///
/// Returns an error if NCCL cannot report its version.
pub fn version() -> Result<i32> {
    let mut value = 0;
    unsafe {
        try_ffi!(sys::ncclGetVersion(&raw mut value))?;
    }
    Ok(value)
}

/// Returns a human-readable string for `status`.
pub fn error_string(status: Status) -> String {
    match sys::ncclResult_t::try_from(status) {
        Ok(raw) => unsafe { string_from_c_ptr(sys::ncclGetErrorString(raw)) },
        Err(_) => String::from("unknown nccl error"),
    }
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
    unsafe {
        try_ffi!(sys::ncclGetUniqueId(&raw mut value))?;
    }
    Ok(value.into())
}

/// Start a group call.
///
/// Subsequent NCCL calls do not block on inter-CPU synchronization until
/// [`group_end`].
///
/// # Errors
///
/// Returns an error if NCCL rejects the group start.
pub fn group_start() -> Result<()> {
    unsafe { try_ffi!(sys::ncclGroupStart()) }
}

/// End a group call.
///
/// Returns when all operations since [`group_start`] have been processed.
/// This means the communication primitives have been enqueued to the provided streams, but are not necessarily complete.
///
/// When used with the [`Communicator::create_rank`](crate::communicator::Communicator::create_rank) call, the [`group_end`] call waits for all communicators to be initialized.
///
/// # Errors
///
/// Returns an error if any grouped NCCL operation fails or if NCCL rejects the
/// group end.
pub fn group_end() -> Result<()> {
    unsafe { try_ffi!(sys::ncclGroupEnd()) }
}

pub fn group() -> Result<Group> {
    Group::start()
}

pub fn with_group<R>(f: impl FnOnce() -> Result<R>) -> Result<R> {
    group::with_group(f)
}

pub fn group_simulation() -> Result<SimulationInfo> {
    group()?.simulate_end()
}

#[macro_export]
macro_rules! try_ffi {
    ($expr:expr) => {{
        let status = { $expr };
        if status != singe_nccl_sys::ncclResult_t::ncclSuccess {
            Err($crate::error::Error::from(status))
        } else {
            Ok(())
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        assert_ne!(version()?, 0);
        Ok(())
    }
}
