//! Safe Rust wrappers for NVIDIA CUPTI profiling, activity tracing, callbacks, events, and metrics.

/// Activity tracing APIs and decoded activity record types.
pub mod activity;
/// Callback subscriber APIs and typed callback payload views.
pub mod callback;
/// CUDA context helpers and process-wide CUPTI utility APIs.
pub mod context;
/// Device support and virtualization queries.
pub mod device;
/// CUPTI error and status types.
pub mod error;
/// Event and metric collection APIs.
pub mod event;
/// Stream identifier queries.
pub mod stream;
/// Shared CUPTI identifier, enum, flag, and configuration types.
pub mod types;

#[cfg(feature = "testing")]
/// Test support for CUPTI integration tests.
pub mod testing;

pub(crate) mod utility;

use singe_cupti_sys as sys;

use singe_core::LibraryVersion;

use crate::error::Result;

/// CUPTI API version compiled into this crate.
pub const API_VERSION: LibraryVersion = LibraryVersion::new(
    (sys::CUPTI_API_VERSION / 10000) as u64,
    ((sys::CUPTI_API_VERSION / 100) % 100) as u64,
    (sys::CUPTI_API_VERSION % 100) as u64,
);

/// Returns the CUPTI API version reported by the loaded runtime library.
pub fn linked_version() -> Result<LibraryVersion> {
    let mut version = 0;
    unsafe {
        try_ffi!(sys::cuptiGetVersion(&raw mut version))?;
    }
    let version = version as u64;
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}
