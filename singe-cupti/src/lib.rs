//! Safe Rust wrappers for NVIDIA CUPTI profiling, activity tracing, callbacks, events, and metrics.
//!
//! # Thread safety
//!
//! Wrappers around raw CUPTI handles, such as [`callback::Subscriber`], are neither `Send`
//! nor `Sync`. CUPTI requires each subscriber to serialize access to its callback state,
//! so these wrappers stay on the thread that created them. Callbacks themselves must be
//! `Send` because CUPTI invokes them on whichever thread makes the traced CUDA call.

/// Activity tracing APIs and decoded activity record types.
pub mod activity;

/// Callback subscriber APIs and typed callback payload views.
pub mod callback;

/// CUDA driver and runtime callback identifiers.
pub mod callback_id;

/// CUDA context helpers and process-wide CUPTI utility APIs.
pub mod context;

/// Device support and virtualization queries.
pub mod device;

/// CUPTI error and status types.
pub mod error;

/// Event and metric collection APIs.
pub mod event;

/// Host metric configuration and modern range profiling.
pub mod profiler;

/// Stream identifier queries.
pub mod stream;

/// Shared CUPTI identifier, enum, flag, and configuration types.
pub mod types;

#[cfg(feature = "testing")]
/// Test support for CUPTI integration tests.
pub mod testing;

pub(crate) mod utility;

use singe_core::LibraryVersion;
use singe_cupti_sys as sys;

use crate::error::Result;

/// CUPTI API version compiled into this crate.
pub const API_VERSION: LibraryVersion = LibraryVersion::new(
    (sys::CUPTI_API_VERSION / 10000) as u64,
    ((sys::CUPTI_API_VERSION / 100) % 100) as u64,
    (sys::CUPTI_API_VERSION % 100) as u64,
);

/// Returns the CUPTI API version reported by the loaded runtime library.
///
/// # Errors
///
/// Returns an error if the loaded CUPTI runtime cannot report its API version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    // SAFETY: `cuptiGetVersion` receives valid output locations.
    unsafe {
        try_ffi!(sys::cuptiGetVersion(&raw mut version))?;
    }
    let version = u64::from(version);
    Ok(LibraryVersion::new(
        version / 10000,
        (version % 10000) / 100,
        version % 100,
    ))
}
