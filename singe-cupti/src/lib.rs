//! Safe Rust wrappers for NVIDIA CUPTI, the CUDA Profiling Tools Interface, for tracing and profiling CUDA applications.
//!
//! This crate provides:
//!
//! - **Callbacks:** [`Subscriber`](callback::Subscriber) runs a closure on CUDA driver and runtime API calls and other CUPTI callback points, with decoded payloads.
//! - **Activity tracing:** records of kernels, memory copies, API calls, and other CUDA activity, collected in owned or borrowed buffers.
//! - **Hardware metrics:** metric configuration and range profiling through [`ProfilerHost`](profiler::ProfilerHost) and [`RangeProfiler`](profiler::RangeProfiler).
//! - **Identifiers:** context, device, stream, and graph IDs as CUPTI reports them, and the driver and runtime callback IDs.
//!
//! # Example
//!
//! Print the CUPTI version, then count the CUDA driver API callbacks made while a context and a stream are created:
//!
//! ```rust,no_run
//! use std::sync::{
//!     Arc,
//!     atomic::{AtomicUsize, Ordering},
//! };
//!
//! use singe_cuda::{context::Context, stream::StreamOptions};
//! use singe_cupti::{
//!     callback::{Subscriber, SubscriberOptions},
//!     types::CallbackDomain,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     println!("CUPTI {}", singe_cupti::version()?);
//!
//!     let callbacks = Arc::new(AtomicUsize::new(0));
//!     let counter = Arc::clone(&callbacks);
//!     let subscriber = Subscriber::create(
//!         move |_domain, _callback_id, _data| {
//!             counter.fetch_add(1, Ordering::Relaxed);
//!         },
//!         &SubscriberOptions::default(),
//!     )?;
//!     subscriber.set_domain_enabled(CallbackDomain::DriverApi, true)?;
//!
//!     let context = Context::create()?;
//!     let _stream = context.create_stream(StreamOptions::default())?;
//!     subscriber.unsubscribe()?;
//!
//!     println!("driver API callbacks: {}", callbacks.load(Ordering::Relaxed));
//!     Ok(())
//! }
//! ```
//!
//! A new [`Subscriber`](callback::Subscriber) has every callback disabled until a domain or callback ID is enabled.
//! CUPTI calls the closure on the thread that makes the traced call, once on entry and once on exit of each API function.
//! After [`unsubscribe`](callback::Subscriber::unsubscribe) returns, CUPTI makes no more calls to the closure; dropping the [`Subscriber`](callback::Subscriber) also unsubscribes.
//! CUPTI allows one subscriber per process unless every subscriber sets [`SubscriberOptions::allow_multiple_subscribers`](callback::SubscriberOptions::allow_multiple_subscribers), so an attached profiler or debugger can make [`Subscriber::create`](callback::Subscriber::create) fail.
//!
//! Hardware metrics need a GPU that supports them, and the NVIDIA driver restricts performance counters to administrators unless it is configured to allow every user.
//!
//! # Thread safety
//!
//! Wrappers around raw CUPTI handles, such as [`callback::Subscriber`], are neither [`Send`] nor [`Sync`].
//! CUPTI requires each subscriber to serialize access to its callback state, so these wrappers stay on the thread that created them.
//! Callbacks must be `Send + Sync`, because CUPTI invokes them on whichever thread makes the traced CUDA call, concurrently and re-entrantly.

#[cfg(not(feature = "cupti_13_4"))]
compile_error!("singe-cupti requires a version feature, such as `cupti_13_4`");

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

/// Host metric configuration and modern range profiling.
pub mod profiler;

/// Stream identifier queries and the default-stream setting they depend on.
pub mod stream;

/// Shared CUPTI identifier, enum, flag, and configuration types.
pub mod types;

#[cfg(feature = "testing")]
/// Test support for CUPTI integration tests.
pub mod testing;

pub(crate) mod utility;

pub use singe_core::LibraryVersion;
use singe_core::PackedVersion;
pub use singe_cupti_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// CUPTI API version compiled into this crate.
pub const API_VERSION: LibraryVersion = LibraryVersion::new(
    sys::CUPTI_API_VERSION / 10000,
    (sys::CUPTI_API_VERSION / 100) % 100,
    sys::CUPTI_API_VERSION % 100,
);

/// Returns the CUPTI API version reported by the loaded runtime library.
///
/// # Errors
///
/// Returns an error if the loaded CUPTI runtime cannot report its API version.
pub fn version() -> Result<LibraryVersion> {
    let mut version = 0;
    // SAFETY: `version` is a local output for the call.
    unsafe {
        try_ffi!(sys::cuptiGetVersion(&raw mut version))?;
    }

    LibraryVersion::from_packed(version, PackedVersion::Major10000Minor100)
        .ok_or_else(|| Error::out_of_range("cupti version"))
}
