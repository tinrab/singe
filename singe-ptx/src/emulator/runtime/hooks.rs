//! Test-only observation points that runtime tests install on [`Shared`](super::Shared).

use std::sync::Mutex;

use crate::emulator::engine::{Engine, LaunchTraceEntry, TraceEntry};

pub(crate) type LaunchHook = Box<dyn FnMut(&Engine) + Send>;

pub(crate) type InitializationHook = Box<dyn FnMut(u64) + Send>;

pub(crate) type TransferHook = Box<dyn FnMut(usize, usize) + Send>;

#[derive(Default)]
pub(crate) struct Hooks {
    /// Runs before each scheduler turn of a launch.
    pub(crate) launch: Mutex<Option<LaunchHook>>,
    /// Runs after each grid-construction turn, with the launch identifier.
    pub(crate) initialization: Mutex<Option<InitializationHook>>,
    /// Runs at each transfer checkpoint, with the completed and total byte counts.
    pub(crate) transfer: Mutex<Option<TransferHook>>,
    /// The latest launch's instruction trace.
    pub(crate) trace: Mutex<Vec<TraceEntry>>,
    /// Instructions in scheduler order across interleaved launches, bounded to 4096.
    pub(crate) scheduler_trace: Mutex<Vec<LaunchTraceEntry>>,
}
