//! Test-only observation points that runtime tests install on [`Shared`](super::Shared).

use std::sync::Mutex;

use crate::emulator::engine::{Engine, LaunchTraceEntry, TraceEntry};

pub type LaunchHook = Box<dyn FnMut(&Engine) + Send>;

pub type InitializationHook = Box<dyn FnMut(u64) + Send>;

pub type TransferHook = Box<dyn FnMut(usize, usize) + Send>;

#[derive(Default)]
pub struct Hooks {
    /// Runs before each scheduler turn of a launch.
    pub launch: Mutex<Option<LaunchHook>>,
    /// Runs after each grid-construction turn, with the launch identifier.
    pub initialization: Mutex<Option<InitializationHook>>,
    /// Runs at each transfer checkpoint, with the completed and total byte counts.
    pub transfer: Mutex<Option<TransferHook>>,
    /// The latest launch's instruction trace.
    pub trace: Mutex<Vec<TraceEntry>>,
    /// Instructions in scheduler order across interleaved launches, bounded to 4096.
    pub scheduler_trace: Mutex<Vec<LaunchTraceEntry>>,
}
