//! Public emulator handles, command admission, and the command queue served by the
//! worker thread.
//!
//! - `options`: emulator limits, launch shapes, and execution reports.
//! - `scalar`: the sealed host scalar types accepted by transfers and arguments.
//! - `completion`: single-consumer submission results and shutdown observation.
//! - `device`: emulators, streams, and events.
//! - `buffer`: buffers, buffer views, and kernel arguments.
//! - `module`: loaded modules and kernel launches.
//! - `submit`: command admission and result publication.
//! - `worker`: the worker loop, module loading, and grid execution.
//! - `debug`: breakpoint control that bypasses user queues.
//! - `hooks`: test-only observation points.

mod buffer;
mod completion;
mod debug;
mod device;
#[cfg(test)]
mod hooks;
mod module;
mod options;
mod scalar;
mod submit;
mod worker;

#[cfg(test)]
use std::sync::mpsc::Sender;
use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Formatter},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak,
        atomic::{AtomicBool, Ordering},
    },
    task::Waker,
};

#[cfg(test)]
use crate::emulator::runtime::hooks::Hooks;
pub use crate::emulator::runtime::{
    buffer::{Buffer, BufferView, KernelArg},
    completion::{Shutdown, Submission},
    debug::{Breakpoint, BreakpointId},
    device::{Emulator, Event, Stream},
    module::{Kernel, Module},
    options::{EmulatorOptions, ExecutionReport, LaunchConfig, ShutdownMode},
    scalar::PtxScalar,
};
use crate::emulator::{
    budget::{Budget, ChargedQueue},
    execution_error::{ExecutionError, SubmitError},
    runtime::{
        completion::{Completion, Pending},
        submit::{Producer, RetainedFailure},
        worker::Worker,
    },
};

pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Running,
    Faulted,
    Draining,
    Cancelling,
    Closed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CommandKind {
    Immediate,
    Event,
    Launch,
    Dependency,
    Transfer,
}

pub type CommandBody = Box<dyn FnMut(&mut Worker, Option<ExecutionError>) -> bool + Send>;

#[derive(Debug, Default)]
pub struct CommandControl {
    suspended: AtomicBool,
    inspection_pending: AtomicBool,
}

impl CommandControl {
    fn is_suspended(&self) -> bool {
        self.suspended.load(Ordering::Acquire) && !self.inspection_pending.load(Ordering::Acquire)
    }
}

pub struct Command {
    stream: u64,
    kind: CommandKind,
    poll: CommandBody,
    control: Arc<CommandControl>,
}

pub struct Admission {
    debug: debug::State,
    phase: Phase,
    queue: ChargedQueue<Command>,
    reservations: usize,
    pending_dependencies: usize,
    next_stream: u64,
    next_launch: u64,
    staged: usize,
    reclaim: Vec<u64>,
    shutdown_result: Option<Result<(), RetainedFailure>>,
    shutdown_wakers: BTreeMap<usize, Waker>,
}

impl Admission {
    /// Returns the next launch identifier.
    fn allocate_launch_id(&mut self) -> Result<u64, SubmitError> {
        let id = self.next_launch;
        self.next_launch = id
            .checked_add(1)
            .ok_or_else(|| SubmitError::InvalidArgument("launch identifiers exhausted".into()))?;
        Ok(id)
    }
}

impl Debug for Admission {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Admission").finish_non_exhaustive()
    }
}

pub struct Shared {
    execution_budget: Arc<Budget>,
    #[cfg(test)]
    hooks: Hooks,
    admission: Mutex<Admission>,
    changed: Condvar,
    cancel: AtomicBool,
    options: EmulatorOptions,
    // Declared last so test observers see every other shared field released first.
    #[cfg(test)]
    closed_notify: Mutex<Option<Sender<()>>>,
}

impl Debug for Shared {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared").finish_non_exhaustive()
    }
}

pub struct Client {
    shared: Arc<Shared>,
}

impl Debug for Client {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let mode = {
            let mut state = lock(&self.shared.admission);
            state.debug.orphaned = true;
            if state.debug.has_paused_launches() {
                ShutdownMode::Cancel
            } else {
                ShutdownMode::Drain
            }
        };
        self.shared.shutdown(mode);
    }
}

impl Shared {
    fn shutdown(&self, mode: ShutdownMode) {
        let mut state = lock(&self.admission);
        if state.phase == Phase::Closed {
            return;
        }
        if mode == ShutdownMode::Cancel {
            self.cancel.store(true, Ordering::Release);
            state.phase = Phase::Cancelling;
        } else if matches!(state.phase, Phase::Running | Phase::Faulted) {
            state.phase = Phase::Draining;
        }
        drop(state);
        self.changed.notify_all();
    }
}

pub struct QueueReservation {
    shared: Arc<Shared>,
    active: bool,
}

impl Drop for QueueReservation {
    fn drop(&mut self) {
        if self.active {
            lock(&self.shared.admission).reservations -= 1;
        }
    }
}

pub fn admission(phase: Phase, diagnostic: bool) -> Result<(), SubmitError> {
    match phase {
        Phase::Running => Ok(()),
        Phase::Faulted if diagnostic => Ok(()),
        Phase::Faulted => Err(SubmitError::Faulted),
        _ => Err(SubmitError::Closed),
    }
}

pub struct Staging {
    shared: Weak<Shared>,
    bytes: usize,
    launch_id: Option<u64>,
    control: Arc<CommandControl>,
}

impl Drop for Staging {
    fn drop(&mut self) {
        if self.bytes != 0
            && let Some(shared) = self.shared.upgrade()
        {
            lock(&shared.admission).staged -= self.bytes;
        }
    }
}

#[derive(Debug)]
pub struct Lease {
    shared: Weak<Shared>,
    addresses: Vec<u64>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.upgrade() {
            let mut state = lock(&shared.admission);
            if state.phase != Phase::Closed {
                state.reclaim.append(&mut self.addresses);
            }
            drop(state);
            shared.changed.notify_one();
        }
    }
}

#[cfg(test)]
mod tests;
