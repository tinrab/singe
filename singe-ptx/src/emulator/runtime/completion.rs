use std::{
    fmt::{self, Debug, Formatter},
    future::Future,
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    sync::{Arc, Condvar, Mutex, PoisonError},
    task::{Context, Poll, Waker},
};

use crate::emulator::{
    budget::Reservation,
    execution_error::ExecutionError,
    runtime::{Client, Shared, lock},
};

/// A published result plus the storage that keeps its diagnostics alive until it is observed.
pub struct Retained<T> {
    pub result: Result<T, ExecutionError>,
    pub storage: Option<Arc<Reservation>>,
}

impl<T> Retained<T> {
    pub fn new(result: Result<T, ExecutionError>) -> Self {
        Self {
            result,
            storage: None,
        }
    }
}

/// Where a completion is between submission and observation.
enum Cell<T> {
    /// Not published; the observer may be waiting on `waker`.
    Waiting { waker: Option<Waker> },
    /// Published and not yet observed.
    Ready(Retained<T>),
    /// Published and observed.
    Consumed,
    /// The observer was dropped before publication.
    Abandoned,
    /// The observer was dropped after publication; the result was discarded.
    Discarded,
}

struct State<T> {
    cell: Cell<T>,
    /// The command's own outcome, kept for events that are queried after consumption.
    terminal: Option<Result<(), ExecutionError>>,
    /// Storage charged for this completion; released once the result is observed or
    /// discarded, unless the completion retains its status.
    storage: Option<Reservation>,
}

pub struct Completion<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
    retain_status: bool,
}

impl<T> Completion<T> {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::create(None, false)
    }

    pub fn create(storage: Option<Reservation>, retain_status: bool) -> Self {
        Self {
            state: Mutex::new(State {
                cell: Cell::Waiting { waker: None },
                terminal: None,
                storage,
            }),
            ready: Condvar::new(),
            retain_status,
        }
    }

    pub fn publish(&self, result: Result<T, ExecutionError>) {
        self.publish_retained(Retained::new(result));
    }

    pub fn publish_retained(&self, retained: Retained<T>) {
        let mut state = lock(&self.state);
        let waker = match &mut state.cell {
            Cell::Waiting { waker } => waker.take(),
            Cell::Abandoned => None,
            Cell::Ready(_) | Cell::Consumed | Cell::Discarded => {
                drop(state);
                drop(retained);
                return;
            }
        };
        if self.retain_status {
            state.terminal = Some(retained.result.as_ref().map(|_| ()).map_err(Clone::clone));
        }
        if matches!(state.cell, Cell::Abandoned) {
            state.cell = Cell::Discarded;
            let storage = self.release_storage(&mut state);
            drop(state);
            drop((retained, storage));
            return;
        }
        state.cell = Cell::Ready(retained);
        drop(state);
        self.ready.notify_all();
        if let Some(waker) = waker {
            // Wakers run executor code; a panicking waker must not unwind through the worker.
            let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
        }
    }

    pub fn take(&self) -> Result<T, ExecutionError> {
        let mut state = lock(&self.state);
        loop {
            match self.consume(&mut state) {
                Some(result) => {
                    drop(state);
                    return result;
                }
                None => {
                    state = self
                        .ready
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
    }

    pub fn poll(&self, cx: &Context<'_>) -> Poll<Result<T, ExecutionError>> {
        let replacement = cx.waker().clone();
        let mut state = lock(&self.state);
        if let Some(result) = self.consume(&mut state) {
            drop(state);
            return Poll::Ready(result);
        }
        let previous = match &mut state.cell {
            Cell::Waiting { waker } => waker.replace(replacement),
            _ => None,
        };
        drop(state);
        drop(previous);
        Poll::Pending
    }

    /// Takes a published result, or returns `None` while the command is still running.
    fn consume(&self, state: &mut State<T>) -> Option<Result<T, ExecutionError>> {
        match mem::replace(&mut state.cell, Cell::Consumed) {
            Cell::Ready(retained) => {
                let storage = self.release_storage(state);
                drop((retained.storage, storage));
                Some(retained.result)
            }
            Cell::Consumed | Cell::Discarded => Some(Err(ExecutionError::ResultConsumed)),
            cell @ (Cell::Waiting { .. } | Cell::Abandoned) => {
                state.cell = cell;
                None
            }
        }
    }

    pub fn abandon(&self) {
        let mut state = lock(&self.state);
        let previous = match mem::replace(&mut state.cell, Cell::Discarded) {
            Cell::Waiting { waker } => {
                state.cell = Cell::Abandoned;
                (None, waker)
            }
            Cell::Ready(retained) => (Some(retained), None),
            Cell::Abandoned => {
                state.cell = Cell::Abandoned;
                (None, None)
            }
            Cell::Consumed | Cell::Discarded => (None, None),
        };
        let storage = if matches!(state.cell, Cell::Discarded) {
            self.release_storage(&mut state)
        } else {
            None
        };
        drop(state);
        drop((previous, storage));
    }

    /// Takes this completion's storage unless it retains its status.
    fn release_storage(&self, state: &mut State<T>) -> Option<Reservation> {
        if self.retain_status {
            None
        } else {
            state.storage.take()
        }
    }

    pub fn status(&self) -> Option<Result<(), ExecutionError>> {
        lock(&self.state).terminal.clone()
    }

    pub fn is_abandoned(&self) -> bool {
        matches!(lock(&self.state).cell, Cell::Abandoned | Cell::Discarded)
    }

    #[cfg(test)]
    pub fn is_published(&self) -> bool {
        !matches!(
            lock(&self.state).cell,
            Cell::Waiting { .. } | Cell::Abandoned
        )
    }
}

pub type Pending<T> = Box<dyn FnOnce(Arc<Client>) -> T + Send>;

/// The pending result of a command.
///
/// Call [`wait`](Self::wait) to block for the result, or await the submission.
/// Dropping it discards the result but does not cancel the command.
#[must_use]
pub struct Submission<T> {
    pub(crate) completion: Arc<Completion<Pending<T>>>,
    pub(crate) client: Arc<Client>,
    pub(crate) launch_id: Option<u64>,
}

impl<T> Debug for Submission<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission").finish_non_exhaustive()
    }
}

impl<T> Submission<T> {
    /// Returns the launch identifier for a kernel launch, or `None` for other commands.
    ///
    /// The identifier is known as soon as the launch is accepted.
    #[must_use]
    pub fn launch_id(&self) -> Option<u64> {
        self.launch_id
    }

    /// Blocks until the command completes and returns its result.
    ///
    /// # Errors
    ///
    /// Returns the command's failure, [`ExecutionError::Cancelled`] if shutdown cancelled
    /// it, or [`ExecutionError::ResultConsumed`] if the result was already taken by
    /// polling.
    pub fn wait(self) -> Result<T, ExecutionError> {
        self.completion
            .take()
            .map(|promote| promote(Arc::clone(&self.client)))
    }
}

impl<T> Future for Submission<T> {
    type Output = Result<T, ExecutionError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.completion
            .poll(cx)
            .map(|result| result.map(|promote| promote(Arc::clone(&self.client))))
    }
}

impl<T> Drop for Submission<T> {
    fn drop(&mut self) {
        self.completion.abandon();
        drop(lock(&self.client.shared.admission));
        self.client.shared.changed.notify_one();
    }
}

/// The pending result of [`Emulator::shutdown`](super::Emulator::shutdown).
///
/// Call [`wait`](Self::wait) or await it.
#[derive(Debug)]
#[must_use]
pub struct Shutdown {
    pub(crate) shared: Arc<Shared>,
    // The allocation gives this observer a stable identity when the future moves.
    pub(crate) observer: Arc<()>,
}

impl Shutdown {
    pub(crate) fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            observer: Arc::new(()),
        }
    }

    /// Blocks until every accepted command has finished and the worker has released its
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns the fault that stopped the emulator, if any.
    pub fn wait(self) -> Result<(), ExecutionError> {
        let mut state = lock(&self.shared.admission);
        loop {
            if let Some(result) = &state.shutdown_result {
                return result
                    .as_ref()
                    .copied()
                    .map_err(|failure| failure.error.clone());
            }
            state = self
                .shared
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

impl Future for Shutdown {
    type Output = Result<(), ExecutionError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let replacement = cx.waker().clone();
        let mut state = lock(&self.shared.admission);
        if let Some(result) = &state.shutdown_result {
            let result = result
                .as_ref()
                .copied()
                .map_err(|failure| failure.error.clone());
            drop(state);
            Poll::Ready(result)
        } else {
            let previous = state
                .shutdown_wakers
                .insert(Arc::as_ptr(&self.observer).addr(), replacement);
            drop(state);
            drop(previous);
            Poll::Pending
        }
    }
}

impl Drop for Shutdown {
    fn drop(&mut self) {
        let waker = lock(&self.shared.admission)
            .shutdown_wakers
            .remove(&Arc::as_ptr(&self.observer).addr());
        drop(waker);
    }
}
