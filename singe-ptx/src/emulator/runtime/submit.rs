use std::{
    fmt::{self, Debug, Formatter},
    sync::{Arc, atomic::Ordering},
    task::Poll,
};

use crate::emulator::{
    budget::{ALLOCATION_OVERHEAD, NODE_OVERHEAD, Reservation, allocation},
    execution_error::{ExecutionError, SubmitError},
    runtime::{
        Client, Command, CommandControl, CommandKind, QueueReservation, Staging, admission,
        completion::{Completion, Pending, Retained, Submission},
        lock,
        worker::Worker,
    },
};

/// Reserved with each command so a failure can always retain its error report.
pub const TERMINAL_ERROR_BYTES: usize = 512;

pub struct Producer<T> {
    pub completion: Arc<Completion<Pending<T>>>,
    pub finished: bool,
}

impl<T> Debug for Producer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Producer").finish_non_exhaustive()
    }
}

/// The fault that stopped an emulator, with the storage that keeps its report reserved.
#[derive(Clone)]
pub struct RetainedFailure {
    pub error: ExecutionError,
    _storage: Option<Arc<Reservation>>,
}

impl RetainedFailure {
    pub fn new(error: ExecutionError, storage: Option<Arc<Reservation>>) -> Self {
        Self {
            error,
            _storage: storage,
        }
    }

    #[cfg(test)]
    pub fn retained_bytes(&self) -> usize {
        let Self {
            _storage: storage, ..
        } = self;
        storage.as_ref().map_or(0, |storage| storage.bytes())
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        if !self.finished {
            self.completion.publish(Err(ExecutionError::WorkerFailed));
        }
    }
}

/// Admits a command whose closure, built once its staging token exists, is polled by the
/// worker until it returns a result. `terminal` carries cancellation or a prior fault.
pub fn submit_poll<T: Send + 'static, F, B>(
    client: &Arc<Client>,
    stream: u64,
    kind: CommandKind,
    diagnostic: bool,
    size: usize,
    build: B,
) -> Result<Submission<T>, SubmitError>
where
    F: FnMut(&mut Worker, Option<ExecutionError>) -> Poll<Result<Pending<T>, ExecutionError>>
        + Send
        + 'static,
    B: FnOnce(Staging) -> F,
{
    submit_retained(client, stream, kind, diagnostic, size, move |staging| {
        let mut step = build(staging);
        move |worker: &mut Worker, terminal| step(worker, terminal).map(Retained::new)
    })
}

/// Returns the execution storage charged while a command is queued or running: its queue
/// node, its closure and producer, a reserved error report, and its stream entry.
fn command_charge<T, F>() -> Result<usize, SubmitError> {
    [
        size_of::<Command>()
            .checked_mul(8)
            .ok_or(SubmitError::ExecutionStateLimit)?,
        size_of::<F>(),
        size_of::<Producer<T>>(),
        TERMINAL_ERROR_BYTES,
        // One stream entry in the queue-selection set, and the producer allocation.
        NODE_OVERHEAD,
        ALLOCATION_OVERHEAD,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(SubmitError::ExecutionStateLimit)
}

/// [`submit_poll`] for commands that keep storage alive with their result, such as a launch
/// whose fault report retains diagnostics.
pub fn submit_retained<T: Send + 'static, F, B>(
    client: &Arc<Client>,
    stream: u64,
    kind: CommandKind,
    diagnostic: bool,
    size: usize,
    build: B,
) -> Result<Submission<T>, SubmitError>
where
    F: FnMut(&mut Worker, Option<ExecutionError>) -> Poll<Retained<Pending<T>>> + Send + 'static,
    B: FnOnce(Staging) -> F,
{
    let shared = &client.shared;
    let mut state = lock(&shared.admission);
    admission(state.phase, diagnostic)?;
    if state.queue.len() + state.pending_dependencies + state.reservations
        >= shared.options.queued_commands
    {
        return Err(SubmitError::QueueFull);
    }
    let staged = state
        .staged
        .checked_add(size)
        .filter(|v| *v <= shared.options.staging_storage_bytes)
        .ok_or(SubmitError::StagingLimit)?;
    let mut execution = Some(
        shared
            .execution_budget
            .reserve(command_charge::<T, F>()?)
            .map_err(|_| SubmitError::ExecutionStateLimit)?,
    );
    let completion_storage = shared
        .execution_budget
        .reserve(allocation::<Completion<Pending<T>>>())
        .map_err(|_| SubmitError::ExecutionStateLimit)?;
    let launch_id = if kind == CommandKind::Launch {
        Some(state.allocate_launch_id()?)
    } else {
        None
    };
    state.staged = staged;
    state.reservations += 1;
    drop(state);
    let mut reservation = QueueReservation {
        shared: Arc::clone(shared),
        active: true,
    };
    // Reserve before copying, but never hold the admission lock during a snapshot.
    // Acceptance is the final enqueue, ordered with shutdown under the same lock.
    let control = Arc::new(CommandControl::default());
    let mut run = Some(build(Staging {
        shared: Arc::downgrade(shared),
        bytes: size,
        launch_id,
        control: Arc::clone(&control),
    }));
    let mut state = lock(&shared.admission);
    state.reservations -= 1;
    reservation.active = false;
    if let Err(error) = admission(state.phase, diagnostic) {
        drop(state);
        drop(run);
        return Err(error);
    }
    let completion = Arc::new(Completion::create(
        Some(completion_storage),
        kind == CommandKind::Event,
    ));
    let mut producer = Producer {
        completion: Arc::clone(&completion),
        finished: false,
    };
    state.queue.push_back(Command {
        stream,
        kind,
        control,
        poll: Box::new(move |worker, terminal| {
            let terminal = terminal.or_else(|| {
                if worker.shared.cancel.load(Ordering::Acquire) {
                    Some(ExecutionError::Cancelled)
                } else if worker.failure.is_some() && !diagnostic {
                    Some(ExecutionError::Faulted)
                } else {
                    None
                }
            });
            let Some(step) = run.as_mut() else {
                return true;
            };
            let Poll::Ready(mut retained) = step(worker, terminal) else {
                return false;
            };
            // Release input snapshots and private operation state before waking observers.
            drop(run.take());
            if retained.result.is_err() && retained.storage.is_none() {
                retained.storage = execution
                    .as_mut()
                    .map(|storage| Arc::new(storage.split(TERMINAL_ERROR_BYTES)));
            }
            drop(execution.take());
            if kind == CommandKind::Dependency {
                lock(&worker.shared.admission).pending_dependencies -= 1;
            }
            producer.finished = true;
            producer.completion.publish_retained(retained);
            true
        }),
    });
    drop(state);
    shared.changed.notify_one();
    Ok(Submission {
        completion,
        client: Arc::clone(client),
        launch_id,
    })
}
