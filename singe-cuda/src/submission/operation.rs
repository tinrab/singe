use std::{
    fmt::{self, Debug, Formatter},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    context::Context,
    error::{Error, Result},
    graph::owned::{Recordable, Recording},
    stream::{Stream, StreamFlags},
    submission::{Completion, Execution, Resources, Submission},
};

/// Lazy owned CUDA work.
///
/// # Safety
/// Implementations must register all resources and accesses through [`Execution::enqueue`]
/// before submitting native work, including on error and unwind paths.
/// Returned pending values must not permit safe initialized reads before successful completion.
/// Keep capture-incompatible preparation outside implementations intended for graph recording.
pub unsafe trait Operation: Sized {
    /// Value produced by enqueue before successful CUDA completion. It must not permit safe reads of results that are still pending.
    type Pending;

    /// Host-visible result produced by resolution after successful completion.
    type Output: 'static;

    /// Registers resources and enqueues this operation on the execution's stream, returning its pending result.
    ///
    /// # Errors
    ///
    /// Returns operation-specific preparation or enqueue errors, including resource, context, dependency, and CUDA failures. Implementations must retain resources for any work already queued when an error occurs.
    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending>;

    /// Resolves host results after successful native completion and resource retirement.
    ///
    /// # Safety
    /// `pending` must come from this operation's enqueue and its execution must have completed successfully.
    ///
    /// # Errors
    ///
    /// Returns operation-specific result-conversion or host-resolution errors after native work has completed successfully.
    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output>;

    /// Enqueues this operation on `stream` and returns its pending result with independent checked completion.
    ///
    /// Dropping the returned submission does not cancel its native work.
    ///
    /// # Errors
    ///
    /// Propagates execution setup, operation enqueue, and completion registration errors. An execution ledger drains already submitted work on failure before releasing its resources, or preserves the resources if completion cannot be established.
    fn submit_on(self, stream: &Stream) -> Result<Submission<Self::Output, Self::Pending>> {
        let mut execution = Execution::create(stream)?;
        let output = self.enqueue(&mut execution)?;
        execution.finish_with(output, Self::resolve)
    }

    /// Chains dependent device work on the same stream without promoting pending values.
    fn then<F, N>(self, next: F) -> Then<Self, F>
    where
        Self: DeviceOperation,
        F: FnOnce(Self::Output) -> N,
        N: Operation,
    {
        Then { first: self, next }
    }

    /// Pairs independent work on the selected stream; separate submissions can overlap.
    fn pair<O: Operation>(self, other: O) -> Pair<Self, O> {
        Pair(self, other)
    }

    /// Runs host code on the waiting or polling thread after successful completion.
    /// Dropping the submission before resolution discards this callback.
    fn map<F, T: 'static>(self, map: F) -> Map<Self, F>
    where
        F: FnOnce(Self::Output) -> T,
    {
        Map {
            operation: self,
            map,
        }
    }

    /// Shares one execution and orders every consuming stream after its completion event.
    fn shared(self) -> Shared<Self>
    where
        Self: DeviceOperation,
        Self::Output: Send + Sync,
    {
        Shared {
            state: Arc::new(Mutex::new(SharedState::New(Some(self)))),
        }
    }
}

/// Device work whose pending output is already suitable for composing further device work.
/// Host consumers still need checked completion before reading produced values.
///
/// # Safety
/// Resolution must be the identity function, and pending outputs must uphold [`Operation`]'s contract.
/// Enqueue must never rely on host resolution to schedule native work or initialize output metadata.
pub unsafe trait DeviceOperation: Operation<Pending = <Self as Operation>::Output> {}

/// Two operations composed so the second consumes the first device operation's pending output on the same stream.
///
/// Construct with [`Operation::then`].
#[derive(Debug)]
pub struct Then<O, F> {
    first: O,
    next: F,
}

unsafe impl<O, F, N> Operation for Then<O, F>
where
    O: DeviceOperation,
    F: FnOnce(O::Output) -> N,
    N: Operation,
{
    type Pending = N::Pending;

    type Output = N::Output;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending> {
        let output = self.first.enqueue(execution)?;
        (self.next)(output).enqueue(execution)
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { N::resolve(pending) }
    }
}

/// Two operations enqueued in order on one execution stream and resolved as a pair.
///
/// Construct with [`Operation::pair`].
#[derive(Debug)]
pub struct Pair<A, B>(A, B);

unsafe impl<A: Operation, B: Operation> Operation for Pair<A, B> {
    type Pending = (A::Pending, B::Pending);

    type Output = (A::Output, B::Output);

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending> {
        Ok((self.0.enqueue(execution)?, self.1.enqueue(execution)?))
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { Ok((A::resolve(pending.0)?, B::resolve(pending.1)?)) }
    }
}

/// An operation whose successful host result is transformed by a callback on the waiting or polling thread.
///
/// Construct with [`Operation::map`]. The callback is discarded if the submission is dropped before resolution.
#[derive(Debug)]
pub struct Map<O, F> {
    operation: O,
    map: F,
}

unsafe impl<O, F, T: 'static> Operation for Map<O, F>
where
    O: Operation,
    F: FnOnce(O::Output) -> T,
{
    type Pending = (O::Pending, F);

    type Output = T;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending> {
        Ok((self.operation.enqueue(execution)?, self.map))
    }

    unsafe fn resolve((pending, map): Self::Pending) -> Result<T> {
        // SAFETY: the caller upholds this function's safety contract.
        Ok(map(unsafe { O::resolve(pending)? }))
    }
}

#[derive(Debug)]
enum SharedState<O: DeviceOperation> {
    New(Option<O>),
    Submitted {
        output: Arc<O::Output>,
        completion: Completion,
    },
    Failed(Arc<Error>),
}

/// Clones of one lazily submitted device operation, sharing its output and checked completion.
///
/// The first consumer submits the producer. Later consumers order their streams after its completion event and retain the output. Producer submission errors are shared with consumers. Construct with [`Operation::shared`].
pub struct Shared<O: DeviceOperation> {
    state: Arc<Mutex<SharedState<O>>>,
}

impl<O: DeviceOperation> Debug for Shared<O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared").finish_non_exhaustive()
    }
}

impl<O: DeviceOperation> Clone for Shared<O> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

unsafe impl<O: DeviceOperation> Operation for Shared<O>
where
    O::Output: Send + Sync,
{
    type Pending = Arc<O::Output>;

    type Output = Arc<O::Output>;

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        Ok(pending)
    }

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Output> {
        let mut state = self.state.lock().map_err(|_| Error::ResourcePoisoned {
            name: "shared operation",
        })?;
        if let SharedState::New(operation) = &mut *state {
            let operation = operation.take().expect("shared operation has not executed");
            *state = match operation.submit_on(execution.stream()) {
                Ok(mut submission) => SharedState::Submitted {
                    output: Arc::new(submission.pending.take().expect("new shared output")),
                    completion: submission.completion.clone(),
                },
                Err(error) => SharedState::Failed(Arc::new(error)),
            };
        }
        match &*state {
            SharedState::Submitted { output, completion } => {
                let mut resources = Resources::new();
                resources.depend_on(completion.clone());
                resources.retain(Arc::clone(output));
                // SAFETY: the producer owns all native work; this operation only enqueues its dependency.
                unsafe {
                    execution.enqueue(resources, |_| Ok::<_, Error>(()))?;
                }
                Ok(Arc::clone(output))
            }
            SharedState::Failed(error) => Err(Error::Completion(Arc::clone(error))),
            SharedState::New(_) => unreachable!("shared operation was submitted"),
        }
    }
}

/// Optional context-bound round-robin scheduling with no global stream selection.
#[derive(Debug)]
pub struct StreamPool {
    streams: Vec<Stream>,
    next: AtomicUsize,
}

impl StreamPool {
    /// Creates `streams` nonblocking CUDA streams in `context` for round-robin submission.
    ///
    /// # Errors
    ///
    /// Returns `Error::ZeroValue` if `streams` is zero. Also propagates context binding or stream creation errors. Streams already created are dropped if a later creation fails.
    pub fn create(context: &Arc<Context>, streams: usize) -> Result<Self> {
        if streams == 0 {
            return Err(Error::ZeroValue {
                name: "stream pool size".into(),
            });
        }
        Ok(Self {
            streams: (0..streams)
                .map(|_| context.create_stream_with_flags(StreamFlags::NON_BLOCKING))
                .collect::<Result<_>>()?,
            next: AtomicUsize::new(0),
        })
    }

    /// Submits `operation` on the next stream selected in round-robin order.
    ///
    /// The selection advances even when submission fails. Separate operations can overlap unless their declared accesses or dependencies require ordering.
    ///
    /// # Errors
    ///
    /// Propagates errors from the operation's [`Operation::submit_on`] implementation. An error can occur after native work has been queued; its execution ledger retains resources through cleanup.
    pub fn submit<O: Operation>(&self, operation: O) -> Result<Submission<O::Output, O::Pending>> {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        operation.submit_on(&self.streams[index])
    }

    /// Borrows the pool's streams in creation order.
    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }
}

unsafe impl<A: Recordable, B: Recordable> Recordable for Pair<A, B> {
    type Output = (A::Output, B::Output);

    fn record(self, recording: &mut Recording) -> Result<Self::Output> {
        Ok((self.0.record(recording)?, self.1.record(recording)?))
    }
}

unsafe impl<O: Operation> Operation for Result<O> {
    type Pending = O::Pending;

    type Output = O::Output;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending> {
        self?.enqueue(execution)
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { O::resolve(pending) }
    }
}

// SAFETY: these compositions preserve identity resolution and eager device enqueue.
unsafe impl<O: DeviceOperation, F, N: DeviceOperation> DeviceOperation for Then<O, F> where
    F: FnOnce(O::Output) -> N
{
}

unsafe impl<A: DeviceOperation, B: DeviceOperation> DeviceOperation for Pair<A, B> {}

unsafe impl<O: DeviceOperation> DeviceOperation for Result<O> {}

unsafe impl<O: DeviceOperation> DeviceOperation for Shared<O> where O::Output: Send + Sync {}
