use std::{
    future::Future,
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    result::Result as StdResult,
    sync::{
        Arc, Condvar, Mutex, PoisonError,
        mpsc::{self, Receiver, Sender},
    },
    task::{Context, Poll, Waker},
    thread,
    time::Duration,
};

use crate::{
    error::{Error, Result},
    event::Event,
    stream::Stream,
    submission::Resources,
};

type CompletionResult = StdResult<(), Arc<Error>>;

type SharedState = (Mutex<State>, Condvar);

#[derive(Debug)]
struct Pending {
    resources: Resources,
    event: Arc<Event>,
    stream: Stream,
}

#[derive(Debug)]
enum Phase {
    Pending(Pending),
    Retiring,
    Complete(CompletionResult),
}

#[derive(Debug)]
struct State {
    phase: Phase,
    wakers: Vec<Waker>,
}

/// Checked CUDA completion, independent of callbacks and of any particular async runtime.
/// Clones observe the same result and may wait from different host threads.
#[derive(Debug, Clone)]
pub struct Completion {
    event: Arc<Event>,
    stream: Stream,
    state: Arc<SharedState>,
}

impl Completion {
    pub(crate) fn start(event: Arc<Event>, stream: Stream, resources: Resources) -> Result<Self> {
        let completion = Self {
            event: Arc::clone(&event),
            stream: stream.clone(),
            state: Arc::new((
                Mutex::new(State {
                    phase: Phase::Pending(Pending {
                        resources,
                        event,
                        stream,
                    }),
                    wakers: Vec::new(),
                }),
                Condvar::new(),
            )),
        };
        let work = Work {
            state: Arc::clone(&completion.state),
        };
        submit(work)?;
        Ok(completion)
    }

    fn signal(shared: &SharedState, result: Result<()>) {
        let (lock, ready) = shared;
        let wakers = {
            let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
            state.phase = Phase::Complete(result.map_err(|error| match error {
                Error::Completion(source) => source,
                error => Arc::new(error),
            }));
            mem::take(&mut state.wakers)
        };
        ready.notify_all();
        for waker in wakers {
            let _ = catch_unwind(AssertUnwindSafe(|| waker.wake()));
        }
    }

    /// Returns the stream whose recorded work this completion observes.
    #[must_use]
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Blocks for CUDA completion and resource retirement, observing the shared checked result.
    ///
    /// # Errors
    /// Returns a CUDA, dependency or resource-retirement error.
    pub fn wait(&self) -> Result<()> {
        if self.query()? {
            return Ok(());
        }
        Self::retire(&self.state, self.event.synchronize());
        let (lock, ready) = &*self.state;
        let mut state = lock.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Phase::Complete(result) = &state.phase {
                return result.clone().map_err(Error::Completion);
            }
            state = ready.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn retire(shared: &SharedState, native_result: Result<()>) {
        let pending = {
            let mut state = shared.0.lock().unwrap_or_else(PoisonError::into_inner);
            if !matches!(state.phase, Phase::Pending(_)) {
                return;
            }
            match mem::replace(&mut state.phase, Phase::Retiring) {
                Phase::Pending(pending) => pending,
                _ => unreachable!("pending phase was checked under the lock"),
            }
        };
        let mut retirement = Retirement {
            state: shared,
            pending: Some(pending),
            native_complete: native_result.is_ok(),
            result: native_result.err().map(Err),
        };
        // A dependency or user destructor can unwind. Retirement still signals every waiter.
        let result = catch_unwind(AssertUnwindSafe(|| {
            let pending = retirement.pending.as_ref().expect("retiring submission");
            if !retirement.native_complete {
                match pending.stream.synchronize() {
                    Ok(()) => retirement.native_complete = true,
                    Err(error) => {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "cuda completion failed; retaining live submission resources: {error}"
                        );
                        return Ok(());
                    }
                }
            }
            if retirement.result.is_none() {
                for dependency in &pending.resources.dependencies {
                    dependency.wait()?;
                }
                retirement.result = Some(Ok(()));
            }
            Ok::<_, Error>(())
        }));
        if let Ok(Err(error)) = result {
            retirement.result = Some(Err(error));
        }
    }

    /// Returns completion status without consuming the shared result.
    ///
    /// # Errors
    ///
    /// Returns the shared completion error once retirement has finished unsuccessfully. This can contain a CUDA, dependency, worker, or resource-retirement error. Pending or currently retiring work returns `Ok(false)` without waiting on the host.
    pub fn query(&self) -> Result<bool> {
        let state = self.state.0.lock().unwrap_or_else(PoisonError::into_inner);
        match &state.phase {
            Phase::Complete(Ok(())) => Ok(true),
            Phase::Complete(Err(error)) => Err(Error::Completion(Arc::clone(error))),
            Phase::Pending(_) | Phase::Retiring => Ok(false),
        }
    }

    /// Orders another stream after this submission without waiting on the host.
    ///
    /// # Errors
    ///
    /// Propagates context binding and CUDA event-wait errors from [`Stream::wait_event`]. This establishes device ordering without propagating the producer's checked host result; use [`Resources::depend_on`] to propagate it.
    pub fn order_on(&self, stream: &Stream) -> Result<()> {
        stream.wait_event(&self.event)
    }
}

// Keeps an acquired retirement terminal even if dependency resolution unwinds.
struct Retirement<'a> {
    state: &'a SharedState,
    pending: Option<Pending>,
    native_complete: bool,
    result: Option<Result<()>>,
}

impl Drop for Retirement<'_> {
    fn drop(&mut self) {
        let mut result = self.result.take().unwrap_or(Err(Error::CompletionPanicked));
        let pending = self.pending.take().expect("retiring submission");
        if !self.native_complete {
            pending
                .stream
                .context()
                .access_tracker
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .fail();
            mem::forget(pending);
        } else if catch_unwind(AssertUnwindSafe(|| drop(pending))).is_err() && result.is_ok() {
            result = Err(Error::ResourceDestructionPanicked);
        }
        Completion::signal(self.state, result);
    }
}

impl Future for Completion {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Waker cloning and destruction can run user code, so keep them outside the mutex.
        let mut waker = Some(cx.waker().clone());
        let result = {
            let mut state = self.state.0.lock().unwrap_or_else(PoisonError::into_inner);
            if let Phase::Complete(result) = &state.phase {
                Poll::Ready(result.clone().map_err(Error::Completion))
            } else {
                if !state
                    .wakers
                    .iter()
                    .any(|previous| previous.will_wake(cx.waker()))
                {
                    state.wakers.push(waker.take().expect("new waker"));
                }
                Poll::Pending
            }
        };
        drop(waker);
        result
    }
}

struct Work {
    // Native owners belong to Pending and retire before completion is published.
    // A queued worker record must not extend their lifetime past a successful wait.
    state: Arc<SharedState>,
}

impl Work {
    fn query(&self) -> Result<bool> {
        let state = self.state.0.lock().unwrap_or_else(PoisonError::into_inner);
        match &state.phase {
            Phase::Pending(pending) => pending.event.query(),
            Phase::Retiring | Phase::Complete(_) => Ok(true),
        }
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        // A failed worker or channel cannot release queued owners or strand a waiter.
        Completion::retire(&self.state, Err(Error::CompletionWorkerStopped));
    }
}

fn submit(work: Work) -> Result<()> {
    static WORKER: Mutex<Option<Sender<Work>>> = Mutex::new(None);
    let mut worker = WORKER.lock().unwrap_or_else(PoisonError::into_inner);
    if worker.is_none() {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("singe-cuda-completion".into())
            .spawn(move || run(&receiver))?;
        *worker = Some(sender);
    }
    if let Err(mpsc::SendError(work)) = worker.as_ref().expect("worker was started").send(work) {
        *worker = None;
        drop(worker);
        drop(work);
        return Err(Error::CompletionWorkerStopped);
    }
    Ok(())
}

fn run(receiver: &Receiver<Work>) {
    let mut pending = Vec::<Work>::new();
    loop {
        let work = if pending.is_empty() {
            match receiver.recv() {
                Ok(work) => Some(work),
                Err(_) => return,
            }
        } else {
            match receiver.recv_timeout(Duration::from_micros(100)) {
                Ok(work) => Some(work),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        };
        pending.extend(work);
        pending.extend(receiver.try_iter());
        let mut index = 0;
        while index < pending.len() {
            match pending[index].query() {
                Ok(false) => index += 1,
                result => {
                    let work = pending.swap_remove(index);
                    Completion::retire(&work.state, result.map(|_| ()));
                }
            }
        }
    }
}
