//! A model on a thread of its own: Singe's models and decoders stay on the thread that loaded them, so the HTTP side sends them work and awaits the answer.

use std::{
    error::Error,
    io,
    panic::{self, AssertUnwindSafe},
    thread,
};

use singe::Device;
#[cfg(feature = "cuda_13_4")]
use singe::backend::cuda::{CudaStream, set_current_stream};
use tokio::sync::{
    mpsc::{self, error::TrySendError},
    oneshot, watch,
};

use crate::error::{ApiError, ApiResult, error_chain};

/// Why a model failed to load.
pub type LoadError = Box<dyn Error + Send + Sync>;

/// Whether a model can take work yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    /// The model's thread is still loading it; requests get 503.
    Loading,
    /// The model takes requests.
    Ready,
    /// Loading failed, or the thread stopped, for this reason.
    Failed(String),
}

type Job<M> = Box<dyn FnOnce(&mut M) + Send>;

/// A model `M` that a dedicated thread loads and then runs jobs on, one at a time, in the order they came.
pub struct Worker<M> {
    name: String,
    jobs: mpsc::Sender<Job<M>>,
    state: watch::Receiver<ModelState>,
}

impl<M> Clone for Worker<M> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            jobs: self.jobs.clone(),
            state: self.state.clone(),
        }
    }
}

impl<M: 'static> Worker<M> {
    /// Spawns a thread named `singe-{name}` that loads its model on `device` with `load`, then runs jobs, up to `queue` of which wait at once.
    /// On a CUDA device the thread runs on a stream of its own, so its model's work runs beside other models' rather than behind it.
    /// A `load` that fails or panics leaves the worker [`ModelState::Failed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the thread cannot be spawned.
    pub fn spawn(
        name: &str,
        device: Device,
        queue: usize,
        load: impl FnOnce() -> Result<M, LoadError> + Send + 'static,
    ) -> io::Result<Self> {
        let (jobs, mut receiver) = mpsc::channel::<Job<M>>(queue.max(1));
        let (states, state) = watch::channel(ModelState::Loading);
        thread::Builder::new()
            .name(format!("singe-{name}"))
            .spawn(move || {
                let _stream = match own_stream(device) {
                    Ok(stream) => stream,
                    Err(error) => {
                        states.send_replace(ModelState::Failed(error_chain(&error)));
                        return;
                    }
                };
                let mut model = match panic::catch_unwind(AssertUnwindSafe(load)) {
                    Ok(Ok(model)) => model,
                    Ok(Err(error)) => {
                        states.send_replace(ModelState::Failed(error_chain(&*error)));
                        return;
                    }
                    Err(_) => {
                        states.send_replace(ModelState::Failed("loading panicked".to_owned()));
                        return;
                    }
                };
                states.send_replace(ModelState::Ready);
                while let Some(job) = receiver.blocking_recv() {
                    job(&mut model);
                }
            })?;
        Ok(Self {
            name: name.to_owned(),
            jobs,
            state,
        })
    }

    /// Runs `work` on the model's thread and returns what it gives, or an error when the model is not ready or its queue is full.
    /// Work whose caller has gone, as a client that disconnected, is skipped when its turn comes.
    ///
    /// # Errors
    ///
    /// Returns a 503 error while the model loads, a 429 error when `queue` jobs wait already, a 500 error if the model failed to load, its thread stopped, or `work` panicked, and any error `work` returns.
    pub async fn call<T: Send + 'static>(
        &self,
        work: impl FnOnce(&mut M) -> ApiResult<T> + Send + 'static,
    ) -> ApiResult<T> {
        ready(&self.name, &self.state)?;
        let (reply, answer) = oneshot::channel();
        let job: Job<M> = Box::new(move |model| {
            if reply.is_closed() {
                return;
            }

            let result = panic::catch_unwind(AssertUnwindSafe(|| work(model)))
                .unwrap_or_else(|_| Err(ApiError::internal("the model panicked")));
            reply.send(result).ok();
        });
        self.jobs.try_send(job).map_err(|error| match error {
            TrySendError::Full(_) => ApiError::busy(&self.name),
            TrySendError::Closed(_) => stopped(&self.name),
        })?;
        answer.await.map_err(|_| stopped(&self.name))?
    }

    /// Returns whether the model can take work.
    #[must_use]
    pub fn state(&self) -> ModelState {
        self.state.borrow().clone()
    }

    /// Waits until the model has loaded or failed to.
    pub async fn loaded(&self) -> ModelState {
        loaded(self.state.clone()).await
    }
}

pub(crate) async fn loaded(mut state: watch::Receiver<ModelState>) -> ModelState {
    state
        .wait_for(|state| *state != ModelState::Loading)
        .await
        .map_or_else(
            |_| ModelState::Failed("the model's thread stopped".to_owned()),
            |state| state.clone(),
        )
}

pub(crate) fn ready(name: &str, state: &watch::Receiver<ModelState>) -> ApiResult<()> {
    match &*state.borrow() {
        ModelState::Ready => Ok(()),
        ModelState::Loading => Err(ApiError::loading(name)),
        ModelState::Failed(error) => Err(ApiError::internal(format!(
            "model {name} failed to load: {error}"
        ))),
    }
}

fn stopped(name: &str) -> ApiError {
    ApiError::internal(format!("model {name}'s thread stopped"))
}

/// Makes a new stream on `device` this thread's current one until what it returns is dropped, so the thread's model runs beside other models rather than behind them on the device's shared eager stream.
#[cfg(feature = "cuda_13_4")]
pub(crate) fn own_stream(device: Device) -> singe::Result<Option<impl Sized>> {
    if !matches!(device, Device::Cuda(_)) {
        return Ok(None);
    }

    let stream = CudaStream::create(device)?;
    Ok(Some(set_current_stream(&stream)?))
}

/// Gives the thread nothing: without CUDA its model runs on the host.
#[cfg(not(feature = "cuda_13_4"))]
pub(crate) const fn own_stream(_device: Device) -> singe::Result<Option<()>> {
    Ok(None)
}
