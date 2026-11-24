//! The thread a text model generates on: one decoder of `max_num_seqs` rows, and a running batch requests join between its steps.
//! Each request reserves its prompt and token limit in the shared cache before it joins, so a step never overruns it; requests that don't fit yet wait in order.

use std::{
    collections::HashMap,
    io,
    panic::{self, AssertUnwindSafe},
    thread,
};

use singe::{
    generation::{Decoder, DecoderOptions, GenerationOptions, RunningBatch, SequenceOptions},
    random::Generator,
};
use singe_model::task::text::text_generation::TextGenerationPipeline;
use tokio::sync::{
    mpsc::{self, error::TrySendError},
    watch,
};

use crate::{
    error::{ApiError, ApiResult, error_chain},
    worker::{self, LoadError, ModelState},
};

/// A model the engine generates with: anything that gives a decoder over rows and a capacity.
pub trait TextModel: 'static {
    /// Creates a decoder over `options`' rows and cache, compiled when `compile` is set.
    fn decoder(
        &self,
        options: DecoderOptions,
        compile: bool,
    ) -> singe_model::Result<Box<dyn Decoder>>;
}

impl TextModel for TextGenerationPipeline {
    fn decoder(
        &self,
        options: DecoderOptions,
        compile: bool,
    ) -> singe_model::Result<Box<dyn Decoder>> {
        Self::decoder(self, options, compile)
    }
}

/// How a [`TextEngine`] runs its model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEngineOptions {
    /// The sequences a step runs at most.
    pub max_num_seqs: usize,
    /// The most tokens a sequence holds, prompt included.
    pub max_model_len: usize,
    /// The tokens the shared cache holds across sequences.
    pub cache_tokens: usize,
    /// The tokens in one page of the cache.
    pub page_size: usize,
    /// Runs compiled steps rather than eager ones.
    pub compile: bool,
    /// The requests that wait for a row at most; more are refused with 429.
    pub max_queued_requests: usize,
    /// Seeds the generator sampled tokens are drawn from.
    pub seed: u64,
}

/// What a request may ask of an engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextLimits {
    /// The most tokens a sequence holds, prompt included.
    pub max_model_len: usize,
    /// The tokens the shared cache holds across sequences.
    pub cache_tokens: usize,
    /// The tokens in one page of the cache.
    pub page_size: usize,
}

impl TextLimits {
    /// Returns the cache tokens a sequence of `tokens` takes: whole pages.
    #[must_use]
    pub const fn reservation(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.page_size) * self.page_size
    }

    /// Checks that a prompt of `prompt` tokens and `max_tokens` more fit a sequence.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if they exceed `max_model_len`, or if their reservation exceeds `cache_tokens`.
    pub fn check(&self, prompt: usize, max_tokens: usize) -> ApiResult<()> {
        let total = prompt + max_tokens;
        if total > self.max_model_len {
            return Err(ApiError::invalid(format!(
                "a prompt of {prompt} tokens and {max_tokens} to generate exceed the model's {} tokens",
                self.max_model_len
            )));
        }

        if self.reservation(total) > self.cache_tokens {
            return Err(ApiError::invalid(format!(
                "a prompt of {prompt} tokens and {max_tokens} to generate exceed the cache's {} tokens",
                self.cache_tokens
            )));
        }
        Ok(())
    }
}

/// Why a sequence ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token or a stop string ended the sequence.
    Stop,
    /// The sequence reached its token limit.
    Length,
}

impl FinishReason {
    /// Returns the reason as OpenAI's `finish_reason` names it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// What the engine sends a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationEvent {
    /// A generated token, the stop token included.
    Token(i64),
    /// The sequence ended; no event follows.
    Finished(FinishReason),
    /// The request failed to join, or a step failed, for this reason; no event follows.
    Failed(String),
}

/// A request's events, in order; dropping it cancels the request.
#[derive(Debug)]
pub struct GenerationStream {
    events: mpsc::UnboundedReceiver<GenerationEvent>,
}

impl GenerationStream {
    /// Returns the next event, or `None` once the engine has dropped the request.
    pub async fn recv(&mut self) -> Option<GenerationEvent> {
        self.events.recv().await
    }
}

struct Submission {
    prompt: Vec<i64>,
    options: SequenceOptions,
    reservation: usize,
    events: mpsc::UnboundedSender<GenerationEvent>,
}

struct Running {
    events: mpsc::UnboundedSender<GenerationEvent>,
    stop_tokens: Vec<i64>,
    reservation: usize,
}

/// A handle to a text model's engine thread, which requests are submitted through.
#[derive(Clone)]
pub struct TextEngine {
    name: String,
    submissions: mpsc::Sender<Submission>,
    state: watch::Receiver<ModelState>,
    limits: TextLimits,
}

impl TextEngine {
    /// Spawns the engine's thread, named `singe-{name}`, which loads its model with `load`, then generates.
    /// A load that fails or panics, or a later failure that stops the engine, leaves it [`ModelState::Failed`].
    ///
    /// # Errors
    ///
    /// Returns an error if the thread cannot be spawned.
    pub fn spawn<M: TextModel>(
        name: &str,
        options: TextEngineOptions,
        load: impl FnOnce() -> Result<M, LoadError> + Send + 'static,
    ) -> io::Result<Self> {
        let (submissions, mut receiver) = mpsc::channel(options.max_queued_requests.max(1));
        let (states, state) = watch::channel(ModelState::Loading);
        thread::Builder::new()
            .name(format!("singe-{name}"))
            .spawn(move || {
                // A failure before the first step fails the load; a later one stops the engine.
                let failure = panic::catch_unwind(AssertUnwindSafe(|| match load() {
                    Ok(model) => run(&model, options, &mut receiver, &states)
                        .err()
                        .map(|error| error_chain(&error)),
                    Err(error) => Some(error_chain(&*error)),
                }))
                .unwrap_or_else(|_| Some("the engine panicked".to_owned()));
                if let Some(failure) = failure {
                    states.send_replace(ModelState::Failed(failure));
                }
            })?;
        Ok(Self {
            name: name.to_owned(),
            submissions,
            state,
            limits: TextLimits {
                max_model_len: options.max_model_len,
                cache_tokens: options.cache_tokens,
                page_size: options.page_size,
            },
        })
    }

    /// Returns what a request may ask of the engine.
    #[must_use]
    pub const fn limits(&self) -> TextLimits {
        self.limits
    }

    /// Returns whether the engine can take requests.
    #[must_use]
    pub fn state(&self) -> ModelState {
        self.state.borrow().clone()
    }

    /// Waits until the model has loaded or failed to.
    pub async fn loaded(&self) -> ModelState {
        worker::loaded(self.state.clone()).await
    }

    /// Queues a request to generate after `prompt` as `options` asks, and returns its events.
    /// The request joins the running batch once the cache has room for its prompt and token limit.
    ///
    /// # Errors
    ///
    /// Returns a 503 error while the model loads, a 400 error for an empty prompt, a token limit of zero, or a request beyond [`TextLimits`], a 429 error when `max_queued_requests` wait already, and a 500 error if the model failed to load or the engine stopped.
    pub fn submit(
        &self,
        prompt: Vec<i64>,
        options: SequenceOptions,
    ) -> ApiResult<GenerationStream> {
        worker::ready(&self.name, &self.state)?;
        if prompt.is_empty() {
            return Err(ApiError::invalid("an empty prompt"));
        }

        if options.max_new_tokens == 0 {
            return Err(ApiError::invalid("a limit of no tokens to generate"));
        }

        self.limits.check(prompt.len(), options.max_new_tokens)?;
        let (events, receiver) = mpsc::unbounded_channel();
        let submission = Submission {
            reservation: self
                .limits
                .reservation(prompt.len() + options.max_new_tokens),
            prompt,
            options,
            events,
        };
        self.submissions
            .try_send(submission)
            .map_err(|error| match error {
                TrySendError::Full(_) => ApiError::busy(&self.name),
                TrySendError::Closed(_) => {
                    ApiError::internal(format!("model {}'s engine stopped", self.name))
                }
            })?;
        Ok(GenerationStream { events: receiver })
    }
}

fn run<M: TextModel>(
    model: &M,
    options: TextEngineOptions,
    submissions: &mut mpsc::Receiver<Submission>,
    states: &watch::Sender<ModelState>,
) -> singe_model::Result<()> {
    let to_i64 = |value: usize, what: &str| {
        i64::try_from(value).map_err(|_| singe::Error::out_of_range(what.to_owned()))
    };
    let decoder_options = DecoderOptions::new(
        options.max_num_seqs,
        to_i64(options.max_model_len, "max model length")?,
    )
    .with_page_size(to_i64(options.page_size, "page size")?)
    .with_cache_tokens(to_i64(options.cache_tokens, "cache tokens")?);
    let mut decoder = model.decoder(decoder_options, options.compile)?;
    // The batch's own options only bound `join`, which the engine doesn't call.
    let defaults = GenerationOptions {
        max_new_tokens: 1,
        ..GenerationOptions::default()
    };
    let mut batch = RunningBatch::create(decoder.as_mut(), defaults)?;
    let generator = Generator::new(options.seed);
    states.send_replace(ModelState::Ready);

    // The request that came first and found no room; the others wait in the channel, whose capacity bounds the queue.
    let mut waiting = None::<Submission>;
    let mut running = HashMap::<usize, Running>::new();
    let mut reserved = 0;
    loop {
        if running.is_empty() && waiting.is_none() {
            match submissions.blocking_recv() {
                Some(submission) => waiting = Some(submission),
                None => return Ok(()),
            }
        }

        // Requests join in the order they came, while the cache has room for the first.
        while let Some(submission) = waiting.take().or_else(|| submissions.try_recv().ok()) {
            if submission.events.is_closed() {
                continue;
            }

            if reserved + submission.reservation > options.cache_tokens {
                waiting = Some(submission);
                break;
            }
            match batch.join_with_options(&submission.prompt, submission.options.clone(), |_, _| {
                Ok(())
            }) {
                Ok(Some(id)) => {
                    reserved += submission.reservation;
                    running.insert(
                        id,
                        Running {
                            events: submission.events,
                            stop_tokens: submission.options.stop_tokens,
                            reservation: submission.reservation,
                        },
                    );
                }
                Ok(None) => {
                    waiting = Some(submission);
                    break;
                }
                Err(error) => {
                    submission
                        .events
                        .send(GenerationEvent::Failed(error.to_string()))
                        .ok();
                }
            }
        }

        // A request whose receiver is gone was cancelled: its client left, or its text met a stop string.
        let cancelled = running
            .iter()
            .filter(|(_, request)| request.events.is_closed())
            .map(|(&id, _)| id)
            .collect::<Vec<_>>();
        for id in cancelled {
            batch.leave(id)?;
            if let Some(request) = running.remove(&id) {
                reserved -= request.reservation;
            }
        }

        if batch.is_empty() {
            continue;
        }
        match batch.advance(Some(&generator)) {
            Ok(tokens) => {
                for token in tokens {
                    let Some(request) = running.get(&token.sequence) else {
                        continue;
                    };
                    let delivered = request
                        .events
                        .send(GenerationEvent::Token(token.token))
                        .is_ok();
                    if token.finished {
                        let reason = if request.stop_tokens.contains(&token.token) {
                            FinishReason::Stop
                        } else {
                            FinishReason::Length
                        };
                        request.events.send(GenerationEvent::Finished(reason)).ok();
                    } else if delivered {
                        continue;
                    } else {
                        batch.leave(token.sequence)?;
                    }
                    if let Some(request) = running.remove(&token.sequence) {
                        reserved -= request.reservation;
                    }
                }
            }
            Err(error) => {
                // A failed step changes nothing, so every running request fails with it and leaves.
                tracing::error!("a generation step failed: {error}");
                for (id, request) in running.drain() {
                    request
                        .events
                        .send(GenerationEvent::Failed(error.to_string()))
                        .ok();
                    batch.leave(id)?;
                }

                reserved = 0;
            }
        }
    }
}
