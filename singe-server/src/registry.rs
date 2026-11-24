//! The models a server serves, by name: each with its task, which decides the endpoints it answers, and the engine that runs it.

use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use serde_json::Value;
use singe_model::{
    processor::text::TextTokenizer,
    task::{AnyPipeline, Task},
};

use crate::{
    error::{ApiError, ApiResult},
    generation::{engine::TextEngine, format::TextFormat},
    worker::{ModelState, Worker},
};

/// A boxed future a [`TaskRunner`] returns.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// A model of a task of one's own, served through the task API.
pub trait TaskRunner: Send + Sync + 'static {
    /// Returns whether the model can take requests, as `/health` and `/v1/models` report it.
    fn state(&self) -> ModelState;

    /// Runs the model on `inputs`, one or an array of them, with `parameters`.
    fn run(&self, inputs: Value, parameters: Value) -> BoxFuture<ApiResult<Value>>;
}

/// What runs a served model.
pub enum Engine {
    /// A causal language model generating with continuous batching.
    Text {
        /// The engine thread's handle.
        engine: TextEngine,
        /// How chat messages become the prompt and output becomes text.
        format: Arc<TextFormat>,
    },
    /// Any `singe-model` pipeline, run one request at a time.
    Pipeline(Worker<Box<dyn AnyPipeline>>),
    /// A [`TaskRunner`] of one's own, served through the task API.
    Custom(Arc<dyn TaskRunner>),
}

/// A model the server serves, under the name requests use.
pub struct ServedModel {
    /// The name requests use in `model`.
    pub name: String,
    /// The task's id, as the task API's path names it.
    pub task: String,
    /// What runs the model.
    pub engine: Engine,
    /// The model's tokenizer, for counting tokens and for `/tokenize`, when it has one.
    pub tokenizer: Option<Arc<TextTokenizer>>,
    /// The model's creation time in seconds since the Unix epoch, as `/v1/models` lists it.
    pub created: u64,
}

impl ServedModel {
    /// Returns whether the model can take requests.
    #[must_use]
    pub fn state(&self) -> ModelState {
        match &self.engine {
            Engine::Text { engine, .. } => engine.state(),
            Engine::Pipeline(worker) => worker.state(),
            Engine::Custom(runner) => runner.state(),
        }
    }

    /// Waits until the model has loaded or failed to.
    pub async fn loaded(&self) -> ModelState {
        match &self.engine {
            Engine::Text { engine, .. } => engine.loaded().await,
            Engine::Pipeline(worker) => worker.loaded().await,
            Engine::Custom(runner) => runner.state(),
        }
    }

    /// Returns the endpoints that serve the model, as [`endpoints`] gives them for its task.
    #[must_use]
    pub fn endpoints(&self) -> Vec<&'static str> {
        endpoints(&self.task)
    }

    /// Returns the most tokens a sequence holds, prompt included, for a text generator, and `None` for any other model.
    #[must_use]
    pub fn max_model_len(&self) -> Option<usize> {
        match &self.engine {
            Engine::Text { engine, .. } => Some(engine.limits().max_model_len),
            Engine::Pipeline(_) | Engine::Custom(_) => None,
        }
    }

    /// Returns the model's text engine and format, which `endpoint` needs.
    ///
    /// # Errors
    ///
    /// Returns a 422 [`ApiError::wrong_task`] if the model is not a text generator.
    pub fn text(&self, endpoint: &str) -> ApiResult<(&TextEngine, &Arc<TextFormat>)> {
        match &self.engine {
            Engine::Text { engine, format } => Ok((engine, format)),
            Engine::Pipeline(_) | Engine::Custom(_) => {
                Err(ApiError::wrong_task(&self.name, &self.task, endpoint))
            }
        }
    }

    /// Returns the model's pipeline worker, when the model runs one of `tasks`, which `endpoint` takes.
    ///
    /// # Errors
    ///
    /// Returns a 422 [`ApiError::wrong_task`] if the model runs no pipeline or a task outside `tasks`.
    pub fn pipeline(
        &self,
        tasks: &[Task],
        endpoint: &str,
    ) -> ApiResult<&Worker<Box<dyn AnyPipeline>>> {
        match &self.engine {
            Engine::Pipeline(worker) if tasks.iter().any(|task| task.id() == self.task) => {
                Ok(worker)
            }
            _ => Err(ApiError::wrong_task(&self.name, &self.task, endpoint)),
        }
    }

    /// Returns the model's tokenizer.
    ///
    /// # Errors
    ///
    /// Returns a 400 error if the model has none.
    pub fn tokenizer(&self) -> ApiResult<&Arc<TextTokenizer>> {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| ApiError::invalid(format!("model {} has no tokenizer", self.name)))
    }
}

/// Returns the endpoints that serve `task`, ending with the task API every task has.
#[must_use]
pub fn endpoints(task: &str) -> Vec<&'static str> {
    let served: &[&str] = match Task::from_id(task) {
        Ok(Task::TextGeneration) => &[
            "/v1/chat/completions",
            "/v1/completions",
            "/tokenize",
            "/detokenize",
        ],
        Ok(Task::SentenceSimilarity) => &["/v1/embeddings"],
        Ok(Task::TextRanking) => &["/v1/rerank"],
        Ok(Task::AutomaticSpeechRecognition) => &["/v1/audio/transcriptions"],
        Ok(Task::TextToAudio) => &["/v1/audio/speech"],
        Ok(Task::TextToImage) => &["/v1/images/generations"],
        Ok(Task::ImageTextToImage) => &["/v1/images/edits"],
        _ => &[],
    };
    let mut endpoints = served.to_vec();
    endpoints.push("/v1/tasks/<task>");
    endpoints
}

/// Two models given to [`Served::create`] share the name it holds.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("two models named {0}")]
pub struct DuplicateModel(pub String);

struct ServedModels {
    models: Vec<Arc<ServedModel>>,
    stopping: AtomicBool,
}

/// The served models, as handlers take them for their state; clones share one set.
#[derive(Clone)]
pub struct Served {
    inner: Arc<ServedModels>,
}

impl Served {
    /// Creates the set of served `models`, none of them stopping.
    ///
    /// # Errors
    ///
    /// Returns an error if two of `models` share a name.
    pub fn create(models: Vec<ServedModel>) -> Result<Self, DuplicateModel> {
        for (index, model) in models.iter().enumerate() {
            if models[..index].iter().any(|other| other.name == model.name) {
                return Err(DuplicateModel(model.name.clone()));
            }
        }
        Ok(Self {
            inner: Arc::new(ServedModels {
                models: models.into_iter().map(Arc::new).collect(),
                stopping: AtomicBool::new(false),
            }),
        })
    }

    /// Returns the models in the order they were given.
    #[must_use]
    pub fn models(&self) -> &[Arc<ServedModel>] {
        &self.inner.models
    }

    /// Returns the model named `name`.
    ///
    /// # Errors
    ///
    /// Returns a 404 error if no model has that name.
    pub fn get(&self, name: &str) -> ApiResult<Arc<ServedModel>> {
        self.inner
            .models
            .iter()
            .find(|model| model.name == name)
            .cloned()
            .ok_or_else(|| ApiError::not_found(format!("no model {name}")))
    }

    /// Returns the model `name` names, or else the one model serving `task`.
    ///
    /// # Errors
    ///
    /// Returns a 404 error if `name` names no model or no model serves `task`, a 422 error if the named model serves another task, and a 400 error if `name` is `None` and several models serve `task`.
    pub fn for_task(&self, task: &str, name: Option<&str>) -> ApiResult<Arc<ServedModel>> {
        if let Some(name) = name {
            let model = self.get(name)?;
            if model.task != task {
                return Err(ApiError::wrong_task(
                    &model.name,
                    &model.task,
                    &format!("/v1/tasks/{task}"),
                ));
            }
            return Ok(model);
        }

        let mut serving = self.inner.models.iter().filter(|model| model.task == task);
        match (serving.next(), serving.next()) {
            (Some(model), None) => Ok(Arc::clone(model)),
            (None, _) => Err(ApiError::not_found(format!("no model serves {task}"))),
            (Some(_), Some(_)) => Err(ApiError::invalid_field(
                "model",
                format!("several models serve {task}, so a request names one"),
            )),
        }
    }

    /// Returns whether every model is ready and the server is not stopping.
    #[must_use]
    pub fn ready(&self) -> bool {
        !self.inner.stopping.load(Ordering::Relaxed)
            && self
                .inner
                .models
                .iter()
                .all(|model| model.state() == ModelState::Ready)
    }

    /// Marks the server as stopping, so `/health` fails while running requests finish.
    pub fn stop(&self) {
        self.inner.stopping.store(true, Ordering::Relaxed);
    }
}
