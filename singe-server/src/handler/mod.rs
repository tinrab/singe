//! The handlers, and routers of them by endpoint family; each takes the served models from any state they can be had from, so an application mounts what it wants under its own state.

pub mod audio;
pub mod chat;
pub mod completion;
pub mod embedding;
pub mod health;
pub mod image;
pub mod model;
pub mod rerank;
pub mod stream;
pub mod task;
pub mod tensor;
pub mod tokenize;

use axum::{
    Json, Router,
    extract::{FromRef, FromRequest, Request, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use serde::de::DeserializeOwned;
use singe_model::task::{AnyPipeline, Pipeline};
use tokio::task::spawn_blocking;

use crate::{
    error::{ApiError, ApiResult},
    generation::engine::TextLimits,
    registry::Served,
};

/// Returns a router of every endpoint, with no API keys, CORS, or body limit of its own.
pub fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .merge(chat_router())
        .merge(embedding_router())
        .merge(audio_router())
        .merge(image_router())
        .merge(task_router())
        .merge(tensor_router())
        .merge(model_router())
        .merge(tokenize_router())
        .merge(health_router())
}

/// Returns a router of `/v1/chat/completions` and `/v1/completions`.
pub fn chat_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v1/chat/completions", post(chat::create))
        .route("/v1/completions", post(completion::create))
}

/// Returns a router of `/v1/embeddings` and `/v1/rerank`.
pub fn embedding_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v1/embeddings", post(embedding::create))
        .route("/v1/rerank", post(rerank::create))
}

/// Returns a router of `/v1/audio/transcriptions` and `/v1/audio/speech`.
pub fn audio_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v1/audio/transcriptions", post(audio::transcribe))
        .route("/v1/audio/speech", post(audio::speak))
}

/// Returns a router of `/v1/images/generations` and `/v1/images/edits`.
pub fn image_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v1/images/generations", post(image::generate))
        .route("/v1/images/edits", post(image::edit))
}

/// Returns a router of `/v1/tasks` and `/v1/tasks/{task}`.
pub fn task_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v1/tasks", get(task::list))
        .route("/v1/tasks/{task}", post(task::run))
}

pub fn tensor_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v2/models/{model}/infer", post(tensor::infer))
        .route("/v2/models/{model}", get(tensor::metadata))
}

/// Returns a router of `/v1/models` and `/v1/models/{model}`.
pub fn model_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/v1/models", get(model::list))
        .route("/v1/models/{model}", get(model::get))
}

/// Returns a router of `/tokenize` and `/detokenize`.
pub fn tokenize_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
}

/// Returns a router of `/health` and `/version`, which [`crate::server::app`] leaves open when it asks for keys elsewhere.
pub fn health_router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Served: FromRef<S>,
{
    Router::new()
        .route("/health", get(health::health))
        .route("/version", get(health::version))
}

/// A JSON body extractor whose rejection is an [`ApiError`] in OpenAI's shape: 413 for a body over the limit, 415 without a JSON content type, and 400 otherwise.
pub struct ApiJson<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequest<S> for ApiJson<T> {
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(rejected(&rejection)),
        }
    }
}

fn rejected(rejection: &JsonRejection) -> ApiError {
    match rejection.status() {
        StatusCode::PAYLOAD_TOO_LARGE => ApiError::too_large(rejection.body_text()),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ApiError::unsupported_media(rejection.body_text()),
        _ => ApiError::invalid(rejection.body_text()),
    }
}

/// Returns the request's id: its `X-Request-Id`, or else a new UUID.
#[must_use]
pub fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map_or_else(|| uuid::Uuid::new_v4().simple().to_string(), str::to_owned)
}

/// Runs `work` on Tokio's blocking threads, as tokenizing and decoding media do.
///
/// # Errors
///
/// Returns the error `work` returns, and a 500 error if it panics.
pub async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> ApiResult<T> + Send + 'static,
) -> ApiResult<T> {
    spawn_blocking(work)
        .await
        .map_err(|_| ApiError::internal("a blocking task panicked"))?
}

/// Returns the pipeline as `P`, which its task's endpoints run.
///
/// # Errors
///
/// Returns a 500 error if the pipeline is not a `P`.
pub fn downcast<P: Pipeline + 'static>(pipeline: &dyn AnyPipeline) -> ApiResult<&P> {
    pipeline
        .downcast::<P>()
        .ok_or_else(|| ApiError::internal("a model whose pipeline is not its task's"))
}

/// Returns the tokens a completion of a prompt of `prompt` tokens generates: what it asks, or what the context leaves.
///
/// # Errors
///
/// Returns a 400 error if the prompt fills `max_model_len` or `requested` is zero.
pub fn max_tokens(requested: Option<usize>, prompt: usize, limits: TextLimits) -> ApiResult<usize> {
    if prompt >= limits.max_model_len {
        return Err(ApiError::invalid(format!(
            "a prompt of {prompt} tokens fills the model's {} tokens",
            limits.max_model_len
        )));
    }
    match requested {
        Some(0) => Err(ApiError::invalid_field(
            "max_tokens",
            "a completion of no tokens",
        )),
        Some(tokens) => Ok(tokens),
        None => Ok(limits.max_model_len - prompt),
    }
}

/// Returns a random seed for a request that names none.
#[must_use]
pub fn fresh_seed() -> u64 {
    uuid::Uuid::new_v4().as_u64_pair().0
}
