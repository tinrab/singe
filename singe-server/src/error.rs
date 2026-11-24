//! Errors as OpenAI's API returns them: a status, and `{"error": {"message", "type", "param", "code"}}`.

use std::error::Error as StdError;

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use singe_model::Error as ModelError;

/// The result of a handler, or of work a handler waits on.
pub type ApiResult<T> = Result<T, ApiError>;

/// An error a request ends in, which responds with `status` and OpenAI's error body.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    /// The response's HTTP status.
    pub status: StatusCode,
    /// The body's `message`.
    pub message: String,
    /// The request field the error is about.
    pub param: Option<String>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    param: Option<&'a str>,
    code: u16,
}

impl ApiError {
    /// Creates an error with `status` and `message`, about no field.
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            param: None,
        }
    }

    /// Creates a 400 Bad Request error.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    /// Creates a 400 Bad Request error about the request field `param`.
    pub fn invalid_field(param: &str, message: impl Into<String>) -> Self {
        Self {
            param: Some(param.to_owned()),
            ..Self::invalid(message)
        }
    }

    /// Creates a 400 Bad Request error for a field the server reads but does not support yet, set to other than its default.
    #[must_use]
    pub fn unsupported_field(param: &str) -> Self {
        Self::invalid_field(param, format!("`{param}` is not supported"))
    }

    /// Creates a 401 Unauthorized error for a missing or rejected API key.
    #[must_use]
    pub fn unauthorized() -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "a missing or invalid api key")
    }

    /// Creates a 404 Not Found error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }

    /// Creates a 413 Payload Too Large error.
    pub fn too_large(message: impl Into<String>) -> Self {
        Self::new(StatusCode::PAYLOAD_TOO_LARGE, message)
    }

    /// Creates a 415 Unsupported Media Type error.
    pub fn unsupported_media(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNSUPPORTED_MEDIA_TYPE, message)
    }

    /// Creates a 422 Unprocessable Entity error for a model asked for an endpoint its task does not serve.
    #[must_use]
    pub fn wrong_task(model: &str, task: &str, endpoint: &str) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("model {model} serves {task}, which {endpoint} does not take"),
        )
    }

    /// Creates a 429 Too Many Requests error for a model whose queue is full.
    #[must_use]
    pub fn busy(model: &str) -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            format!("model {model} has too many queued requests"),
        )
    }

    /// Creates a 503 Service Unavailable error for a model that is still loading.
    #[must_use]
    pub fn loading(model: &str) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("model {model} is loading"),
        )
    }

    /// Creates a 500 Internal Server Error, which the response logs.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    fn kind(&self) -> &'static str {
        match self.status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::SERVICE_UNAVAILABLE => "service_unavailable_error",
            status if status.is_server_error() => "server_error",
            _ => "invalid_request_error",
        }
    }

    /// Returns the error as OpenAI's JSON body, which a stream also sends as its last event.
    #[must_use]
    pub fn body(&self) -> serde_json::Value {
        serde_json::to_value(ErrorBody {
            error: ErrorDetail {
                message: &self.message,
                kind: self.kind(),
                param: self.param.as_deref(),
                code: self.status.as_u16(),
            },
        })
        .unwrap_or_default()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::error!(status = self.status.as_u16(), "{}", self.message);
        }
        (self.status, Json(self.body())).into_response()
    }
}

/// A model's error is the request's fault when it rejects what the request gave it, and the server's otherwise.
impl From<ModelError> for ApiError {
    fn from(error: ModelError) -> Self {
        let message = error_chain(&error);
        match error {
            ModelError::JsonValue { .. }
            | ModelError::InvalidInput(_)
            | ModelError::Media(_)
            | ModelError::Template(_)
            | ModelError::UnsupportedConfig(_) => Self::invalid(message),
            _ => Self::internal(message),
        }
    }
}

impl From<singe::Error> for ApiError {
    fn from(error: singe::Error) -> Self {
        Self::internal(error_chain(&error))
    }
}

/// An error's message followed by its sources', as one line.
pub(crate) fn error_chain(error: &dyn StdError) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}
