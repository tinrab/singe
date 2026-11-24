//! `GET /health`: 200 once every model is loaded, 503 while any loads or the server stops; `GET /version`.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Map, Value, json};

use crate::{registry::Served, worker::ModelState};

/// Serves `GET /health`: each model's state as `{"models": {name: state}}`, with 200 when every model is ready and 503 while any loads or failed, or the server stops.
pub async fn health(State(served): State<Served>) -> Response {
    let states = served
        .models()
        .iter()
        .map(|model| {
            let state = match model.state() {
                ModelState::Loading => "loading".to_owned(),
                ModelState::Ready => "ready".to_owned(),
                ModelState::Failed(error) => format!("failed: {error}"),
            };
            (model.name.clone(), Value::String(state))
        })
        .collect::<Map<_, _>>();
    let status = if served.ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(json!({ "models": states }))).into_response()
}

/// Serves `GET /version`: the crate's version as `{"version": ...}`.
pub async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}
