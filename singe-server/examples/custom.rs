//! Serves a model of one's own, which `singe-model` does not know, with `singe-server`'s pieces: on a `Worker`, through the task API as `POST /v1/tasks/word-count`, and through an endpoint of one's own at `POST /count`, beside the models and health endpoints, in an application with its own state.
//! The model here counts words, standing in for one built on `singe`; like one, it lives on the worker's thread, so it need not be `Send`.

use std::{error::Error, rc::Rc, sync::Arc};

use axum::{
    Json, Router,
    extract::{FromRef, State},
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use singe_server::{
    api::now,
    error::{ApiError, ApiResult},
    handler::{ApiJson, health_router, model_router, task_router},
    registry::{BoxFuture, Engine, Served, ServedModel, TaskRunner},
    worker::{ModelState, Worker},
};
use tokio::net::TcpListener;

struct WordCounter {
    calls: Rc<usize>,
}

impl WordCounter {
    fn count(&mut self, text: &str) -> (usize, usize) {
        self.calls = Rc::new(*self.calls + 1);
        (text.split_whitespace().count(), *self.calls)
    }
}

#[derive(Clone)]
struct AppState {
    served: Served,
    counter: Worker<WordCounter>,
}

impl FromRef<AppState> for Served {
    fn from_ref(state: &AppState) -> Self {
        state.served.clone()
    }
}

/// The task API's way in: inputs and outputs as JSON.
struct WordCount(Worker<WordCounter>);

impl TaskRunner for WordCount {
    fn state(&self) -> ModelState {
        self.0.state()
    }

    fn run(&self, inputs: Value, _parameters: Value) -> BoxFuture<ApiResult<Value>> {
        let worker = self.0.clone();
        Box::pin(async move {
            let text = inputs
                .as_str()
                .ok_or_else(|| ApiError::invalid_field("inputs", "an input that is not a text"))?
                .to_owned();
            let (words, _) = worker.call(move |model| Ok(model.count(&text))).await?;
            Ok(json!({ "words": words }))
        })
    }
}

#[derive(Deserialize)]
struct CountRequest {
    text: String,
}

#[derive(Serialize)]
struct CountResponse {
    words: usize,
    calls: usize,
}

async fn count(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<CountRequest>,
) -> ApiResult<Json<CountResponse>> {
    let (words, calls) = state
        .counter
        .call(move |model| Ok(model.count(&request.text)))
        .await?;
    Ok(Json(CountResponse { words, calls }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let counter = Worker::spawn("word-counter", 16, || Ok(WordCounter { calls: Rc::new(0) }))?;
    let served = Served::create(vec![ServedModel {
        name: "word-counter".to_owned(),
        task: "word-count".to_owned(),
        engine: Engine::Custom(Arc::new(WordCount(counter.clone()))),
        tokenizer: None,
        created: now(),
    }])?;
    let app = Router::new()
        .route("/count", post(count))
        .merge(task_router())
        .merge(model_router())
        .merge(health_router())
        .with_state(AppState { served, counter });
    let listener = TcpListener::bind("127.0.0.1:8000").await?;
    println!("listening on http://127.0.0.1:8000");
    axum::serve(listener, app).await?;
    Ok(())
}
