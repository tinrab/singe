//! `GET /v1/models` and `GET /v1/models/{model}`.

use axum::{
    Json,
    extract::{Path, State},
};

use crate::{
    api::model::{ModelList, ModelObject},
    error::ApiResult,
    registry::{Served, ServedModel},
    worker::ModelState,
};

/// Serves `GET /v1/models`: every served model, as OpenAI's model list with each model's task, endpoints, and readiness.
pub async fn list(State(served): State<Served>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: served.models().iter().map(|model| object(model)).collect(),
    })
}

/// Serves `GET /v1/models/{model}`: the model named `model`, or 404 if none is.
pub async fn get(
    State(served): State<Served>,
    Path(name): Path<String>,
) -> ApiResult<Json<ModelObject>> {
    let model = served.get(&name)?;
    Ok(Json(object(&model)))
}

fn object(model: &ServedModel) -> ModelObject {
    ModelObject {
        id: model.name.clone(),
        object: "model",
        created: model.created,
        owned_by: "singe",
        task: model.task.clone(),
        endpoints: model.endpoints(),
        max_model_len: model.max_model_len(),
        ready: model.state() == ModelState::Ready,
    }
}
