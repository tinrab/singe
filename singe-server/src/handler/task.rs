//! The task API: `POST /v1/tasks/{task}` runs any served task on JSON, and `GET /v1/tasks` lists them.
//! A text-generation model generates through its engine, so its task requests batch with its chat and completion requests.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
};
use futures::future::try_join_all;
use serde_json::Value;
use singe::{generation::SequenceOptions, tensor::SamplingOptions};
use singe_model::task::{
    self, Task,
    outputs::GeneratedSequence,
    text::text_generation::{TextGenerationInput, TextGenerationParameters},
};

use crate::{
    api::task::{ServedTask, TaskList, TaskRequest, TaskResponse},
    error::{ApiError, ApiResult},
    generation::{engine::TextEngine, format::TextFormat, output::TextGeneration},
    handler::{ApiJson, blocking},
    registry::{Engine, Served},
    text::Detokenizer,
};

/// Serves `POST /v1/tasks/{task}`: runs `inputs` with `parameters` on the model `model` names, or on the one model serving `task`, which may be a Hub alias, and returns the pipeline's JSON outputs.
///
/// # Errors
///
/// Returns a 404 error if no model serves the task, 422 if the named model serves another, 400 if several serve it and the request names none or the inputs or parameters are invalid, 429 when the model's queue is full, and 503 while it loads.
pub async fn run(
    State(served): State<Served>,
    Path(task): Path<String>,
    ApiJson(request): ApiJson<TaskRequest>,
) -> ApiResult<Json<TaskResponse>> {
    // A Hub alias, such as `ner`, names its task.
    let task = Task::from_id(&task).map_or(task, |task| task.id().to_owned());
    let model = served.for_task(&task, request.model.as_deref())?;
    let outputs = match &model.engine {
        Engine::Pipeline(worker) => {
            let (inputs, parameters) = (request.inputs, request.parameters);
            worker
                .call(move |pipeline| Ok(pipeline.run_json(&inputs, &parameters)?))
                .await?
        }
        Engine::Custom(runner) => runner.run(request.inputs, request.parameters).await?,
        Engine::Tensors(_) | Engine::Realtime(_) => {
            return Err(ApiError::wrong_task(
                &model.name,
                &model.task,
                &format!("/v1/tasks/{task}"),
            ));
        }
        Engine::Text { engine, format } => {
            generate(engine, format, request.inputs, &request.parameters).await?
        }
    };
    Ok(Json(TaskResponse {
        model: model.name.clone(),
        task,
        outputs,
    }))
}

/// Serves `GET /v1/tasks`: each served task with the names of the models serving it.
pub async fn list(State(served): State<Served>) -> Json<TaskList> {
    let mut data = Vec::<ServedTask>::new();
    for model in served.models() {
        match data.iter_mut().find(|served| served.task == model.task) {
            Some(served) => served.models.push(model.name.clone()),
            None => data.push(ServedTask {
                task: model.task.clone(),
                models: vec![model.name.clone()],
            }),
        }
    }
    Json(TaskList {
        object: "list",
        data,
    })
}

/// `text-generation`'s pipeline, on the engine: each input's text and tokens, as the pipeline gives them.
async fn generate(
    engine: &TextEngine,
    format: &Arc<TextFormat>,
    inputs: Value,
    parameters: &Value,
) -> ApiResult<Value> {
    let parameters = task::parameters::<TextGenerationParameters>(parameters)?;
    // An array is one input when it reads as one, a conversation, as the pipeline reads it.
    let (inputs, many) = match (
        &inputs,
        serde_json::from_value::<TextGenerationInput>(inputs.clone()),
    ) {
        (Value::Array(items), Err(_)) => (
            items
                .iter()
                .map(|item| {
                    serde_json::from_value(item.clone())
                        .map_err(|error| ApiError::invalid_field("inputs", error.to_string()))
                })
                .collect::<ApiResult<Vec<TextGenerationInput>>>()?,
            true,
        ),
        (_, input) => (
            vec![input.map_err(|error| ApiError::invalid_field("inputs", error.to_string()))?],
            false,
        ),
    };
    let rendering = Arc::clone(format);
    let arguments = parameters.chat_template_kwargs.clone();
    let prompts = blocking(move || {
        inputs
            .iter()
            .map(|input| match input {
                TextGenerationInput::Text(text) => rendering.encode(text, false),
                TextGenerationInput::Chat(messages) => {
                    let text = rendering.render(messages, true, &arguments)?;
                    rendering.encode(&text, false)
                }
            })
            .collect::<ApiResult<Vec<_>>>()
    })
    .await?;

    // A sequence's sampling runs in the step it shares with others, so what would fail it is refused here.
    if !(parameters.repetition_penalty.is_finite() && parameters.repetition_penalty > 0.0) {
        return Err(ApiError::invalid_field(
            "parameters",
            "a repetition_penalty that is not positive",
        ));
    }

    if parameters.do_sample {
        if !(parameters.temperature.is_finite() && parameters.temperature > 0.0) {
            return Err(ApiError::invalid_field(
                "parameters",
                "a temperature that is not positive",
            ));
        }

        if !(parameters.top_p > 0.0 && parameters.top_p <= 1.0) {
            return Err(ApiError::invalid_field(
                "parameters",
                "a top_p outside (0, 1]",
            ));
        }

        if parameters.top_k == Some(0) {
            return Err(ApiError::invalid_field("parameters", "a top_k of zero"));
        }
    }

    let sampling = parameters.do_sample.then(|| {
        SamplingOptions::new()
            .with_temperature(parameters.temperature)
            .with_top_k(parameters.top_k.and_then(|top_k| i64::try_from(top_k).ok()))
            .with_top_p(parameters.top_p)
    });
    let mut generations = prompts
        .into_iter()
        .zip(0_u64..)
        .map(|(prompt, index)| {
            // Each prompt takes the seed after the last's, as image generation seeds its images.
            let options = SequenceOptions {
                max_new_tokens: parameters.max_new_tokens,
                stop_tokens: format.stop_tokens.clone(),
                sampling,
                seed: Some(parameters.seed.wrapping_add(index)),
                penalties: parameters.penalties(),
                ..SequenceOptions::default()
            };
            let detokenizer = Detokenizer::new(Arc::clone(&format.tokenizer), true, &prompt)?;
            Ok(TextGeneration::new(
                engine.submit(prompt, options)?,
                detokenizer,
                Vec::new(),
                format.stop_tokens.clone(),
                0,
            ))
        })
        .collect::<ApiResult<Vec<_>>>()?;
    try_join_all(generations.iter_mut().map(TextGeneration::collect)).await?;
    let mut outputs = generations
        .iter()
        .map(|generation| {
            let tokens = generation.tokens().to_vec();
            let output = GeneratedSequence {
                generated_text: format.tokenizer.decode(&tokens, true)?,
                generated_token_ids: tokens,
            };
            serde_json::to_value(output)
                .map_err(|error| ApiError::internal(format!("failed to write an output: {error}")))
        })
        .collect::<ApiResult<Vec<_>>>()?;
    if many {
        return Ok(Value::Array(outputs));
    }
    outputs
        .pop()
        .ok_or_else(|| ApiError::internal("no output for the input"))
}
