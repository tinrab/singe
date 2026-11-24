//! Singe's task API: `POST /v1/tasks/{task}` runs any task's pipeline on JSON, as `transformers`' `pipeline(task)(inputs, **parameters)` takes and returns it, and `GET /v1/tasks` lists the served tasks.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A request to `POST /v1/tasks/{task}`.
#[derive(Debug, Clone, Deserialize)]
pub struct TaskRequest {
    /// The model to run, which may be left out when one model serves the task.
    pub model: Option<String>,
    /// One input, or an array of them.
    pub inputs: Value,
    /// The pipeline's keyword arguments, as an object; left out, the pipeline uses its defaults.
    #[serde(default)]
    pub parameters: Value,
}

/// The response to a task request.
#[derive(Debug, Clone, Serialize)]
pub struct TaskResponse {
    /// The name of the model that ran.
    pub model: String,
    /// The task's id, with a Hub alias such as `ner` replaced by the task it names.
    pub task: String,
    /// One output for one input, or an array of one per input.
    pub outputs: Value,
}

/// The response to `GET /v1/tasks`.
#[derive(Debug, Clone, Serialize)]
pub struct TaskList {
    /// Always `list`.
    pub object: &'static str,
    /// Each served task once, with every model that serves it.
    pub data: Vec<ServedTask>,
}

/// A served task and the models that serve it.
#[derive(Debug, Clone, Serialize)]
pub struct ServedTask {
    /// The task's id, such as `text-classification`.
    pub task: String,
    /// The names of the models that serve the task.
    pub models: Vec<String>,
}
