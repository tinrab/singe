use serde::{Deserialize, Serialize};

use crate::{head::ScoreFunction, host};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ClassificationParameters {
    /// The most labels to keep, best first, 5 by default.
    pub top_k: usize,
    /// How scores become what the output reports, or the model's own choice when `None`.
    pub function_to_apply: Option<ScoreFunction>,
}

impl Default for ClassificationParameters {
    fn default() -> Self {
        Self {
            top_k: 5,
            function_to_apply: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LabelScore {
    pub score: f32,
    pub label: String,
}

/// Returns each label's score after `function`, best first, the first `top_k`.
#[must_use]
pub fn ranked(
    scores: &[f32],
    labels: &[String],
    function: ScoreFunction,
    top_k: usize,
) -> Vec<LabelScore> {
    let mut ranked = function
        .apply(scores)
        .into_iter()
        .zip(labels)
        .map(|(score, label)| LabelScore {
            score,
            label: label.clone(),
        })
        .collect::<Vec<_>>();
    host::best_first(&mut ranked, top_k, |label| label.score);
    ranked
}
