use serde::{Deserialize, Serialize};

use crate::{head::ScoreFunction, host};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ClassificationParameters {
    // The labels to keep from the best, at most every one.
    pub top_k: usize,
    // How scores become what the output reports, or the model's own choice when absent.
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

// Each label's score after `function`, best first, the first `top_k`.
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
