// `tabular-classification`: labeled rows of numeric features and other rows to each other row's class probabilities and likeliest class, by an in-context model that reads the labeled rows as context.

use serde::{Deserialize, Serialize};
use singe::{Error as SingeError, Result as SingeResult};

use crate::{
    error::Result,
    head::{TabularClassifier, argmax},
    model,
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct TabularClassificationPipeline {
    model: Box<dyn TabularClassifier>,
}

// Rows of features with their labels, `0` to the largest, and the rows to classify.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TabularClassificationInput {
    pub training_rows: Vec<Vec<f32>>,
    pub training_labels: Vec<i64>,
    pub rows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TabularClassificationOutput {
    pub labels: Vec<i64>,
    pub probabilities: Vec<Vec<f32>>,
}

impl Pipeline for TabularClassificationPipeline {
    type Input = TabularClassificationInput;

    type Parameters = ();

    type Output = TabularClassificationOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run_one(&self, input: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let probabilities =
            self.model
                .classify(&input.training_rows, &input.training_labels, &input.rows)?;
        let labels = probabilities
            .iter()
            .map(|row| {
                let best = argmax(row.iter().copied().enumerate()).map_or(0, |(index, _)| index);
                i64::try_from(best).map_err(|_| SingeError::out_of_range("class"))
            })
            .collect::<SingeResult<_>>()?;
        Ok(TabularClassificationOutput {
            labels,
            probabilities,
        })
    }
}
