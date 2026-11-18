// `tabular-classification`: labeled rows of numeric features and other rows to each other row's class probabilities and likeliest class, by an in-context model that reads the labeled rows as context.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::{TabularClassifier, argmax},
    model,
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
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
    const TASK: Task = Task::TabularClassification;

    type Input = TabularClassificationInput;

    type Parameters = ();

    type Output = TabularClassificationOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                let probabilities = self.model.classify(
                    &input.training_rows,
                    &input.training_labels,
                    &input.rows,
                )?;
                let labels = probabilities
                    .iter()
                    .map(|row| argmax(row).map_or(0, |(index, _)| index))
                    .map(|index| i64::try_from(index).unwrap_or(0))
                    .collect();
                Ok(TabularClassificationOutput {
                    labels,
                    probabilities,
                })
            })
            .collect()
    }
}
