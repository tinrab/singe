//! `tabular-regression`: rows of numeric features with their targets and other rows to each other row's predicted target, by an in-context model that reads the rows with targets as context.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::TabularRegressor,
    model,
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct TabularRegressionPipeline {
    model: Box<dyn TabularRegressor>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TabularRegressionInput {
    pub training_rows: Vec<Vec<f32>>,
    pub training_targets: Vec<f32>,
    pub rows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TabularRegressionOutput {
    pub predictions: Vec<f32>,
}

impl Pipeline for TabularRegressionPipeline {
    type Input = TabularRegressionInput;

    type Parameters = ();

    type Output = TabularRegressionOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run_one(&self, input: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        Ok(TabularRegressionOutput {
            predictions: self.model.regress(
                &input.training_rows,
                &input.training_targets,
                &input.rows,
            )?,
        })
    }
}
