// `time-series-forecasting`: a series' past values to its next values, a point forecast and quantiles for each step, as `transformers`' `*ModelForPrediction` returns them; `transformers` has no pipeline for it.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::Forecaster,
    model,
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct TimeSeriesForecastingPipeline {
    model: Box<dyn Forecaster>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SeriesInput {
    pub past_values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForecastOutput {
    pub mean_predictions: Vec<f32>,
    pub full_predictions: Vec<Vec<f32>>,
}

impl Pipeline for TimeSeriesForecastingPipeline {
    const TASK: Task = Task::TimeSeriesForecasting;

    type Input = SeriesInput;

    type Parameters = ();

    type Output = ForecastOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                let forecast = self.model.forecast(&input.past_values)?;
                Ok(ForecastOutput {
                    mean_predictions: forecast.mean,
                    full_predictions: forecast.full,
                })
            })
            .collect()
    }
}
