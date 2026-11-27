//! `time-series-forecasting`: a series' past values, and any features of its steps, to its next values, as `transformers`' `*ModelForPrediction` returns them: a point forecast and quantiles for each step of one series, each channel's values for each step of several, or each step's distribution, or paths sampled from it as `generate` samples them; `transformers` has no pipeline for it.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::{Draws, Forecast, Forecaster, Series},
    model,
    source::{LoadOptions, Source},
    task::Pipeline,
};

#[derive(Debug)]
pub struct TimeSeriesForecastingPipeline {
    model: Box<dyn Forecaster>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ForecastOutput {
    Quantiles {
        mean_predictions: Vec<f32>,
        full_predictions: Vec<Vec<f32>>,
    },
    Points {
        prediction_outputs: Vec<Vec<f32>>,
    },
    StudentT {
        df: Vec<f32>,
        loc: Vec<f32>,
        scale: Vec<f32>,
    },
    Samples {
        sequences: Vec<Vec<f32>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct TimeSeriesForecastingParameters {
    /// The paths to sample from a distributional forecast, as `generate`'s `num_parallel_samples`, or `None` for the distributions themselves.
    pub num_parallel_samples: Option<usize>,
    pub seed: u64,
}

impl Pipeline for TimeSeriesForecastingPipeline {
    type Input = Series;

    type Parameters = TimeSeriesForecastingParameters;

    type Output = ForecastOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let draws = parameters.num_parallel_samples.map(|count| Draws {
            count,
            seed: parameters.seed,
        });
        Ok(match self.model.forecast(input, draws)? {
            Forecast::Quantiles { mean, full } => ForecastOutput::Quantiles {
                mean_predictions: mean,
                full_predictions: full,
            },
            Forecast::Points(points) => ForecastOutput::Points {
                prediction_outputs: points,
            },
            Forecast::StudentT { df, loc, scale } => ForecastOutput::StudentT { df, loc, scale },
            Forecast::Samples(sequences) => ForecastOutput::Samples { sequences },
        })
    }
}
