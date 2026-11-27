//! A forecaster of time series, as `*ModelForPrediction` heads are: a series' past values, and any features of its steps, to its next values, as a point forecast and quantiles, as each channel's points, or as a distribution per step.

use std::fmt::Debug;

use serde::Deserialize;

use crate::error::{Error, Result};

/// A series' past values: one value per step, or one value per channel at each step, `[T][C]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum PastSeries {
    Univariate(Vec<f32>),
    Multivariate(Vec<Vec<f32>>),
}

/// A series to forecast: its past values and, for models that read them, each past and future step's time features and the series' static features, as `transformers`' `generate` takes them, and its future values, as `forward` takes them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Series {
    pub past_values: PastSeries,
    #[serde(default)]
    pub past_time_features: Vec<Vec<f32>>,
    #[serde(default)]
    pub future_time_features: Vec<Vec<f32>>,
    #[serde(default)]
    pub static_categorical_features: Vec<i64>,
    #[serde(default)]
    pub static_real_features: Vec<f32>,
    /// The values of the steps to forecast, for a model that then gives each step's distribution after the real earlier steps, as `forward` does with `future_values`.
    #[serde(default)]
    pub future_values: Option<Vec<f32>>,
    /// Which past values were observed, 1, or are missing, 0, shaped as `past_values`; every value is observed when `None`.
    #[serde(default)]
    pub past_observed_mask: Option<PastSeries>,
}

impl Series {
    /// Refuses a series with static real features or future values, for a model that reads neither.
    ///
    /// # Errors
    ///
    /// Returns an error if the series has static real features or future values.
    pub fn refuse_extras(&self, model: &str) -> Result<()> {
        if self.static_real_features.is_empty() && self.future_values.is_none() {
            Ok(())
        } else {
            Err(Error::unsupported(format!(
                "{model} with static real features or future values"
            )))
        }
    }

    /// Refuses a series with missing values, for a model that reads every value as observed.
    ///
    /// # Errors
    ///
    /// Returns an error if the series marks any past value missing.
    pub fn refuse_missing(&self, model: &str) -> Result<()> {
        let observed = |value: &f32| (value - 1.0).abs() < f32::EPSILON;
        let complete = match &self.past_observed_mask {
            None => true,
            Some(PastSeries::Univariate(mask)) => mask.iter().all(observed),
            Some(PastSeries::Multivariate(mask)) => mask.iter().flatten().all(observed),
        };
        if complete {
            Ok(())
        } else {
            Err(Error::unsupported(format!(
                "{model} with missing past values"
            )))
        }
    }
}

/// A series' forecast.
#[derive(Debug, Clone, PartialEq)]
pub enum Forecast {
    /// Each step's point forecast, as `transformers`' `mean_predictions` gives it, and each step's outputs, as its `full_predictions` gives them: the model's mean output first, then each quantile's, lowest first.
    Quantiles { mean: Vec<f32>, full: Vec<Vec<f32>> },
    /// Each step's value of each channel, `[H][C]`, as `prediction_outputs` gives them.
    Points(Vec<Vec<f32>>),
    /// Each step's Student-T distribution, its degrees of freedom, location, and scale, mapped back to the series' scale, whose samples `generate` draws.
    StudentT {
        df: Vec<f32>,
        loc: Vec<f32>,
        scale: Vec<f32>,
    },
    /// Sampled paths, `[P][H]`, as `generate`'s `sequences` gives them.
    Samples(Vec<Vec<f32>>),
}

/// The paths a distributional forecast samples, as `num_parallel_samples` asks, and the seed of their draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathSampling {
    pub count: usize,
    pub seed: u64,
}

pub trait Forecaster: Debug {
    /// Forecasts `series`, as sampled paths when `draws` asks for them.
    ///
    /// # Errors
    ///
    /// Returns an error if the series does not fit the model, if the model forecasts no distribution to draw from, or if a step fails.
    fn forecast(&self, series: &Series, draws: Option<PathSampling>) -> Result<Forecast>;
}

/// Refuses `draws`, for a forecaster without a distribution to sample.
///
/// # Errors
///
/// Returns an error if any draws are asked for.
pub fn refuse_draws(draws: Option<PathSampling>, model: &str) -> Result<()> {
    match draws {
        None => Ok(()),
        Some(_) => Err(Error::unsupported(format!(
            "sampled paths from {model}, whose forecast is no distribution"
        ))),
    }
}
