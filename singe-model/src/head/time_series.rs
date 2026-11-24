//! A forecaster of time series, as `*ModelForPrediction` heads are: a series' past values to its next values, as a point forecast and quantiles.

use std::fmt::Debug;

use crate::error::Result;

/// A series' forecast, a point forecast and quantiles for each step.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    /// Each step's point forecast, as `transformers`' `mean_predictions` gives it.
    pub mean: Vec<f32>,
    /// Each step's outputs, as `transformers`' `full_predictions` gives them: the model's mean output first, then each quantile's, lowest first.
    pub full: Vec<Vec<f32>>,
}

pub trait Forecaster: Debug {
    fn forecast(&self, series: &[f32]) -> Result<Forecast>;
}
