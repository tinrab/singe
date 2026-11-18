// A forecaster of time series, as `*ModelForPrediction` heads are: a series' past values to its next values, as a point forecast and quantiles.

use std::fmt::Debug;

use crate::error::Result;

// A series' forecast: each step's point forecast, and each step's outputs, the point forecast's own first and then each quantile's, lowest first.
#[derive(Debug, Clone, PartialEq)]
pub struct Forecast {
    pub mean: Vec<f32>,
    pub full: Vec<Vec<f32>>,
}

pub trait Forecaster: Debug {
    fn forecast(&self, series: &[f32]) -> Result<Forecast>;
}
