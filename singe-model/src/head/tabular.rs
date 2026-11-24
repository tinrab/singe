//! In-context models of tables, as Mitra is: training rows and their targets, read as context, to predictions for other rows, with no training.

use std::fmt::Debug;

use crate::error::Result;

pub trait TabularClassifier: Debug {
    /// Returns each of `queries`' probabilities of the classes `0` to the largest in `labels`, with `rows`, labeled by `labels`, as context.
    fn classify(
        &self,
        rows: &[Vec<f32>],
        labels: &[i64],
        queries: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>>;
}

pub trait TabularRegressor: Debug {
    /// Returns each of `queries`' predicted target, with `rows` and their `targets` as context.
    fn regress(&self, rows: &[Vec<f32>], targets: &[f32], queries: &[Vec<f32>])
    -> Result<Vec<f32>>;
}
