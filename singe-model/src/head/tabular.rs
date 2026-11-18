// In-context models of tables, as Mitra is: training rows and their targets, read as context, to predictions for other rows, with no training.

use std::fmt::Debug;

use crate::error::Result;

pub trait TabularClassifier: Debug {
    // Each of `queries`' probability of each class `labels` holds, `0` to its largest, from `rows` labeled `labels`.
    fn classify(
        &self,
        rows: &[Vec<f32>],
        labels: &[i64],
        queries: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>>;
}

pub trait TabularRegressor: Debug {
    // Each of `queries`' predicted target, from `rows` with targets `targets`.
    fn regress(&self, rows: &[Vec<f32>], targets: &[f32], queries: &[Vec<f32>])
    -> Result<Vec<f32>>;
}
