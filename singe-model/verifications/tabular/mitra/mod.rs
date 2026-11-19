// Mitra, classifying wine and predicting diabetes progression from labeled rows, as `verify.py` predicts them with AutoGluon.

use singe_model::task::tabular::{
    tabular_classification::TabularClassificationPipeline,
    tabular_regression::TabularRegressionPipeline,
};

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TabularClassificationPipeline>("classification");
    session.check::<TabularRegressionPipeline>("regression");
}
