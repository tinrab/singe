// Autoformer, forecasting each step's Student-T distribution of a monthly series from its time features and series id, as `verify.py` forecasts it with `transformers`.

use singe_model::task::tabular::time_series_forecasting::TimeSeriesForecastingPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TimeSeriesForecastingPipeline>("forecast");
}
