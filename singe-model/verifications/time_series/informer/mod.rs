// Informer, forecasting each step's Student-T distribution of a monthly series, once after its real future values and once feeding back each step's location, as `verify.py` forecasts it with `transformers`.
// Both sides read every key once where ProbSparse attention would draw them at random.

use singe_model::task::tabular::time_series_forecasting::TimeSeriesForecastingPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TimeSeriesForecastingPipeline>("teacher_forced");
    session.check::<TimeSeriesForecastingPipeline>("forecast");
}
