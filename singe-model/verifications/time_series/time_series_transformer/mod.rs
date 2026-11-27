// The time series transformer, forecasting each step's Student-T distribution of a monthly series, once after its real future values and once feeding back each step's location, as `verify.py` forecasts it with `transformers`.

use singe_model::task::tabular::time_series_forecasting::TimeSeriesForecastingPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TimeSeriesForecastingPipeline>("teacher_forced");
    session.check::<TimeSeriesForecastingPipeline>("forecast");
}
