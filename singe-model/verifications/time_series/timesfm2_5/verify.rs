// TimesFM 2.5, forecasting a series, as `verify.py` forecasts it with `transformers`.

use singe::DataType;
use singe_model::task::tabular::time_series_forecasting::TimeSeriesForecastingPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TimeSeriesForecastingPipeline>("forecast", DataType::F32);
}
