// PatchTST, forecasting each channel of a multivariate series, once with some values missing, and single series without a class token, by every patch's state, the patches' mean, or under layer norms, as `verify.py` forecasts it with `transformers`.

use singe_model::task::tabular::time_series_forecasting::TimeSeriesForecastingPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<TimeSeriesForecastingPipeline>("forecast");
    session.check::<TimeSeriesForecastingPipeline>("missing");
    for case in ["flatten", "mean", "layer_norm"] {
        session.check::<TimeSeriesForecastingPipeline>(case);
    }
}
