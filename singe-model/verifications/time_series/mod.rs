// Verifications of the models under `model::time_series`.

mod autoformer;
mod informer;
mod patchtst;
mod time_series_transformer;
mod timesfm2_5;

use crate::support::verifications;

verifications!(
    "time_series",
    [
        autoformer,
        informer,
        patchtst,
        time_series_transformer,
        timesfm2_5
    ]
);
