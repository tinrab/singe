"""TimesFM 2.5, forecasting a series with `transformers`' `TimesFm2_5ModelForPrediction` under its defaults, as `mod.rs` forecasts it in Singe.
`transformers` has no time series forecasting pipeline."""

import torch
from transformers import TimesFm2_5ModelForPrediction

from support import DEVICE, Verification


def time_series_forecasting(snapshot, dtype):
    model = (
        TimesFm2_5ModelForPrediction.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            series = torch.tensor(item["past_values"], dtype=dtype, device=DEVICE)
            with torch.no_grad():
                forecast = model(past_values=[series])
            outputs.append(
                {
                    "mean_predictions": forecast.mean_predictions[0].tolist(),
                    "full_predictions": forecast.full_predictions[0].tolist(),
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("forecast", time_series_forecasting)
verification.finish()
