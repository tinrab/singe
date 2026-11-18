"""TimesFM 2.5, forecasting a series with `transformers`' `TimesFm2_5ModelForPrediction` under its defaults, as `verify.rs` forecasts it in Singe.
`transformers` has no time series forecasting pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import TimesFm2_5ModelForPrediction  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def time_series_forecasting(snapshot):
    model = (
        TimesFm2_5ModelForPrediction.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            series = torch.tensor(
                item["past_values"], dtype=torch.float32, device=DEVICE
            )
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
