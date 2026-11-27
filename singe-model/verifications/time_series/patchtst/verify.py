"""PatchTST, forecasting each channel of a multivariate series, once with some values missing, and single series with checkpoints that read every patch's state, the patches' mean, or layer norms instead of a class token and batch norms, with `transformers`' `PatchTSTForPrediction`, as `mod.rs` forecasts it in Singe.
`transformers` has no time series forecasting pipeline."""

import torch
from transformers import PatchTSTForPrediction

from support import DEVICE, Verification


def time_series_forecasting(snapshot, dtype):
    model = (
        PatchTSTForPrediction.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            series = torch.tensor([item["past_values"]], dtype=dtype, device=DEVICE)
            observed = item.get("past_observed_mask")
            if observed is not None:
                observed = torch.tensor([observed], dtype=dtype, device=DEVICE)
            with torch.no_grad():
                forecast = model(past_values=series, past_observed_mask=observed)
            outputs.append(
                {"prediction_outputs": forecast.prediction_outputs[0].tolist()}
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("forecast", time_series_forecasting)
verification.check("missing", time_series_forecasting)
verification.check("flatten", time_series_forecasting)
verification.check("mean", time_series_forecasting)
verification.check("layer_norm", time_series_forecasting)
verification.finish()
