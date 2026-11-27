"""The time series transformer, forecasting each step's Student-T distribution of a monthly series from its time features and static features, with `transformers`' `TimeSeriesTransformerForPrediction`, as `mod.rs` forecasts it in Singe.
`teacher_forced` passes the future values, so `forward` gives each step's distribution after the real earlier steps.
`forecast` feeds each step's location back as the next step's value: `forward` runs once a step on future values holding the locations so far, whose later entries the causal decoder does not read.
`generate` samples instead, whose draws on the GPU cannot be matched; its first step's distribution is `forecast`'s.
`transformers` has no time series forecasting pipeline."""

import torch
from transformers import TimeSeriesTransformerForPrediction

from support import DEVICE, Verification


def time_series_forecasting(snapshot, dtype):
    model = (
        TimeSeriesTransformerForPrediction.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )
    horizon = model.config.prediction_length

    def distributions(item, future):
        tensor = lambda values, kind=dtype: torch.tensor(
            [values], dtype=kind, device=DEVICE
        )
        with torch.no_grad():
            outputs = model(
                past_values=tensor(item["past_values"]),
                past_time_features=tensor(item["past_time_features"]),
                past_observed_mask=torch.ones_like(tensor(item["past_values"])),
                static_categorical_features=tensor(
                    item["static_categorical_features"], torch.long
                ),
                static_real_features=tensor(item["static_real_features"]),
                future_values=tensor(future),
                future_time_features=tensor(item["future_time_features"]),
            )
        df, loc, scale = (parameter[0] for parameter in outputs.params)
        outer_loc, outer_scale = outputs.loc[0], outputs.scale[0]
        return df, outer_loc + outer_scale * loc, outer_scale * scale

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            future = item.get("future_values")
            if future is None:
                future = [0.0] * horizon
                for step in range(horizon):
                    future[step] = distributions(item, future)[1][step].item()
            df, loc, scale = distributions(item, future)
            outputs.append(
                {"df": df.tolist(), "loc": loc.tolist(), "scale": scale.tolist()}
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("teacher_forced", time_series_forecasting)
verification.check("forecast", time_series_forecasting)
verification.finish()
