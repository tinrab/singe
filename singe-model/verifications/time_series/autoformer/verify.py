"""Autoformer, forecasting a monthly series with `transformers`' `AutoformerForPrediction.generate`, as `mod.rs` forecasts it in Singe.
`generate` samples each step from a Student-T distribution, whose draws on the GPU cannot be matched, so the case records the distribution it builds: each step's degrees of freedom, and its location and scale mapped back to the series' scale.
`transformers` has no time series forecasting pipeline."""

import torch
from transformers import AutoformerForPrediction

from support import DEVICE, Verification


def time_series_forecasting(snapshot, dtype):
    model = (
        AutoformerForPrediction.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
    )
    built = {}
    output_distribution = model.output_distribution

    def recorded(params, loc=None, scale=None, trailing_n=None):
        built.update(params=params, loc=loc, scale=scale)
        return output_distribution(params, loc=loc, scale=scale, trailing_n=trailing_n)

    model.output_distribution = recorded

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            tensor = lambda values, kind=dtype: torch.tensor(
                [values], dtype=kind, device=DEVICE
            )
            model.generate(
                past_values=tensor(item["past_values"]),
                past_time_features=tensor(item["past_time_features"]),
                future_time_features=tensor(item["future_time_features"]),
                static_categorical_features=tensor(
                    item["static_categorical_features"], torch.long
                ),
            )
            # Every row of the batch `generate` repeats is the same series; row 0 is kept.
            df, loc, scale = (parameter[0] for parameter in built["params"])
            outer_loc, outer_scale = built["loc"][0], built["scale"][0]
            outputs.append(
                {
                    "df": df.tolist(),
                    "loc": (outer_loc + outer_scale * loc).tolist(),
                    "scale": (outer_scale * scale).tolist(),
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("forecast", time_series_forecasting)
verification.finish()
