"""Informer, forecasting each step's Student-T distribution of a monthly series from its time features and series id, with `transformers`' `InformerForPrediction`, as `mod.rs` forecasts it in Singe.
ProbSparse attention draws the keys its query selection reads with `torch.randint`, as many as there are keys in self-attention, so the reference draws each key once, as Singe reads them.
`teacher_forced` passes the future values to `forward`, whose query selection also reads the later steps.
`forecast` runs `generate`'s loop, the decoder over each step's prefix, feeding back each step's location where `generate` feeds back a sample.
`transformers` has no time series forecasting pipeline."""

import torch
from transformers import InformerForPrediction

from support import DEVICE, Verification

torch.randint = lambda low, high, size, **kwargs: torch.arange(size[0]) % high


def time_series_forecasting(snapshot, dtype):
    model = (
        InformerForPrediction.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
    )
    horizon = model.config.prediction_length

    def inputs(item):
        tensor = lambda values, kind=dtype: torch.tensor(
            [values], dtype=kind, device=DEVICE
        )
        return {
            "past_values": tensor(item["past_values"]),
            "past_time_features": tensor(item["past_time_features"]),
            "past_observed_mask": torch.ones_like(tensor(item["past_values"])),
            "static_categorical_features": tensor(
                item["static_categorical_features"], torch.long
            ),
            "future_time_features": tensor(item["future_time_features"]),
        }

    def teacher_forced(item):
        with torch.no_grad():
            outputs = model(
                **inputs(item),
                future_values=torch.tensor(
                    [item["future_values"]], dtype=dtype, device=DEVICE
                ),
            )
        df, loc, scale = (parameter[0] for parameter in outputs.params)
        outer_loc, outer_scale = outputs.loc[0], outputs.scale[0]
        return df, outer_loc + outer_scale * loc, outer_scale * scale

    def rolled(item):
        given = inputs(item)
        with torch.no_grad():
            outputs = model(**given, future_values=None)
            loc, scale = outputs.loc, outputs.scale
            past = (given["past_values"] - loc) / scale
            features = torch.cat(
                (
                    outputs.static_features.unsqueeze(1).expand(-1, horizon, -1),
                    given["future_time_features"],
                ),
                dim=-1,
            )
            steps = []
            for step in range(horizon):
                lagged = model.model.get_lagged_subsequences(
                    sequence=past, subsequences_length=1 + step, shift=1
                )
                decoder_input = torch.cat(
                    (lagged.reshape(1, step + 1, -1), features[:, : step + 1]), dim=-1
                )
                hidden = model.model.get_decoder()(
                    inputs_embeds=decoder_input,
                    encoder_hidden_states=outputs.encoder_last_hidden_state,
                ).last_hidden_state
                df, location, spread = (
                    parameter[0, -1]
                    for parameter in model.parameter_projection(hidden[:, -1:])
                )
                steps.append(
                    (df, loc[0, 0] + scale[0, 0] * location, scale[0, 0] * spread)
                )
                past = torch.cat((past, location.reshape(1, 1)), dim=1)
        return [torch.stack(values) for values in zip(*steps)]

    def run(items, parameters):
        outputs = []
        for item in items:
            df, loc, scale = (
                teacher_forced(item) if "future_values" in item else rolled(item)
            )
            outputs.append(
                {"df": df.tolist(), "loc": loc.tolist(), "scale": scale.tolist()}
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("teacher_forced", time_series_forecasting)
verification.check("forecast", time_series_forecasting)
verification.finish()
