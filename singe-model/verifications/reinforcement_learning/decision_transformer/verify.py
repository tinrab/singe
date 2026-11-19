"""The Decision Transformer, predicting each step of a trajectory with `transformers`' `DecisionTransformerModel`, as `mod.rs` predicts them in Singe.
`transformers` has no reinforcement learning pipeline."""

import torch
from transformers import DecisionTransformerModel

from support import DEVICE, Verification


def reinforcement_learning(snapshot, dtype):
    model = (
        DecisionTransformerModel.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            tensor = lambda values, **kwargs: torch.tensor(
                values, device=DEVICE, **kwargs
            ).unsqueeze(0)  # noqa: E731
            with torch.no_grad():
                predicted = model(
                    states=tensor(item["states"], dtype=dtype),
                    actions=tensor(item["actions"], dtype=dtype),
                    returns_to_go=tensor(item["returns_to_go"], dtype=dtype).unsqueeze(
                        -1
                    ),
                    timesteps=tensor(item["timesteps"], dtype=torch.long),
                )
            outputs.append(
                {
                    "state_preds": predicted.state_preds[0].tolist(),
                    "action_preds": predicted.action_preds[0].tolist(),
                    "return_preds": predicted.return_preds[0, :, 0].tolist(),
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("trajectory", reinforcement_learning)
verification.finish()
