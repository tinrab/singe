"""The Decision Transformer, predicting each step of a trajectory with `transformers`' `DecisionTransformerModel`, as `verify.rs` predicts them in Singe.
`transformers` has no reinforcement learning pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import DecisionTransformerModel  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def reinforcement_learning(snapshot):
    model = (
        DecisionTransformerModel.from_pretrained(snapshot, dtype=torch.float32)
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
                    states=tensor(item["states"], dtype=torch.float32),
                    actions=tensor(item["actions"], dtype=torch.float32),
                    returns_to_go=tensor(
                        item["returns_to_go"], dtype=torch.float32
                    ).unsqueeze(-1),
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
