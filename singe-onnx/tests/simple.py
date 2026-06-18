import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def export_onnx():
    torch.manual_seed(0)
    torch_model = ImageClassifierModel()
    torch_model.eval()
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    example_inputs = (torch.randn(1, 1, 32, 32),)
    with torch.no_grad():
        expected_output = torch_model(*example_inputs)
    onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
    onnx_program.save(data_dir / "image_classifier_model.onnx")
    (data_dir / "image_classifier_model_input.json").write_text(
        json.dumps(example_inputs[0].reshape(-1).tolist())
    )
    (data_dir / "image_classifier_model_output.json").write_text(
        json.dumps(expected_output.reshape(-1).tolist())
    )


if __name__ == '__main__':
    export_onnx()
