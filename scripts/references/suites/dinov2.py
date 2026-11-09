"""Generate tiny DINOv2 references for singe-model with uv run --locked.

The real model's references hold only expected values, and its weights stay in the Hugging Face cache.
"""

import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import Dinov2Config, Dinov2Model

from ..format import half_gap, write
from ..inputs import pattern_image as image
from ..models import OUTPUT as OUTPUT_ROOT, save_weights, write_json


BASE = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "mlp_ratio": 4,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-6,
    "image_size": 16,
    "patch_size": 4,
    "num_channels": 3,
    "qkv_bias": True,
}

MODELS = {"base": {}, "no_qkv_bias": {"qkv_bias": False, "hidden_act": "gelu_new"}}

BATCH = 2
REAL_MODEL = "facebook/dinov2-small"
REAL_SIZE = 518
# Every this many tokens' final hidden states are kept, so the reference covers patches without holding all 1370.
REAL_TOKEN_STRIDE = 137


def hidden_states(model, images):
    with torch.no_grad():
        return (
            model(pixel_values=images.to(model.device, model.dtype))
            .last_hidden_state.float()
            .cpu()
        )


def generate_tiny(output):
    generator = torch.Generator().manual_seed(11)
    images = (
        torch.rand(
            BATCH, 3, BASE["image_size"], BASE["image_size"], generator=generator
        )
        * 2
        - 1
    )
    for name, overrides in MODELS.items():
        config = Dinov2Config(**{**BASE, **overrides})
        torch.manual_seed(sum(name.encode()))
        model = Dinov2Model._from_config(config, attn_implementation="eager").eval()
        # Layer scales start at one, which would hide a scale applied in the wrong place.
        with torch.no_grad():
            for parameter_name, parameter in model.named_parameters():
                if parameter_name.endswith("lambda1"):
                    parameter.uniform_(0.5, 1.5)
        model = model.to(torch.bfloat16)
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        save_weights(model, directory, set(), 1)
        write_json(
            directory / "config.json", {**config.to_dict(), "model_type": "dinov2"}
        )
        half = hidden_states(model, images)
        # The same weights in F32, whose reference leaves room only for arithmetic differences.
        wide = hidden_states(model.to(torch.float32), images)
        write(
            directory,
            {"inputs.images": images, "bf16.hidden": half, "f32.hidden": wide},
            {"half_gap": half_gap(half, wide)},
        )


def generate_real(output):
    snapshot = snapshot_download(
        REAL_MODEL, allow_patterns=["config.json", "model.safetensors"]
    )
    print(f"snapshot: {snapshot}")
    images = image(REAL_SIZE)
    expected = {}
    # PyTorch convolves F32 on CUDA in TF32 by default, so the F32 reference runs on the CPU.
    for name, dtype, device in [
        ("f32", torch.float32, "cpu"),
        ("bf16", torch.bfloat16, "cuda"),
    ]:
        model = (
            Dinov2Model.from_pretrained(
                snapshot, dtype=dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
        hidden = hidden_states(model, images)[0]
        expected[f"{name}.tokens"] = hidden[::REAL_TOKEN_STRIDE].contiguous()
    write(
        output / "real",
        expected,
        {
            "model": REAL_MODEL,
            "size": REAL_SIZE,
            "token_stride": REAL_TOKEN_STRIDE,
            "half_gap": half_gap(expected["bf16.tokens"], expected["f32.tokens"]),
        },
    )


OUTPUT = OUTPUT_ROOT / "image_feature_extraction/dinov2"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output)
