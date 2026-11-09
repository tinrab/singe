"""Generate tiny TAESD references for singe-model with uv run --locked.

Each tiny model is a `diffusers` `AutoencoderTiny` snapshot with random weights, and its references are latents for fixed images and images for fixed latents.
The real model's references hold only expected values, sampled from its decoded image, and its weights stay in the Hugging Face cache.
"""

import json
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderTiny
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from ..format import half_gap, write
from ..inputs import pattern_image as image
from ..models import OUTPUT as OUTPUT_ROOT, typed, write_json


BASE = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "encoder_block_out_channels": (8, 8, 8),
    "decoder_block_out_channels": (8, 8, 8),
    "num_encoder_blocks": (1, 2, 1),
    "num_decoder_blocks": (1, 2, 1),
    "act_fn": "relu",
    "upsample_fn": "nearest",
    "upsampling_scaling_factor": 2,
}

MODELS = {"base": {}, "bilinear": {"upsample_fn": "bilinear"}}

BATCH = 2
SIZE = 16
REAL_MODEL = "madebyollin/taesd"
REAL_SIZE = 256
# The decoded image is sampled every this many pixels along each axis, to keep its reference small.
REAL_STRIDE = 16


def references(model, images, latents):
    with torch.no_grad():
        encoded = model.encode(images.to(model.dtype)).latents.float()
        decoded = model.decode(latents.to(model.dtype)).sample.float()
    return {"latents": encoded, "images": decoded}


def generate_tiny(output):
    generator = torch.Generator().manual_seed(7)
    images = torch.rand(BATCH, 3, SIZE, SIZE, generator=generator) * 2 - 1
    latents = torch.randn(BATCH, 4, SIZE // 4, SIZE // 4, generator=generator) * 2
    for name, overrides in MODELS.items():
        torch.manual_seed(sum(name.encode()))
        model = AutoencoderTiny(**{**BASE, **overrides}).to(torch.float16).eval()
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        state = {key: tensor.contiguous() for key, tensor in model.state_dict().items()}
        save_file(state, directory / "diffusion_pytorch_model.safetensors")
        config = {
            key: value
            for key, value in dict(model.config).items()
            if not key.startswith("_")
        }
        write_json(
            directory / "config.json",
            {"_class_name": "AutoencoderTiny", **json.loads(json.dumps(config))},
        )
        half = references(model, images, latents)
        # The same weights in F32, whose reference leaves room only for arithmetic differences.
        wide = references(model.to(torch.float32), images, latents)
        write(
            directory,
            {"inputs.images": images, "inputs.latents": latents}
            | typed("f16", half)
            | typed("f32", wide),
            {"half_gap": half_gap(half["images"], wide["images"])},
        )


def generate_real(output):
    snapshot = snapshot_download(
        REAL_MODEL,
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
    )
    print(f"snapshot: {snapshot}")
    images = image(REAL_SIZE)
    expected = {}
    latents = None
    # PyTorch convolves F32 on CUDA in TF32 by default, so the F32 reference runs on the CPU.
    for name, dtype, device in [
        ("f32", torch.float32, "cpu"),
        ("f16", torch.float16, "cuda"),
    ]:
        model = (
            AutoencoderTiny.from_pretrained(snapshot, torch_dtype=dtype)
            .to(device)
            .eval()
        )
        with torch.no_grad():
            encoded = model.encode(images.to(device, dtype)).latents.float().cpu()
            if latents is None:
                latents = encoded
            decoded = model.decode(latents.to(device, dtype)).sample.float().cpu()
        expected[f"{name}.latents"] = encoded
        expected[f"{name}.image_samples"] = decoded[
            ..., ::REAL_STRIDE, ::REAL_STRIDE
        ].contiguous()
    write(
        output / "real",
        expected,
        {"model": REAL_MODEL, "size": REAL_SIZE, "stride": REAL_STRIDE},
    )


OUTPUT = OUTPUT_ROOT / "image_to_image/taesd"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output)
