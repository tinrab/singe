"""Shap-E's image variant, generating a mesh from an image and a seed with `diffusers`' `ShapEImg2ImgPipeline`, as `mod.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both read the half-precision weights, the only safetensors of the snapshot, in F32; the case records a summary of the mesh, its counts and sampled vertices and colors."""

import torch
from diffusers import ShapEImg2ImgPipeline
from PIL import Image

from support import DEVICE, Verification


def image_to_3d(snapshot, dtype):
    generate = ShapEImg2ImgPipeline.from_pretrained(
        snapshot, variant="fp16", torch_dtype=dtype
    ).to(DEVICE)
    generate.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            mesh = generate(
                Image.open(item["image"]).convert("RGB"),
                generator=torch.Generator("cpu").manual_seed(item["seed"]),
                output_type="mesh",
                **parameters,
            ).images[0]
            colors = torch.stack(
                [mesh.vertex_channels[channel] for channel in "RGB"], dim=1
            )
            outputs.append(
                {
                    "vertices": mesh.verts.tolist(),
                    "faces": mesh.faces.tolist(),
                    "colors": colors.tolist(),
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("mesh", image_to_3d)
verification.finish()
