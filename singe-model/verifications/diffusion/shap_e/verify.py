"""Shap-E, generating a mesh from a prompt and a seed with `diffusers`' `ShapEPipeline`, as `verify.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both read the half-precision weights, the only safetensors of the snapshot, in F32; the case records a summary of the mesh, its counts and sampled vertices and colors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from diffusers import ShapEPipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def text_to_3d(snapshot):
    generate = ShapEPipeline.from_pretrained(
        snapshot, variant="fp16", torch_dtype=torch.float32
    ).to(DEVICE)
    generate.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            mesh = generate(
                item["prompt"],
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
verification.check("mesh", text_to_3d)
verification.finish()
