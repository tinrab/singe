"""Generate Qwen3.5 references for singe-model with uv run --locked."""

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from ..models import OUTPUT as OUTPUT_ROOT, write_tiny
from .qwen3 import generate_real

# Three linear-attention layers and one of full attention, with more value heads than key heads, so the delta rule repeats its keys.
TINY_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_value_head_dim": 16,
    "linear_num_value_heads": 4,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
}


class PerturbedNorms:
    """Builds the model with noise on its norms' weights, which it initializes to zero for those that scale by one plus their weight, so a norm read without its offset would show."""

    @staticmethod
    def _from_config(config, **kwargs):
        model = Qwen3_5ForCausalLM._from_config(config, **kwargs)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.add_(torch.randn_like(parameter) * 0.2)
        return model


def generate_tiny(output):
    write_tiny(
        output,
        "tiny",
        PerturbedNorms,
        Qwen3_5TextConfig(**TINY_CONFIG),
        "eager",
        ["lm_head.weight"],
    )


OUTPUT = OUTPUT_ROOT / "text_generation/qwen3_5"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output, "Qwen/Qwen3.5-0.8B")
