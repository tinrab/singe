"""Generate tiny Llama references for singe-model with uv run --locked."""

import sys
from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM

from ..models import OUTPUT as OUTPUT_ROOT, write_tiny

BASE = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
}

# The base model's checkpoint comes in shards, as a large model's does.
MODELS = {
    "base": ({}, 2),
    "biased": ({"attention_bias": True, "mlp_bias": True}, 1),
    "tied": ({"tie_word_embeddings": True}, 1),
    "gelu": ({"hidden_act": "gelu"}, 1),
    "narrow_heads": ({"hidden_size": 60, "num_attention_heads": 5}, 1),
}


def generate(output):
    for name, (overrides, shards) in MODELS.items():
        values = {**BASE, **overrides}
        rope = {"rope_type": "default", "rope_theta": 10000.0}
        tied = ["lm_head.weight"] if values.get("tie_word_embeddings") else []
        write_tiny(
            output,
            name,
            LlamaForCausalLM,
            LlamaConfig(**values, rope_parameters=rope),
            "eager",
            tied,
            shards,
        )


OUTPUT = OUTPUT_ROOT / "text_generation/llama"


def tiny(output):
    generate(output)
