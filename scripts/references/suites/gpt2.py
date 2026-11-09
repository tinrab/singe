"""Generate tiny GPT-2 references for singe-model with uv run --locked."""

import sys
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel

from ..models import OUTPUT as OUTPUT_ROOT, write_tiny

BASE = {
    "vocab_size": 64,
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "n_positions": 64,
    "n_inner": 128,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
    "tie_word_embeddings": True,
}

MODELS = {
    "base": {},
    "exact_gelu": {"activation_function": "gelu"},
    "default_inner": {"n_inner": None},
}


def generate(output):
    for name, overrides in MODELS.items():
        config = GPT2Config(**{**BASE, **overrides})
        write_tiny(output, name, GPT2LMHeadModel, config, "eager", ["lm_head.weight"])


OUTPUT = OUTPUT_ROOT / "text_generation/gpt2"


def tiny(output):
    generate(output)
