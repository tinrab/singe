"""Generate tiny Gemma 2 references for singe-model with uv run --locked."""

import sys
from pathlib import Path

from transformers import Gemma2Config, Gemma2ForCausalLM

from ..models import OUTPUT as OUTPUT_ROOT, write_tiny

BASE = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
    "query_pre_attn_scalar": 32,
    "attn_logit_softcapping": None,
    "final_logit_softcapping": None,
    "tie_word_embeddings": True,
    # Its default layers alternate sliding and full attention, and this window is shorter than a prompt.
    "sliding_window": 4,
}

MODELS = {
    "base": {},
    "capped": {"attn_logit_softcapping": 5.0},
    "final_capped": {"final_logit_softcapping": 2.0},
}


def generate(output):
    for name, overrides in MODELS.items():
        values = {**BASE, **overrides}
        rope = {"rope_type": "default", "rope_theta": 10000.0}
        write_tiny(
            output,
            name,
            Gemma2ForCausalLM,
            Gemma2Config(**values, rope_parameters=rope),
            "eager",
            ["lm_head.weight"],
        )


OUTPUT = OUTPUT_ROOT / "text_generation/gemma2"


def tiny(output):
    generate(output)
