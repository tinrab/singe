"""Generate tiny DeepSeek-V3 references for singe-model with uv run --locked.

Most have dense layers only, and the `moe` models make every layer after the first mixture-of-experts.
Real checkpoints hold each expert's projections separately, as `moe_per_expert` does, and `transformers` packs them as it loads.
"""

import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

from ..models import OUTPUT as OUTPUT_ROOT, write_tiny

BASE = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "kv_lora_rank": 64,
    "q_lora_rank": 48,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 32,
    "v_head_dim": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "first_k_dense_replace": 2,
    "rope_interleave": True,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 128,
}

YARN = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}

MOE = {
    "first_k_dense_replace": 1,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "n_group": 4,
    "topk_group": 2,
    "num_experts_per_tok": 3,
}

MODELS = {
    # The shape the tensor-core kernel takes.
    "base": {},
    "direct_query": {"q_lora_rank": None},
    "half_split": {"rope_interleave": False},
    "yarn": {"rope_scaling": YARN},
    "tied": {"tie_word_embeddings": True},
    # A latent width only the general kernel takes.
    "narrow": {"kv_lora_rank": 40, "qk_rope_head_dim": 12},
    "moe": MOE,
    "moe_per_expert": MOE,
}


def split_experts(directory, inner):
    """Rewrites packed experts as one tensor per expert, as a real checkpoint holds them."""
    state = load_file(directory / "model.safetensors")
    for name in [name for name in state if name.endswith(".experts.gate_up_proj")]:
        stem = name.removesuffix("gate_up_proj")
        gate_up, down = state.pop(name), state.pop(f"{stem}down_proj")
        for expert in range(gate_up.shape[0]):
            state[f"{stem}{expert}.gate_proj.weight"] = gate_up[expert, :inner].clone()
            state[f"{stem}{expert}.up_proj.weight"] = gate_up[expert, inner:].clone()
            state[f"{stem}{expert}.down_proj.weight"] = down[expert].clone()
    save_file(state, directory / "model.safetensors")


def build(values):
    """A `transformers` config from what `config.json` would hold."""
    arguments = {
        key: value
        for key, value in values.items()
        if key not in {"rope_theta", "rope_scaling"}
    }
    rope = {"rope_type": "default", "rope_theta": values["rope_theta"]}
    if "rope_scaling" in values:
        scaling = dict(values["rope_scaling"])
        scaling.pop("type")
        rope = {"rope_type": "yarn", "rope_theta": values["rope_theta"], **scaling}
    return DeepseekV3Config(**arguments, rope_parameters=rope)


def generate(output):
    for name, overrides in MODELS.items():
        values = {**BASE, **overrides}
        tied = ["lm_head.weight"] if values["tie_word_embeddings"] else []
        write_tiny(output, name, DeepseekV3ForCausalLM, build(values), "eager", tied)
        if name == "moe_per_expert":
            split_experts(output / name, values["moe_intermediate_size"])


OUTPUT = OUTPUT_ROOT / "text_generation/deepseek_v3"


def tiny(output):
    generate(output)
