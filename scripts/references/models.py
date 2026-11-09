"""What the model suites share: tiny models, their weights, and decoders' references against `transformers`.

Each model has random weights and is checked in whole, so its tests need no download.
Each writes references in BF16 and in F32, since half-precision rounding would hide a real difference.
"""

import json
import torch
from safetensors.torch import save_file
from transformers import DynamicCache

from .format import ROOT, half_gap, write

OUTPUT = ROOT / "singe-model/tests/references"
PROMPT = [1, 5, 9, 13, 2, 41, 7]
OTHER_PROMPT = [3, 17, 8]
SPLIT = 4
NEW_TOKENS = 6
SHARD_INDEX = "model.safetensors.index.json"


def step(model, tokens, cache):
    """The logits of the last of `tokens` after `cache`, which the call extends, in F32."""
    input_ids = torch.tensor([tokens], device=model.device)
    with torch.no_grad():
        return (
            model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            .logits[0, -1]
            .float()
            .cpu()
        )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def references(model):
    """Logits for a whole prompt, a split prompt, each greedy step, and a second prompt's, in F32, and the greedy tokens."""
    split = DynamicCache()
    step(model, PROMPT[:SPLIT], split)
    split_logits = step(model, PROMPT[SPLIT:], split)
    cache = DynamicCache()
    logits = step(model, PROMPT, cache)
    prompt_logits = logits
    steps, tokens = [], []
    for _ in range(NEW_TOKENS):
        token = int(logits.argmax())
        steps.append(logits)
        tokens.append(token)
        logits = step(model, [token], cache)
    other = step(model, OTHER_PROMPT, DynamicCache())
    tensors = {
        "prompt_logits": prompt_logits,
        "split_logits": split_logits,
        "other_logits": other,
        "step_logits": torch.stack(steps),
    }
    return tensors, tokens


def typed(name, tensors):
    """Names each tensor after the type its model ran in, as `bf16.step_logits`."""
    return {f"{name}.{key}": value for key, value in tensors.items()}


def save_weights(model, output, tied_names, shards):
    """Writes `model.safetensors`, or that many shards and their index."""
    state = {
        key: tensor.clone().contiguous()
        for key, tensor in model.state_dict().items()
        if key not in tied_names
    }
    if shards == 1:
        save_file(state, output / "model.safetensors")
        return
    names = sorted(state)
    weight_map = {}
    for index in range(shards):
        file = f"model-{index + 1:05}-of-{shards:05}.safetensors"
        part = names[index::shards]
        save_file({key: state[key] for key in part}, output / file)
        weight_map.update(dict.fromkeys(part, file))
    write_json(output / SHARD_INDEX, {"metadata": {}, "weight_map": weight_map})


def write_tiny(output, name, model_class, config, attention, tied_names=(), shards=1):
    """Writes a config as `transformers` saves it, weights in BF16, and references in both types."""
    torch.manual_seed(sum(name.encode()))
    model = (
        model_class._from_config(config, attn_implementation=attention)
        .to(torch.bfloat16)
        .eval()
    )
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    save_weights(model, directory, set(tied_names), shards)
    write_json(
        directory / "config.json", json.loads(config.to_json_string(use_diff=False))
    )
    half, half_tokens = references(model)
    # The same weights in F32, whose reference leaves room only for arithmetic differences.
    wide, wide_tokens = references(model.to(torch.float32))
    write(
        directory,
        typed("bf16", half) | typed("f32", wide),
        {
            "prompt": PROMPT,
            "split": SPLIT,
            "other_prompt": OTHER_PROMPT,
            "tokens": {"bf16": half_tokens, "f32": wide_tokens},
            "half_gap": half_gap(half["step_logits"], wide["step_logits"]),
        },
    )
