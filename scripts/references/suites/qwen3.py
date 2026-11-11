"""Generate Qwen3 references for singe-model with uv run --locked.

The real model's references hold only expected values, and its weights stay in the Hugging Face cache.
"""

import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from ..format import write
from ..models import OUTPUT as OUTPUT_ROOT, new_cache, step, write_tiny

REAL_MODEL = "Qwen/Qwen3-0.6B"
REAL_TOP_LOGITS = 10
REAL_NEW_TOKENS = 64
PROMPTS = [
    "What is the capital of France?",
    "Write a short poem about the sea.",
    "List three prime numbers.",
]

TINY_CONFIG = {
    "model_type": "qwen3",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_act": "silu",
    "attention_bias": False,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
}


def greedy(model, prompt, new_tokens):
    """Yields the logits behind each greedy token and the token."""
    cache = new_cache(model)
    logits = step(model, prompt, cache)
    for _ in range(new_tokens):
        token = int(logits.argmax())
        yield logits, token
        logits = step(model, [token], cache)


def generate_tiny(output):
    arguments = {
        key: value
        for key, value in TINY_CONFIG.items()
        if key not in {"model_type", "rope_theta", "rope_scaling"}
    }
    config = Qwen3Config(
        **arguments,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": TINY_CONFIG["rope_theta"],
        },
    )
    write_tiny(output, "tiny", Qwen3ForCausalLM, config, "sdpa", ["lm_head.weight"])


def generate_real(output, name=REAL_MODEL):
    """Writes a chat model's greedy tokens for each prompt, with the BF16 and F32 logits of each step's top tokens."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    load = lambda dtype: (
        AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    model = load(torch.bfloat16)
    cases, tensors = [], {}
    for index, prompt in enumerate(PROMPTS):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer(text).input_ids
        steps, top_logits = [], []
        for logits, token in greedy(model, input_ids, REAL_NEW_TOKENS):
            top = logits.topk(REAL_TOP_LOGITS)
            steps.append({"top_tokens": top.indices.tolist(), "token": token})
            top_logits.append(top.values)
        tensors[f"case{index}.top_logits"] = torch.stack(top_logits)
        cases.append(
            {"prompt": prompt, "text": text, "input_ids": input_ids, "steps": steps}
        )
    del model
    torch.cuda.empty_cache()

    # BF16 rounding moves logits by up to half a unit, which would hide a real difference, so F32 logits on the same tokens give a stricter reference.
    model = load(torch.float32)
    for index, case in enumerate(cases):
        cache = new_cache(model)
        tokens = case["input_ids"]
        wide = []
        for reference in case["steps"]:
            logits = step(model, tokens, cache)
            wide.append(logits[reference["top_tokens"]])
            tokens = [reference["token"]]
        tensors[f"case{index}.top_logits_f32"] = torch.stack(wide)
    write(output / "real", tensors, {"model": name, "cases": cases})


OUTPUT = OUTPUT_ROOT / "text_generation/qwen3"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output)
