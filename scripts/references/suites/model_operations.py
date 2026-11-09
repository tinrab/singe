"""List the ATen operations the reference models' forward passes call: just references model_operations.

Each model is built from a small configuration with random weights, so nothing is downloaded, and runs one forward pass under a dispatch mode that records every ATen operation it reaches.
singe/tests/kernels/model_operations.rs checks that Singe has an operation, a composite, or a recorded reason for every name the fixture lists.
"""

import argparse
import collections
import json
from pathlib import Path

import torch
import transformers
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "singe/tests/references"

DECODER = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
    "max_position_embeddings": 64,
}


class Record(TorchDispatchMode):
    """Counts the ATen operations a forward pass dispatches, by name without overload."""

    def __init__(self):
        super().__init__()
        self.operations = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations[str(func.overloadpacket)] += 1
        return func(*args, **(kwargs or {}))


def tokens(length):
    return {"input_ids": torch.randint(0, DECODER["vocab_size"], (1, length))}


def models():
    """Yields each reference model's name, the model, and its inputs."""
    yield (
        "llama",
        transformers.LlamaForCausalLM(transformers.LlamaConfig(**DECODER)),
        tokens(8),
    )
    yield (
        "qwen3",
        transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**DECODER, head_dim=16)),
        tokens(8),
    )
    mistral = transformers.MistralConfig(**DECODER, sliding_window=4)
    yield "mistral", transformers.MistralForCausalLM(mistral), tokens(8)
    gemma2 = transformers.Gemma2Config(**DECODER, head_dim=16, query_pre_attn_scalar=32)
    yield "gemma2", transformers.Gemma2ForCausalLM(gemma2), tokens(8)
    phi3 = {
        key: value for key, value in DECODER.items() if key != "num_key_value_heads"
    }
    yield (
        "phi3",
        transformers.Phi3ForCausalLM(transformers.Phi3Config(**phi3, pad_token_id=0)),
        tokens(8),
    )
    gpt2 = transformers.GPT2Config(
        n_embd=64, n_head=4, n_layer=2, n_positions=64, vocab_size=128
    )
    yield "gpt2", transformers.GPT2LMHeadModel(gpt2), tokens(8)
    whisper = transformers.WhisperConfig(
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        num_mel_bins=16,
        max_source_positions=8,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    yield (
        "whisper",
        transformers.WhisperForConditionalGeneration(whisper),
        {
            "input_features": torch.randn(1, 16, 16),
            "decoder_input_ids": torch.randint(0, 128, (1, 4)),
        },
    )
    vit = transformers.ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=32,
        patch_size=8,
        num_labels=10,
    )
    yield (
        "vit",
        transformers.ViTForImageClassification(vit),
        {"pixel_values": torch.randn(1, 3, 32, 32)},
    )


def generate():
    torch.manual_seed(0)
    result = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "models": {},
    }
    for name, model, inputs in models():
        model.eval()
        record = Record()
        with torch.no_grad(), record:
            model(**inputs)
        result["models"][name] = dict(sorted(record.operations.items()))
    return json.dumps(result, indent=1) + "\n"


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_operations.json").write_text(generate())
