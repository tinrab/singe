"""Generate tiny BERT sentence embedder references for singe-model with uv run --locked.

Each tiny model is a `sentence-transformers` snapshot: BERT, mean pooling, and for some, normalization.
The real model's references hold only token ids and expected values, and its weights stay in the Hugging Face cache.
"""

import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, BertConfig, BertModel

from ..format import half_gap, write
from ..models import OUTPUT as OUTPUT_ROOT, save_weights, typed, write_json


BASE = {
    "vocab_size": 64,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "hidden_act": "gelu",
    "max_position_embeddings": 32,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
}

# Each model's config overrides, and whether its embeddings are normalized.
MODELS = {
    "base": ({}, True),
    "unnormalized": ({"hidden_act": "gelu_new", "layer_norm_eps": 1e-5}, False),
}

# Sequences of different lengths, so the batch pads, and the first has a second segment.
SEQUENCES = [[2, 17, 5, 9, 33, 3], [2, 41, 7, 3], [2, 11, 3]]
TOKEN_TYPES = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0]]

TRANSFORMER = "sentence_transformers.models.Transformer"
POOLING = "sentence_transformers.models.Pooling"
NORMALIZE = "sentence_transformers.models.Normalize"

REAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REAL_FILES = [
    "config.json",
    "model.safetensors",
    "modules.json",
    "1_Pooling/*",
    "tokenizer*",
    "vocab.txt",
    "special_tokens_map.json",
]
REAL_SENTENCES = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The new movie is awesome.",
    "Singe runs models on the GPU.",
]


def batch(sequences, token_types):
    """Pads with zeros: token ids, token types, and a mask of the real tokens."""
    length = max(map(len, sequences))
    pad = lambda rows: torch.tensor([row + [0] * (length - len(row)) for row in rows])
    mask = torch.tensor(
        [[1] * len(row) + [0] * (length - len(row)) for row in sequences]
    )
    return pad(sequences), pad(token_types), mask


def embed(model, input_ids, token_types, mask, normalize):
    """Each token's last hidden state and each sentence's vector, as `sentence-transformers` pools, in F32."""
    with torch.no_grad():
        hidden = model(
            input_ids=input_ids, attention_mask=mask, token_type_ids=token_types
        ).last_hidden_state
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
    if normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return hidden.float(), pooled.float()


def references(model, normalize):
    hidden, pooled = embed(model, *batch(SEQUENCES, TOKEN_TYPES), normalize)
    return {"hidden": hidden, "embeddings": pooled}


def write_modules(directory, normalize):
    """Writes the files `sentence-transformers` reads to pool: `modules.json` and the pooling module's config."""
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": TRANSFORMER},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": POOLING},
    ]
    if normalize:
        modules.append(
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": NORMALIZE}
        )
    write_json(directory / "modules.json", modules)
    write_json(
        directory / "1_Pooling/config.json",
        {
            "word_embedding_dimension": BASE["hidden_size"],
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        },
    )


def generate_tiny(output):
    for name, (overrides, normalize) in MODELS.items():
        config = BertConfig(**{**BASE, **overrides})
        torch.manual_seed(sum(name.encode()))
        model = (
            BertModel._from_config(config, attn_implementation="eager")
            .to(torch.bfloat16)
            .eval()
        )
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        save_weights(model, directory, set(), 1)
        write_json(
            directory / "config.json", {**config.to_dict(), "model_type": "bert"}
        )
        write_modules(directory, normalize)
        half = references(model, normalize)
        # The same weights in F32, whose reference leaves room only for arithmetic differences.
        wide = references(model.to(torch.float32), normalize)
        write(
            directory,
            typed("bf16", half) | typed("f32", wide),
            {
                "sequences": SEQUENCES,
                "token_types": TOKEN_TYPES,
                "half_gap": half_gap(half["hidden"], wide["hidden"]),
            },
        )


def generate_real(output):
    snapshot = snapshot_download(REAL_MODEL, allow_patterns=REAL_FILES)
    print(f"snapshot: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    encoded = tokenizer(REAL_SENTENCES)
    sequences, token_types = encoded["input_ids"], encoded["token_type_ids"]
    tensors = {}
    for name, dtype in [("bf16", torch.bfloat16), ("f32", torch.float32)]:
        model = BertModel.from_pretrained(
            snapshot, dtype=dtype, attn_implementation="eager", add_pooling_layer=False
        ).eval()
        _, tensors[f"{name}.embeddings"] = embed(
            model, *batch(sequences, token_types), normalize=True
        )
    write(
        output / "real",
        tensors,
        {
            "model": REAL_MODEL,
            "sentences": REAL_SENTENCES,
            "sequences": sequences,
            "token_types": token_types,
        },
    )


OUTPUT = OUTPUT_ROOT / "sentence_similarity/bert"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output)
