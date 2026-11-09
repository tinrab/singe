"""Generate references for attention blocks with just references blocks.

Each function writes the semantics of the kernel-dataset blocks it names at small shapes, since references are read and never copied in.
Everything is in F64-free F32, so a difference is arithmetic rather than half-precision rounding.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "singe/tests/references/blocks"
EPS = 1e-6
THETA = 10000.0
PREFILL = 3


def rms_norm(x, weight):
    """L1/015's per-head normalization."""
    return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def rotate(x, cos, sin):
    """L1/088's half-split rotation of `[T, H, D]` by angles `[T, 1, D / 2]`."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


def angles(positions, dim):
    inverse = 1.0 / THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    phase = positions.float().unsqueeze(-1) * inverse
    return phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)


def attend(query, keys, values, scale, causal):
    """Each query, the sequence's last tokens, over `keys` `[K, H, D]`, in FlashInfer-Bench 014's form."""
    scores = torch.einsum("thd,khd->htk", query, keys) * scale
    if causal:
        tokens, count = query.shape[0], keys.shape[0]
        positions = torch.arange(tokens).unsqueeze(-1) + count - tokens
        scores = scores.masked_fill(
            torch.arange(count).unsqueeze(0) > positions, float("-inf")
        )
    return torch.einsum("htk,khd->thd", torch.softmax(scores, dim=-1), values)


def gqa_arrays(tensors):
    """Weights and references for grouped-query attention, with and without Q and K norms, causal or not: L1/015, L1/018, L1/075."""
    width, heads, kv_heads, dim = 32, 4, 2, 16
    weights = {
        "q": torch.randn(heads * dim, width) * 0.2,
        "k": torch.randn(kv_heads * dim, width) * 0.2,
        "v": torch.randn(kv_heads * dim, width) * 0.2,
        "o": torch.randn(width, heads * dim) * 0.2,
        "q_norm": torch.randn(dim) * 0.5 + 1.0,
        "k_norm": torch.randn(dim) * 0.5 + 1.0,
    }
    tensors.update({f"gqa.{name}": value for name, value in weights.items()})
    hidden = torch.randn(PREFILL + 1, width)
    tensors["gqa.hidden"] = hidden
    positions = torch.arange(PREFILL + 1)
    cos, sin = angles(positions, dim)

    def project(x, cos, sin, norms):
        q = (x @ weights["q"].T).view(-1, heads, dim)
        k = (x @ weights["k"].T).view(-1, kv_heads, dim)
        v = (x @ weights["v"].T).view(-1, kv_heads, dim)
        if norms:
            q, k = rms_norm(q, weights["q_norm"]), rms_norm(k, weights["k_norm"])
        return rotate(q, cos, sin), rotate(k, cos, sin), v

    ratio = heads // kv_heads
    for label, norms, causal in [
        ("causal_norms", True, True),
        ("causal", False, True),
        ("open_norms", True, False),
    ]:
        q, k, v = project(hidden[:PREFILL], cos[:PREFILL], sin[:PREFILL], norms)
        tensors[f"gqa.{label}.keys"] = k
        output = attend(
            q,
            k.repeat_interleave(ratio, 1),
            v.repeat_interleave(ratio, 1),
            1 / math.sqrt(dim),
            causal,
        )
        tensors[f"gqa.{label}.prefill"] = output.reshape(PREFILL, -1) @ weights["o"].T
        last_q, last_k, last_v = project(
            hidden[PREFILL:], cos[PREFILL:], sin[PREFILL:], norms
        )
        keys, values = torch.cat([k, last_k]), torch.cat([v, last_v])
        output = attend(
            last_q,
            keys.repeat_interleave(ratio, 1),
            values.repeat_interleave(ratio, 1),
            1 / math.sqrt(dim),
            causal,
        )
        tensors[f"gqa.{label}.decode"] = output.reshape(1, -1) @ weights["o"].T


def dense_arrays(tensors):
    """Grouped-query attention over whole batches without a cache, causal for a decoder and open for an encoder: L1/075."""
    heads, kv_heads, dim = 4, 2, 16
    get = lambda name: tensors[f"gqa.{name}"]
    hidden = torch.randn(2, PREFILL + 1, 32)
    tensors["gqa.dense.hidden"] = hidden
    cos, sin = angles(torch.arange(PREFILL + 1), dim)
    for label, causal in [("dense_causal", True), ("dense_open", False)]:
        outputs = []
        for x in hidden:
            q = rotate((x @ get("q").T).view(-1, heads, dim), cos, sin)
            k = rotate((x @ get("k").T).view(-1, kv_heads, dim), cos, sin)
            v = (x @ get("v").T).view(-1, kv_heads, dim)
            attended = attend(
                q,
                k.repeat_interleave(2, 1),
                v.repeat_interleave(2, 1),
                1 / math.sqrt(dim),
                causal,
            )
            outputs.append(attended.reshape(PREFILL + 1, -1) @ get("o").T)
        tensors[f"gqa.{label}"] = torch.stack(outputs)


def block_arrays(tensors):
    """A pre-norm decoder layer with a gated feed-forward layer: L2/002, L2/020, L1/074."""
    width, inner = 32, 48
    weights = {
        "input_norm": torch.randn(width) * 0.5 + 1.0,
        "post_norm": torch.randn(width) * 0.5 + 1.0,
        "gate": torch.randn(inner, width) * 0.2,
        "up": torch.randn(inner, width) * 0.2,
        "down": torch.randn(width, inner) * 0.2,
    }
    tensors.update({f"block.{name}": value for name, value in weights.items()})
    hidden = tensors["gqa.hidden"]
    heads, kv_heads, dim = 4, 2, 16
    get = lambda name: tensors[f"gqa.{name}"]
    cos, sin = angles(torch.arange(PREFILL), dim)
    x = hidden[:PREFILL]
    normed = rms_norm(x, weights["input_norm"])
    q = rms_norm((normed @ get("q").T).view(-1, heads, dim), get("q_norm"))
    k = rms_norm((normed @ get("k").T).view(-1, kv_heads, dim), get("k_norm"))
    v = (normed @ get("v").T).view(-1, kv_heads, dim)
    q, k = rotate(q, cos, sin), rotate(k, cos, sin)
    attended = attend(
        q,
        k.repeat_interleave(2, 1),
        v.repeat_interleave(2, 1),
        1 / math.sqrt(dim),
        True,
    )
    y = x + attended.reshape(PREFILL, -1) @ get("o").T
    z = rms_norm(y, weights["post_norm"])
    gate = torch.nn.functional.silu(z @ weights["gate"].T)
    tensors["block.output"] = y + (gate * (z @ weights["up"].T)) @ weights["down"].T


def mla_arrays(tensors):
    """Multi-head latent attention's projections and its expanded form, which the absorbed form must equal: L1/043, L1/064."""
    width, heads, nope, rope, value, latent, rank = 32, 2, 16, 8, 16, 32, 24
    weights = {
        "q_a": torch.randn(rank, width) * 0.2,
        "q_a_norm": torch.randn(rank) * 0.5 + 1.0,
        "q_b": torch.randn(heads * (nope + rope), rank) * 0.2,
        "kv_a": torch.randn(latent + rope, width) * 0.2,
        "kv_a_norm": torch.randn(latent) * 0.5 + 1.0,
        "kv_b": torch.randn(heads * (nope + value), latent) * 0.2,
        "o": torch.randn(width, heads * value) * 0.2,
    }
    tensors.update({f"mla.{name}": tensor for name, tensor in weights.items()})
    hidden = torch.randn(PREFILL + 1, width)
    tensors["mla.hidden"] = hidden
    cos, sin = angles(torch.arange(PREFILL + 1), rope)
    query = rms_norm(hidden @ weights["q_a"].T, weights["q_a_norm"]) @ weights["q_b"].T
    query = query.view(-1, heads, nope + rope)
    q_nope, q_pe = query[..., :nope], rotate(query[..., nope:], cos, sin)
    compressed = hidden @ weights["kv_a"].T
    kv_latent = rms_norm(compressed[..., :latent], weights["kv_a_norm"])
    k_pe = rotate(compressed[..., latent:].unsqueeze(1), cos, sin)
    tensors["mla.latent"], tensors["mla.positional"] = kv_latent, k_pe.squeeze(1)
    expanded = (kv_latent @ weights["kv_b"].T).view(-1, heads, nope + value)
    k_nope, values = expanded[..., :nope], expanded[..., nope:]
    keys = torch.cat([k_nope, k_pe.expand(-1, heads, -1)], dim=-1)
    full_query = torch.cat([q_nope, q_pe], dim=-1)
    scale = 1 / math.sqrt(nope + rope)
    prefill = attend(
        full_query[:PREFILL], keys[:PREFILL], values[:PREFILL], scale, True
    )
    tensors["mla.prefill"] = prefill.reshape(PREFILL, -1) @ weights["o"].T
    decode = attend(full_query[PREFILL:], keys, values, scale, True)
    tensors["mla.decode"] = decode.reshape(1, -1) @ weights["o"].T


def tiny(output):
    torch.manual_seed(0)
    tensors = {}
    gqa_arrays(tensors)
    dense_arrays(tensors)
    block_arrays(tensors)
    mla_arrays(tensors)
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: tensor.contiguous().clone() for name, tensor in tensors.items()},
        output / "blocks.safetensors",
    )
