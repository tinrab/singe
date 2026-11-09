"""Generate tiny Whisper references for singe-model with uv run --locked.

The tiny model has random weights, and its references are encoder outputs and each greedy step's logits for fixed features.
The front end's reference is `WhisperFeatureExtractor`'s features for a fixed synthetic signal.
The real model's reference is whisper-tiny's tokens from `generate` for a LibriSpeech clip (CC BY 4.0), stored as 16-bit samples, and its weights stay in the Hugging Face cache.
"""

import io
import math
import sys
from pathlib import Path

import numpy
import pyarrow.parquet
import soundfile
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)

from ..format import half_gap, write
from ..inputs import tone_and_chirp
from ..models import OUTPUT as OUTPUT_ROOT, save_weights, typed, write_json


BASE = {
    "vocab_size": 64,
    "num_mel_bins": 8,
    "d_model": 32,
    "encoder_layers": 2,
    "encoder_attention_heads": 4,
    "encoder_ffn_dim": 64,
    "decoder_layers": 2,
    "decoder_attention_heads": 4,
    "decoder_ffn_dim": 64,
    "max_source_positions": 16,
    "max_target_positions": 32,
    "activation_function": "gelu",
    "decoder_start_token_id": 1,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

BATCH = 2
PREFIX = [1, 5, 7, 9]
NEW_TOKENS = 8

# The front end's signal: a tone and a rising chirp, 1.5 seconds at 16 kHz.
SIGNAL_SAMPLES = 24_000
# The front end's reference keeps the first frames, which cover the signal, and each mel's mean over every frame.
SIGNAL_FRAMES = 160

REAL_MODEL = "openai/whisper-tiny"
CLIP_DATASET = "hf-internal-testing/librispeech_asr_dummy"
CLIP_FILE = "clean/validation-00000-of-00001.parquet"
CLIP_INDEX = 1
REAL_NEW_TOKENS = 40


def greedy(model, features, prefix, new_tokens, stop=None):
    """Each greedy step's logits and token, recomputing the whole prefix each time, as plain greedy decoding without `generate`'s processors."""
    tokens = list(prefix)
    steps = []
    for _ in range(new_tokens):
        with torch.no_grad():
            logits = (
                model(
                    input_features=features,
                    decoder_input_ids=torch.tensor([tokens], device=features.device),
                )
                .logits[0, -1]
                .float()
                .cpu()
            )
        token = int(logits.argmax())
        steps.append((logits, token))
        tokens.append(token)
        if token == stop:
            break
    return steps


def references(model, features):
    with torch.no_grad():
        encoded = model.model.encoder(
            features.to(model.dtype)
        ).last_hidden_state.float()
    steps = [
        greedy(model, features[index : index + 1].to(model.dtype), PREFIX, NEW_TOKENS)
        for index in range(BATCH)
    ]
    tensors = {
        "encoded": encoded,
        "step_logits": torch.stack(
            [torch.stack([logits for logits, _ in sequence]) for sequence in steps]
        ),
    }
    return tensors, [[token for _, token in sequence] for sequence in steps]


def generate_tiny(output):
    config = WhisperConfig(**BASE)
    torch.manual_seed(sum(b"tiny"))
    model = (
        WhisperForConditionalGeneration._from_config(
            config, attn_implementation="eager"
        )
        .to(torch.bfloat16)
        .eval()
    )
    directory = output / "tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_weights(model, directory, {"proj_out.weight"}, 1)
    write_json(directory / "config.json", {**config.to_dict(), "model_type": "whisper"})
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(
        BATCH,
        BASE["num_mel_bins"],
        2 * BASE["max_source_positions"],
        generator=generator,
    )
    half, half_tokens = references(model, features)
    # The same weights in F32, whose reference leaves room only for arithmetic differences.
    wide, wide_tokens = references(model.to(torch.float32), features)
    write(
        directory,
        {"inputs.features": features} | typed("bf16", half) | typed("f32", wide),
        {
            "prefix": PREFIX,
            "tokens": {"bf16": half_tokens, "f32": wide_tokens},
            "half_gap": half_gap(half["step_logits"], wide["step_logits"]),
        },
    )

    extractor = WhisperFeatureExtractor(feature_size=80)
    extracted = extractor(
        tone_and_chirp(SIGNAL_SAMPLES), sampling_rate=16_000, return_tensors="np"
    ).input_features[0]
    write(
        output / "front_end",
        {
            "leading": torch.from_numpy(extracted[:, :SIGNAL_FRAMES].copy()),
            "means": torch.from_numpy(extracted.mean(axis=1)),
        },
        {"samples": SIGNAL_SAMPLES, "frames": SIGNAL_FRAMES},
    )


def clip():
    path = hf_hub_download(CLIP_DATASET, CLIP_FILE, repo_type="dataset")
    row = pyarrow.parquet.read_table(path).to_pylist()[CLIP_INDEX]
    samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
    assert rate == 16_000
    return samples, row["text"]


def generate_real(output):
    snapshot = snapshot_download(
        REAL_MODEL,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer*",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "normalizer.json",
        ],
    )
    print(f"snapshot: {snapshot}")
    samples, text = clip()
    (output / "real").mkdir(parents=True, exist_ok=True)
    (output / "real" / "clip.i16").write_bytes(samples.astype("<i2").tobytes())
    extractor = WhisperFeatureExtractor.from_pretrained(snapshot)
    features = torch.from_numpy(
        extractor(
            samples.astype(numpy.float32) / 32_768,
            sampling_rate=16_000,
            return_tensors="np",
        ).input_features
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    prefix = [
        model.generation_config.decoder_start_token_id,
        model.generation_config.lang_to_id["<|en|>"],
        model.generation_config.task_to_id["transcribe"],
        model.generation_config.no_timestamps_token_id,
    ]
    # `generate` applies the generation config's suppressed tokens, at every step and at the first, which plain greedy decoding does not.
    with torch.no_grad():
        generated = model.generate(
            features,
            language="en",
            task="transcribe",
            max_new_tokens=REAL_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
        )
    # Whisper's `generate` returns the tokens after the prompt and drops the end of text, which Singe returns as a sequence's last token.
    tokens = generated[0].tolist()
    if len(tokens) < REAL_NEW_TOKENS:
        tokens.append(model.generation_config.eos_token_id)
    from transformers import WhisperTokenizer

    transcript = WhisperTokenizer.from_pretrained(snapshot).decode(
        tokens, skip_special_tokens=True
    )
    print(f"transcript: {transcript!r}, reference text: {text!r}")
    write(
        output / "real",
        {},
        {
            "model": REAL_MODEL,
            "clip": f"{CLIP_DATASET} {CLIP_FILE} row {CLIP_INDEX}",
            "text": text,
            "transcript": transcript,
            "prefix": prefix,
            "tokens": tokens,
        },
    )


OUTPUT = OUTPUT_ROOT / "automatic_speech_recognition/whisper"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output)
