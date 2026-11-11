"""Generate Qwen3-ASR references for singe-model with uv run --locked.

The tiny model has random weights, and its references are its features for a fixed synthetic signal, its encoder's rows, and the greedy steps after a prompt that reads them.
The real model's reference is Qwen3-ASR-0.6B's tokens from `generate` for the LibriSpeech clip the Whisper suite reads, stored as 16-bit samples, and its weights stay in the Hugging Face cache.
"""

import torch
from transformers import (
    AutoProcessor,
    DynamicCache,
    Qwen3ASRConfig,
    Qwen3ASRFeatureExtractor,
    Qwen3ASRForConditionalGeneration,
)

from ..format import write
from ..inputs import tone_and_chirp
from ..models import OUTPUT as OUTPUT_ROOT, save_weights, write_json
from .whisper import clip

REAL_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
REAL_NEW_TOKENS = 64

# 1.3 seconds: a whole chunk of mel frames and a partial one, whose positions fill an attention window and start another.
SIGNAL_SAMPLES = 20_800
AUDIO_TOKEN = 5
NEW_TOKENS = 8

TINY_AUDIO = {
    "num_mel_bins": 32,
    "encoder_layers": 2,
    "encoder_attention_heads": 2,
    "encoder_ffn_dim": 64,
    "d_model": 32,
    "downsample_hidden_size": 8,
    "n_window": 50,
    "n_window_infer": 100,
    "output_dim": 64,
    "max_position_embeddings": 13,
}

TINY_TEXT = {
    "model_type": "qwen3",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "hidden_act": "silu",
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "tie_word_embeddings": True,
}


def generate_tiny(output):
    config = Qwen3ASRConfig(
        audio_config=TINY_AUDIO,
        text_config=TINY_TEXT,
        audio_token_id=AUDIO_TOKEN,
        pad_token_id=0,
        eos_token_id=[1],
    )
    torch.manual_seed(sum(b"tiny"))
    model = Qwen3ASRForConditionalGeneration._from_config(
        config, attn_implementation="eager"
    ).eval()
    directory = output / "tiny"
    directory.mkdir(parents=True, exist_ok=True)
    save_weights(model, directory, {"lm_head.weight"}, 1)
    write_json(directory / "config.json", config.to_dict())

    extractor = Qwen3ASRFeatureExtractor(feature_size=TINY_AUDIO["num_mel_bins"])
    extracted = extractor(
        tone_and_chirp(SIGNAL_SAMPLES),
        sampling_rate=16_000,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    features, mask = extracted["input_features"], extracted["attention_mask"]
    with torch.no_grad():
        encoded = model.get_audio_features(features, mask).pooler_output
    prompt = [2, 3, 4] + [AUDIO_TOKEN] * encoded.shape[0] + [6, 7]
    cache = DynamicCache()
    with torch.no_grad():
        logits = model(
            input_ids=torch.tensor([prompt]),
            input_features=features,
            input_features_mask=mask,
            past_key_values=cache,
            use_cache=True,
        ).logits[0, -1]
    prompt_logits, tokens = logits, []
    for _ in range(NEW_TOKENS):
        token = int(logits.argmax())
        tokens.append(token)
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True
            ).logits[0, -1]
    write(
        directory,
        {
            "samples": torch.from_numpy(tone_and_chirp(SIGNAL_SAMPLES)),
            "features": features[0],
            "encoded": encoded,
            "prompt_logits": prompt_logits,
        },
        {
            "frames": int(mask.sum()),
            "prompt": prompt,
            "tokens": tokens,
        },
    )


def generate_real(output):
    processor = AutoProcessor.from_pretrained(REAL_MODEL)
    model = (
        Qwen3ASRForConditionalGeneration.from_pretrained(
            REAL_MODEL, dtype=torch.float32, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    samples, text = clip()
    (output / "real").mkdir(parents=True, exist_ok=True)
    (output / "real" / "clip.i16").write_bytes(samples.astype("<i2").tobytes())
    inputs = processor.apply_transcription_request(
        audio=samples.astype("float32") / 32_768.0
    ).to("cuda")
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=REAL_NEW_TOKENS, do_sample=False
        )
    prompt = inputs.input_ids[0].tolist()
    tokens = generated[0, len(prompt) :].tolist()
    parsed = processor.decode(tokens, return_format="parsed")
    write_json(
        output / "real" / "reference.json",
        {
            "model": REAL_MODEL,
            "clip_text": text,
            "prompt": prompt,
            "tokens": tokens,
            "language": parsed["language"],
            "transcript": parsed["transcription"],
        },
    )


OUTPUT = OUTPUT_ROOT / "automatic_speech_recognition/qwen3_asr"


def tiny(output):
    generate_tiny(output)


def real(output):
    generate_real(output)
