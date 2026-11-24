"""Parakeet, transcribing a clip, as WAV and as FLAC, its first second, and the clip 8, 50, and 150 times over with a TDT transducer, the 8 times also attending locally with a window wider than the clip, an RNN-T transducer, and a CTC head with `transformers`' `generate`, as `mod.rs` does each in Singe.

Above 8,192 rows Singe's Parakeet attends locally, as NeMo's `rel_pos_local_attn` does, which `transformers` does not, so each encoder layer's attention is replaced there by `local_attention`.
"""

import torch
from transformers import AutoProcessor, ParakeetForCTC, ParakeetForRNNT, ParakeetForTDT

from support import DEVICE, Verification, read_audio


# Each row attends to the rows at most `WINDOW` from it once a clip has more than `ABOVE` rows.
WINDOW = 128
ABOVE = 8192

# Queries per chunk of `local_attention`.
CHUNK = 512


def local_attention(attention, hidden_states, position_embeddings, attention_mask):
    """Relative attention from `ParakeetEncoderAttention`'s weights, scoring each chunk of queries against the keys up to `WINDOW` rows from it and hiding the farther ones."""
    batch, steps, _ = hidden_states.shape
    heads, depth = attention.config.num_attention_heads, attention.head_dim
    shape = (batch, steps, heads, depth)
    query = attention.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = attention.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
    content = query + attention.bias_u.view(1, heads, 1, depth)
    positional = query + attention.bias_v.view(1, heads, 1, depth)
    # The distances `steps - 1` down to `-(steps - 1)`, `[1, heads, depth, 2 · steps - 1]`.
    relative = attention.relative_k_proj(position_embeddings)
    relative = relative.view(relative.shape[0], -1, heads, depth).permute(0, 2, 3, 1)
    outputs = []
    for start in range(0, steps, CHUNK):
        end = min(start + CHUNK, steps)
        low, high = max(0, start - WINDOW), min(steps, end + WINDOW)
        rows = torch.arange(start, end, device=hidden_states.device)[:, None]
        keys = torch.arange(low, high, device=hidden_states.device)[None, :]
        # Query `i`'s distance to key `j` is column `steps - 1 - i + j`.
        columns = (steps - 1 - rows + keys).expand(batch, heads, -1, -1)
        distance = (positional[:, :, start:end] @ relative).gather(-1, columns)
        scores = (
            content[:, :, start:end] @ key[:, :, low:high].transpose(-1, -2) + distance
        ) * attention.scaling
        hidden = (rows - keys).abs() > WINDOW
        if attention_mask is not None:
            hidden = hidden | attention_mask[:, :, start:end, low:high].logical_not()
        scores = scores.masked_fill(hidden, float("-inf"))
        outputs.append(scores.softmax(-1) @ value[:, :, low:high])
    output = torch.cat(outputs, 2).transpose(1, 2).reshape(batch, steps, -1)
    return attention.o_proj(output), None


def attend_locally(model):
    """Replaces each encoder layer's attention with `local_attention` on clips of more than `ABOVE` rows."""
    for layer in model.encoder.layers:
        attention = layer.self_attn
        full = attention.forward

        def forward(
            hidden_states,
            position_embeddings,
            attention_mask=None,
            attention=attention,
            full=full,
            **kwargs,
        ):
            # Singe counts the clip's own rows, which the mask marks, where `transformers` pads one frame.
            rows = (
                hidden_states.shape[1]
                if attention_mask is None
                else int(attention_mask[:, 0].any(-1).sum(-1).max())
            )
            if rows <= ABOVE:
                return full(
                    hidden_states, position_embeddings, attention_mask, **kwargs
                )
            return local_attention(
                attention, hidden_states, position_embeddings, attention_mask
            )

        attention.forward = forward


def automatic_speech_recognition(model_class):
    def reference(snapshot, dtype):
        processor = AutoProcessor.from_pretrained(snapshot)
        model = model_class.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
        attend_locally(model)

        def run(paths, parameters):
            outputs = []
            for path in paths:
                samples, rate = read_audio(path)
                inputs = processor(samples, sampling_rate=rate).to(DEVICE, dtype=dtype)
                with torch.no_grad():
                    generated = model.generate(**inputs, **parameters)
                sequences = getattr(generated, "sequences", generated)
                text = processor.decode(sequences[0], skip_special_tokens=True)
                outputs.append({"text": text.strip()})
            return outputs

        return run

    return reference


verification = Verification(__file__)
verification.check("tdt", automatic_speech_recognition(ParakeetForTDT))
verification.check("tdt-flac", automatic_speech_recognition(ParakeetForTDT))
verification.check("tdt-short", automatic_speech_recognition(ParakeetForTDT))
verification.check("tdt-long", automatic_speech_recognition(ParakeetForTDT))
verification.check("tdt-long-local", automatic_speech_recognition(ParakeetForTDT))
verification.check("tdt-4m", automatic_speech_recognition(ParakeetForTDT))
verification.check("tdt-12m", automatic_speech_recognition(ParakeetForTDT))
verification.check("rnnt", automatic_speech_recognition(ParakeetForRNNT))
verification.check("ctc", automatic_speech_recognition(ParakeetForCTC))
verification.finish()
