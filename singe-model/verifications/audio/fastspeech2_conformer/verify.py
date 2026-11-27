"""FastSpeech2Conformer with HiFi-GAN, speaking ARPAbet phonemes with `transformers`' `FastSpeech2ConformerWithHifiGan`, as `mod.rs` speaks them in Singe.
The phonemes go through the vocabulary with the end token after them, as the tokenizer maps what its grapheme-to-phoneme model gives, since neither side runs `g2p_en`."""

import json
from pathlib import Path

import torch
from transformers import FastSpeech2ConformerWithHifiGan

from support import DEVICE, Verification


def text_to_audio(snapshot, dtype):
    vocabulary = json.loads((Path(snapshot) / "vocab.json").read_text())
    model = (
        FastSpeech2ConformerWithHifiGan.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(texts, parameters):
        outputs = []
        for text in texts:
            ids = [
                vocabulary.get(phoneme, vocabulary["<unk>"]) for phoneme in text.split()
            ]
            ids.append(vocabulary["<sos/eos>"])
            with torch.no_grad():
                waveform = model(torch.tensor([ids], device=DEVICE)).waveform[0]
            outputs.append(
                {
                    "audio": waveform.float().cpu().tolist(),
                    "sampling_rate": model.config.vocoder_config.sampling_rate,
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("speech", text_to_audio)
verification.finish()
