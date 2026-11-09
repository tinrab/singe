"""How `transformers`' tokenizers encode and decode a few texts for each real model, which Singe's tokenizers must match."""

from transformers import AutoTokenizer

from ..format import ROOT, write
from ..snapshots import snapshot

OUTPUT = ROOT / "singe-model/tests/references/tokenization"

TEXTS = [
    "Hello world.",
    "Singe runs models on the GPU, and its tokenizers match transformers'.",
    "Café naïve — 東京, 2.5 × 10⁶ tokens!",
    "",
]
# A batch an encoder reads, of different lengths, the longest past `MAX_LENGTH` tokens.
BATCH = [
    "A short one.",
    "A sentence that is long enough to be cut off by the truncation it asks for.",
    "Two words",
]
MAX_LENGTH = 12


def real(output):
    for name in ["QWEN3", "MINILM", "WHISPER"]:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot(
                name,
                [
                    "tokenizer*",
                    "vocab*",
                    "merges.txt",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "normalizer.json",
                    "config.json",
                ],
            )
        )
        cases = []
        for text in TEXTS:
            plain = tokenizer(text, add_special_tokens=False).input_ids
            special = tokenizer(text).input_ids
            cases.append(
                {
                    "text": text,
                    "plain": plain,
                    "special": special,
                    "decoded": tokenizer.decode(special, skip_special_tokens=True),
                }
            )
        batch = tokenizer(BATCH, padding=True, truncation=True, max_length=MAX_LENGTH)
        write(
            output / name.lower(),
            {},
            {
                "model": name,
                "cases": cases,
                "batch": {
                    "texts": BATCH,
                    "max_length": MAX_LENGTH,
                    "ids": batch.input_ids,
                    "token_types": batch.get(
                        "token_type_ids", [[0] * len(row) for row in batch.input_ids]
                    ),
                    "padding": batch.attention_mask,
                },
            },
        )
