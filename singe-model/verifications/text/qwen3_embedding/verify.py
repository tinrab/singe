"""Qwen3-Embedding, scoring sentences' similarity to a source sentence with `sentence-transformers` and giving each token's features with `transformers`' feature-extraction pipeline, as `verify.rs` scores them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import pipeline  # noqa: E402
from sentence_transformers.util import cos_sim  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def sentence_similarity(snapshot):
    model = SentenceTransformer(
        snapshot, device=DEVICE, model_kwargs={"dtype": torch.float32}
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            embeddings = model.encode(
                [item["source_sentence"], *item["sentences"]], convert_to_tensor=True
            )
            outputs.append(
                {"scores": cos_sim(embeddings[:1], embeddings[1:])[0].tolist()}
            )
        return outputs

    return run


def feature_extraction(snapshot):
    extract = pipeline(
        "feature-extraction", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda texts, parameters: [extract(text)[0] for text in texts]


verification = Verification(__file__)
verification.check("similarity", sentence_similarity)
verification.check("features", feature_extraction)
verification.finish()
