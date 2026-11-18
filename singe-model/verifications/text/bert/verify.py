"""BERT, through every task a BERT snapshot here serves, as `verify.rs` runs them in Singe: a toxicity classifier, a spam classifier, a cross-encoder ranking texts, a named-entity tagger with each way of grouping words, a sentence embedder, and its embeddings pooled by the mean over the square root of the length in a snapshot derived from it, and the pretrained masked language model filling blanks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402
from sentence_transformers.util import cos_sim  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def text_classification(snapshot, dtype=torch.float32):
    classify = pipeline(
        "text-classification", model=snapshot, dtype=dtype, device=DEVICE
    )
    # Without `top_k`, `transformers` gives a single text's best label in a list.
    return lambda texts, parameters: [
        classify(text, **parameters) if "top_k" in parameters else classify(text)[0]
        for text in texts
    ]


def text_ranking(snapshot):
    model = CrossEncoder(snapshot, device=DEVICE)
    return lambda inputs, parameters: [
        model.rank(item["query"], item["texts"]) for item in inputs
    ]


def token_classification(snapshot):
    tag = pipeline(
        "token-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda texts, parameters: [tag(text, **parameters) for text in texts]


def feature_extraction(snapshot):
    extract = pipeline(
        "feature-extraction", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda texts, parameters: [extract(text)[0] for text in texts]


def sentence_similarity(snapshot):
    model = SentenceTransformer(snapshot, device=DEVICE)

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


def fill_mask(snapshot):
    fill = pipeline("fill-mask", model=snapshot, dtype=torch.float32, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


def sentence_embeddings(snapshot):
    model = SentenceTransformer(snapshot, device=DEVICE)
    return lambda texts, parameters: model.encode(texts).tolist()


verification = Verification(__file__)
verification.check("toxicity", text_classification)
verification.check("spam", text_classification)
verification.check(
    "toxicity-bf16", lambda snapshot: text_classification(snapshot, torch.bfloat16)
)
verification.check("ranking", text_ranking)
for strategy in ("none", "simple", "first", "average", "max"):
    verification.check(f"entities-{strategy}", token_classification)
verification.check("features", feature_extraction)
verification.check("similarity", sentence_similarity)
verification.check("embeddings-sqrt", sentence_embeddings)
verification.check("fill", fill_mask)
verification.finish()
