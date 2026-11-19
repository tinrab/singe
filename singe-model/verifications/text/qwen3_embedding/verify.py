"""Qwen3-Embedding, scoring sentences' similarity to a source sentence with `sentence-transformers` and giving each token's features with `transformers`' feature-extraction pipeline, as `mod.rs` scores them in Singe."""

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sentence_transformers.util import cos_sim

from support import DEVICE, Verification


def sentence_similarity(snapshot, dtype):
    model = SentenceTransformer(snapshot, device=DEVICE, model_kwargs={"dtype": dtype})

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


def feature_extraction(snapshot, dtype):
    extract = pipeline("feature-extraction", model=snapshot, dtype=dtype, device=DEVICE)
    return lambda texts, parameters: [extract(text)[0] for text in texts]


verification = Verification(__file__)
verification.check("similarity", sentence_similarity)
verification.check("features", feature_extraction)
verification.finish()
