"""DistilBERT, through every task a DistilBERT snapshot here serves, as `verify.rs` runs them in Singe: a sentiment classifier, an inference model classifying zero-shot, a named-entity tagger, a question answerer over short and long contexts, the pretrained masked language model filling blanks, and a sentence embedder that max-pools its tokens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sentence_transformers.util import cos_sim  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def text_classification(snapshot):
    classify = pipeline(
        "text-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda texts, parameters: [classify(text, **parameters) for text in texts]


def zero_shot_classification(snapshot):
    classify = pipeline(
        "zero-shot-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda inputs, parameters: [
        classify(item["sequences"], item["candidate_labels"], **parameters)
        for item in inputs
    ]


def token_classification(snapshot):
    tag = pipeline(
        "token-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda texts, parameters: [tag(text, **parameters) for text in texts]


def question_answering(snapshot, dtype=torch.float32):
    """`transformers` 5 removed the question-answering pipeline, so this is its model with `transformers` 4.57's span selection, ported from `pipelines/question_answering.py`: windows of the context, softmaxed start and end scores over the context, spans of at most `max_answer_len` tokens aligned to words, and one answer's scores summed across windows."""
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForQuestionAnswering.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def answers(item, parameters):
        top_k = parameters.get("top_k", 1)
        max_answer_len = parameters.get("max_answer_len", 15)
        max_seq_len = parameters.get(
            "max_seq_len", min(tokenizer.model_max_length, 384)
        )
        stride = parameters.get("doc_stride", min(max_seq_len // 2, 128))
        encoded = tokenizer(
            item["question"],
            item["context"],
            truncation="only_second",
            max_length=max_seq_len,
            stride=stride,
            padding=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        inputs = {
            name: encoded[name].to(DEVICE) for name in tokenizer.model_input_names
        }
        with torch.no_grad():
            output = model(**inputs)
        found, null = [], 1e6
        for window in range(encoded["input_ids"].shape[0]):
            ids = encoded["input_ids"][window].tolist()
            mask = encoded["attention_mask"][window].numpy().astype(bool)
            sequences = encoded.sequence_ids(window)
            # The context may answer, and so may the classifier token, whose scores the softmax counts.
            answerable = (
                numpy.array(
                    [
                        s == 1 or i == tokenizer.cls_token_id
                        for s, i in zip(sequences, ids)
                    ]
                )
                & mask
            )

            def normalized(logits):
                logits = numpy.where(
                    answerable, logits[window].float().cpu().numpy(), -10000.0
                )
                scores = numpy.exp(logits - logits.max())
                return scores / scores.sum()

            start, end = normalized(output.start_logits), normalized(output.end_logits)
            null = min(null, float(start[0] * end[0]))
            start[0] = end[0] = 0.0
            candidates = numpy.tril(
                numpy.triu(numpy.outer(start, end)), max_answer_len - 1
            ).flatten()
            for flat in numpy.argsort(-candidates, kind="stable")[: top_k * 2 + 10]:
                first, last = divmod(int(flat), len(ids))
                if not (
                    (first == 0 or answerable[first])
                    and (last == 0 or answerable[last])
                ):
                    continue
                words = encoded.word_ids(window)
                try:
                    span = (
                        encoded.word_to_chars(window, words[first], sequence_index=1)[
                            0
                        ],
                        encoded.word_to_chars(window, words[last], sequence_index=1)[1],
                    )
                except Exception:
                    offsets = encoded["offset_mapping"][window]
                    span = (int(offsets[first][0]), int(offsets[last][1]))
                text = item["context"][span[0] : span[1]]
                same = next(
                    (
                        answer
                        for answer in found
                        if answer["answer"].lower() == text.lower()
                    ),
                    None,
                )
                if same:
                    same["score"] += float(candidates[flat])
                else:
                    found.append(
                        {
                            "score": float(candidates[flat]),
                            "start": span[0],
                            "end": span[1],
                            "answer": text,
                        }
                    )
        if parameters.get("handle_impossible_answer", False):
            found.append({"score": null, "start": 0, "end": 0, "answer": ""})
        found = sorted(found, key=lambda answer: answer["score"], reverse=True)[:top_k]
        return found[0] if len(found) == 1 else found

    return lambda inputs, parameters: [answers(item, parameters) for item in inputs]


def fill_mask(snapshot):
    fill = pipeline("fill-mask", model=snapshot, dtype=torch.float32, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


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


verification = Verification(__file__)
verification.check("sentiment", text_classification)
verification.check("zero-shot", zero_shot_classification)
verification.check("zero-shot-multi-label", zero_shot_classification)
verification.check("entities-simple", token_classification)
verification.check("entities-first", token_classification)
verification.check("answers", question_answering)
verification.check("answers-small-windows", question_answering)
verification.check(
    "answers-bf16", lambda snapshot: question_answering(snapshot, torch.bfloat16)
)
verification.check("fill", fill_mask)
verification.check("similarity-max", sentence_similarity)
verification.finish()
