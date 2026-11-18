"""Donut, answering questions about a document image with `transformers`' document-question-answering pipeline, as `verify.rs` answers them in Singe.
The pipeline is given the model, tokenizer, and image processor, since loading a `VisionEncoderDecoderModel` by name passes it a `code_revision` it does not take."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    pipeline,
)

from support import DEVICE, Verification  # noqa: E402


def document_question_answering(snapshot):
    answer = pipeline(
        "document-question-answering",
        model=VisionEncoderDecoderModel.from_pretrained(snapshot, dtype=torch.float32),
        tokenizer=AutoTokenizer.from_pretrained(snapshot),
        image_processor=AutoImageProcessor.from_pretrained(snapshot),
        device=DEVICE,
    )
    return lambda inputs, parameters: [
        answer(image=item["image"], question=item["question"], **parameters)
        for item in inputs
    ]


verification = Verification(__file__)
verification.check("invoice", document_question_answering)
verification.finish()
