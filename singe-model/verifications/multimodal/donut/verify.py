"""Donut, answering questions about a document image with `transformers`' document-question-answering pipeline, as `mod.rs` answers them in Singe.
The pipeline is given the model, tokenizer, and image processor, since loading a `VisionEncoderDecoderModel` by name passes it a `code_revision` it does not take."""

from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    pipeline,
)

from support import DEVICE, Verification


def document_question_answering(snapshot, dtype):
    answer = pipeline(
        "document-question-answering",
        model=VisionEncoderDecoderModel.from_pretrained(snapshot, dtype=dtype),
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
