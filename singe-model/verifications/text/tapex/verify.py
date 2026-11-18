"""TAPEX, answering questions about a table with `transformers`' BART tokenizer and `generate`, after flattening the table as the table-question-answering pipeline's `TapexTokenizer` did before `transformers` 5 removed it, and as `verify.rs` answers them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import BartForConditionalGeneration, BartTokenizer  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def linearized(table, query):
    """The query and the table as `TapexTokenizer` joined them: `col : ` and the headers, then each `row N : ` and its cells, joined by ` | `, lowercased."""
    headers = list(table)
    rows = zip(*(table[header] for header in headers))
    text = "col : " + " | ".join(headers)
    for index, row in enumerate(rows, start=1):
        text += f" row {index} : " + " | ".join(row)
    return (query + " " + text).lower()


def table_question_answering(snapshot):
    tokenizer = BartTokenizer.from_pretrained(snapshot)
    model = (
        BartForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            encoded = tokenizer(
                linearized(item["table"], item["query"]), return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                generated = model.generate(**encoded, **parameters)
            outputs.append(
                {"answer": tokenizer.decode(generated[0], skip_special_tokens=True)}
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("machines", table_question_answering)
verification.finish()
