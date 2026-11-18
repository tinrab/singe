"""ColModernVBert, scoring document pages for text queries with `sentence-transformers`' multi-vector encoder, as `verify.rs` scores them in Singe.
`transformers` has no pipeline for visual document retrieval; the snapshot is served by `sentence-transformers`, which scores by late interaction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sentence_transformers import MultiVectorEncoder  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def visual_document_retrieval(snapshot):
    model = MultiVectorEncoder(
        snapshot, device=DEVICE, model_kwargs={"dtype": torch.float32}
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            queries = model.encode_query(item["queries"])
            documents = model.encode_document(
                [Image.open(path).convert("RGB") for path in item["documents"]]
            )
            outputs.append({"scores": model.similarity(queries, documents).tolist()})
        return outputs

    return run


verification = Verification(__file__)
verification.check("retrieval", visual_document_retrieval)
verification.finish()
