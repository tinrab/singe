"""ColModernVBert, scoring document pages for text queries with `sentence-transformers`' multi-vector encoder, as `mod.rs` scores them in Singe.
`transformers` has no pipeline for visual document retrieval; the snapshot is served by `sentence-transformers`, which scores by late interaction."""

from PIL import Image
from sentence_transformers import MultiVectorEncoder

from support import DEVICE, Verification


def visual_document_retrieval(snapshot, dtype):
    model = MultiVectorEncoder(snapshot, device=DEVICE, model_kwargs={"dtype": dtype})

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
