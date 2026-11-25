"""Runs a Singe ONNX export beside the reference library's model on a case's inputs and compares their outputs, as `optimum`'s `validate_model_outputs` does: output names, then shapes, then values.

`python onnx_check.py MODEL SNAPSHOT TASK INPUTS DTYPE` prints a JSON report: whether every output matched, what each comparison found, and the runtime.
Each input runs alone, then all of them padded into one batch, so neither shape is the one the export was built at.
Images of different sizes run only alone when the processor keeps each one's size and pads none, as DPT's does.
A clip runs alone, its first half alone, and twice in one batch.
Values at padded tokens are left out, since the two models need not agree there.
An export whose first input has every axis fixed, as `ExportOptions::input_shape` writes it, runs only batches of its size, with images resized bilinearly to its height and width and log-mel features cut or padded to its frames, the reference reading the same adapted inputs.
A float32 export is compared with the reference run in float64, and the tolerance scales with the output's largest value, since float32 rounding in a deep model grows with it.
The export runs in ONNX Runtime, or in ONNX's `ReferenceEvaluator` when ONNX Runtime has no kernel for one of its operators.
When an output disagrees, the batch's inputs go to `mismatch.json` beside the export, from which the verification finds the first value where the export disagrees with Singe's native program, and `diagnose` runs the export with every node's output exposed in ONNX Runtime and in `ReferenceEvaluator` and reports the first node whose values differ, with the Singe operation the exporter wrote in its `doc_string`.
If none differs, the runtime computes what the ONNX operators define, and the export itself computes something other than the reference library.
"""

import json
import os
import sys

import numpy as np
import onnx
import torch
import transformers
from PIL import Image

# The largest difference allowed, as a fraction of the output's largest value or of 1, whichever is larger.
TOLERANCE = {"f32": 1e-4, "f16": 1e-2, "bf16": 5e-2}

TEXT = {
    "text-classification": ("AutoModelForSequenceClassification", ["logits"]),
    "text-ranking": ("AutoModelForSequenceClassification", ["logits"]),
    "zero-shot-classification": ("AutoModelForSequenceClassification", ["logits"]),
    "token-classification": ("AutoModelForTokenClassification", ["logits"]),
    "fill-mask": ("AutoModelForMaskedLM", ["logits"]),
    "question-answering": (
        "AutoModelForQuestionAnswering",
        ["start_logits", "end_logits"],
    ),
    "feature-extraction": ("AutoModel", ["last_hidden_state"]),
}
AUDIO = {
    "audio-classification": ("AutoModelForAudioClassification", ["logits"]),
    "automatic-speech-recognition": ("AutoModelForCTC", ["logits"]),
}
IMAGE = {
    "image-classification": ("AutoModelForImageClassification", ["logits"]),
    "image-feature-extraction": ("AutoModel", ["last_hidden_state", "pooler_output"]),
    "image-segmentation": ("AutoModelForSemanticSegmentation", ["logits"]),
    "object-detection": ("AutoModelForObjectDetection", ["logits", "pred_boxes"]),
    "depth-estimation": ("AutoModelForDepthEstimation", ["predicted_depth"]),
}


def session(path):
    """A function from named inputs to named outputs, and the runtime that runs it."""
    try:
        import onnxruntime

        runner = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        return (
            lambda inputs: dict(
                zip([o.name for o in runner.get_outputs()], runner.run(None, inputs))
            )
        ), "onnxruntime"
    except Exception as error:
        if "NOT_IMPLEMENTED" not in str(error):
            raise
        from onnx.reference import ReferenceEvaluator

        evaluator = ReferenceEvaluator(path)
        return (
            lambda inputs: dict(
                zip(evaluator.output_names, evaluator.run(None, inputs))
            )
        ), "reference"


def diagnose(path, feeds, tolerance):
    """The first node of the export whose outputs ONNX Runtime and ONNX's reference implementation disagree on, or that none does."""
    import tempfile

    import onnxruntime
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(path)
    known = {output.name for output in model.graph.output}
    for node in model.graph.node:
        for name in node.output:
            if name and name not in known:
                model.graph.output.append(onnx.ValueInfoProto(name=name))
                known.add(name)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(path)) as directory:
        exposed = os.path.join(directory, "exposed.onnx")
        onnx.save(
            model, exposed, save_as_external_data=True, location="exposed.onnx_data"
        )
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        runner = onnxruntime.InferenceSession(
            exposed, options, providers=["CPUExecutionProvider"]
        )
        names = [output.name for output in runner.get_outputs()]
        runtime = dict(zip(names, runner.run(None, feeds)))
        reference = ReferenceEvaluator(exposed).run(None, feeds, intermediate=True)
    for node in model.graph.node:
        for name in node.output:
            if name not in runtime or name not in reference:
                continue
            actual = np.asarray(runtime[name])
            expected = np.asarray(reference[name])
            if actual.shape != expected.shape:
                return f"first differing node {node.name}, {node.op_type} from {node.doc_string or 'no Singe operation'}: shape {list(actual.shape)} against {list(expected.shape)}"
            if not np.issubdtype(actual.dtype, np.number):
                continue
            worst = (
                float(
                    np.abs(
                        actual.astype(np.float64) - expected.astype(np.float64)
                    ).max()
                )
                if actual.size
                else 0.0
            )
            allowed = tolerance * max(
                1.0,
                float(np.abs(expected.astype(np.float64)).max())
                if expected.size
                else 1.0,
            )
            if worst > allowed:
                return f"first differing node {node.name}, {node.op_type} from {node.doc_string or 'no Singe operation'}: max difference {worst:.2e} of {allowed:.2e} allowed"
    return "onnxruntime and the onnx reference implementation agree at every node, so the export computes something other than the reference library"


def texts(task, inputs):
    """The case's inputs as the texts, or pairs of texts, the tokenizer reads."""
    if task == "question-answering":
        return [(item["question"], item["context"]) for item in inputs]
    if task == "text-ranking":
        return [(item["query"], text) for item in inputs for text in item["texts"]]
    if task == "zero-shot-classification":
        # Each label becomes a hypothesis paired with the sequence, with the template `transformers` uses by default.
        return [
            (item["sequences"], f"This example is {label}.")
            for item in inputs
            for label in item["candidate_labels"]
        ]
    if task == "sentence-similarity":
        return [
            text
            for item in inputs
            for text in [item["source_sentence"], *item["sentences"]]
        ]
    return inputs


def encode(tokenizer, items, names, shape=None):
    """The tokens of `items`, padded to the longest, or to the fixed `shape`'s length."""
    pairs = isinstance(items[0], tuple)
    length = {} if shape is None else {"max_length": shape[1]}
    encoded = tokenizer(
        [item[0] for item in items] if pairs else items,
        [item[1] for item in items] if pairs else None,
        padding="max_length" if length else True,
        truncation=True,
        return_tensors="np",
        **length,
    )
    return {name: encoded[name].astype(np.int64) for name in names if name in encoded}


def reference_text(snapshot, task, dtype):
    if task == "sentence-similarity":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            snapshot, device="cpu", model_kwargs={"dtype": dtype}
        )

        def run(feeds):
            features = {name: torch.from_numpy(value) for name, value in feeds.items()}
            with torch.no_grad():
                out = model(features)
            return {
                name: out[name].double().numpy()
                for name in ["token_embeddings", "sentence_embedding"]
            }

        return model.tokenizer, ["token_embeddings", "sentence_embedding"], run
    class_name, outputs = TEXT[task]
    model = (
        getattr(transformers, class_name).from_pretrained(snapshot, dtype=dtype).eval()
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot)

    def run(feeds):
        with torch.no_grad():
            out = model(
                **{name: torch.from_numpy(value) for name, value in feeds.items()}
            )
        return {name: getattr(out, name).double().numpy() for name in outputs}

    return tokenizer, outputs, run


def compare(name, actual, expected, mask, tolerance):
    if actual.shape != expected.shape:
        return (
            False,
            f"{name}: shape {list(actual.shape)} against {list(expected.shape)}",
        )
    expected = expected.astype(np.float64)
    difference = np.abs(actual.astype(np.float64) - expected)
    if mask is not None and difference.ndim >= 2 and difference.shape[:2] == mask.shape:
        difference = difference[mask.astype(bool)]
    worst = float(difference.max()) if difference.size else 0.0
    allowed = tolerance * max(
        1.0, float(np.abs(expected).max()) if expected.size else 1.0
    )
    return (
        worst <= allowed,
        f"{name} {list(actual.shape)}: max difference {worst:.2e} of {allowed:.2e} allowed",
    )


def fixed_shape(graph):
    """The first input's shape when every axis of it is fixed, and `None` when one is dynamic."""
    dimensions = graph.input[0].type.tensor_type.shape.dim
    if all(dimension.HasField("dim_value") for dimension in dimensions):
        return [dimension.dim_value for dimension in dimensions]
    return None


def fitted(values, shape):
    """`values`, pixels `[B, 3, H, W]`, features `[B, frames, mels]` or samples `[B, N]`, adapted to the fixed `shape`."""
    if shape is None or list(values.shape) == shape:
        return values
    if values.ndim == 2:
        values = values[:, : shape[1]]
        return np.pad(values, [(0, 0), (0, shape[1] - values.shape[1])])
    if values.ndim == 4:
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(values), size=shape[2:], mode="bilinear"
        )
        return resized.numpy()
    frames = shape[1]
    values = values[:, :frames]
    return np.pad(values, [(0, 0), (0, frames - values.shape[1]), (0, 0)])


def main():
    path, snapshot, task, inputs_path, data_type = sys.argv[1:6]
    inputs = json.load(open(inputs_path))
    dtype = {"f32": torch.float64, "f16": torch.float16, "bf16": torch.bfloat16}[
        data_type
    ]
    tolerance = TOLERANCE[data_type]
    run_export, runtime = session(path)
    exported = [
        output.name for output in onnx.load(path, load_external_data=False).graph.output
    ]
    graph = onnx.load(path, load_external_data=False).graph
    names = [value.name for value in graph.input]
    shape = fixed_shape(graph)
    if task in AUDIO:
        import soundfile

        class_name, outputs = AUDIO[task]
        extractor = transformers.AutoFeatureExtractor.from_pretrained(snapshot)
        model = (
            getattr(transformers, class_name)
            .from_pretrained(snapshot, dtype=dtype)
            .eval()
        )
        clips = []
        for item in inputs:
            samples, rate = soundfile.read(item, dtype="float32", always_2d=True)
            clips.append(samples.mean(axis=1))
        batches = [[clip] for clip in clips] + [
            [clips[0][: len(clips[0]) // 2]],
            [clips[0], clips[0]],
        ]

        # wav2vec2 reads samples as `input_values`, and Parakeet log-mel features as `input_features`.
        (feature,) = names

        def feeds_of(batch):
            features = extractor(batch, sampling_rate=rate, return_tensors="np")
            return {feature: fitted(features[feature].astype(np.float32), shape)}

        def reference(feeds):
            with torch.no_grad():
                out = model(**{feature: torch.from_numpy(feeds[feature]).to(dtype)})
            return {name: getattr(out, name).double().numpy() for name in outputs}
    elif task in IMAGE:
        class_name, outputs = IMAGE[task]
        processor = transformers.AutoImageProcessor.from_pretrained(snapshot)
        model = (
            getattr(transformers, class_name)
            .from_pretrained(snapshot, dtype=dtype)
            .eval()
        )
        images = [Image.open(item).convert("RGB") for item in inputs]
        batches = [[image] for image in images]
        try:
            processor(images=images, return_tensors="np")
            batches.append(images)
        except ValueError:
            pass

        def feeds_of(batch):
            pixels = processor(images=batch, return_tensors="np")[
                "pixel_values"
            ].astype(np.float32)
            return {"pixel_values": fitted(pixels, shape)}

        def reference(feeds):
            with torch.no_grad():
                out = model(
                    pixel_values=torch.from_numpy(feeds["pixel_values"]).to(dtype)
                )
            return {name: getattr(out, name).double().numpy() for name in outputs}
    else:
        tokenizer, outputs, reference = reference_text(snapshot, task, dtype)
        items = texts(task, inputs)
        batches = [[item] for item in items] + [items]

        def feeds_of(batch):
            return encode(tokenizer, batch, names, shape)

    if shape is not None:
        batches = [batch for batch in batches if len(batch) == shape[0]]
    matched = exported == outputs
    details = [] if matched else [f"outputs {exported} against {outputs}"]
    diagnosed = False
    mismatch = None
    for batch in batches:
        feeds = feeds_of(batch)
        actual = run_export(feeds)
        expected = reference(feeds)
        mask = feeds.get("attention_mask")
        agreed = True
        for name in outputs:
            same, detail = compare(
                name, np.asarray(actual[name]), expected[name], mask, tolerance
            )
            agreed &= same
            if not same or batch is batches[-1]:
                details.append(f"batch {len(batch)}: {detail}")
        matched &= agreed
        if not agreed and not diagnosed:
            details.append(diagnose(path, feeds, tolerance))
            diagnosed = True
            mismatch = os.path.join(os.path.dirname(path), "mismatch.json")
            with open(mismatch, "w") as file:
                json.dump(
                    {
                        name: {
                            "dtype": str(value.dtype),
                            "shape": list(value.shape),
                            "values": value.ravel().tolist(),
                        }
                        for name, value in feeds.items()
                    },
                    file,
                )
    print(
        json.dumps(
            {
                "matched": bool(matched),
                "runtime": runtime,
                "details": details,
                "mismatch": mismatch,
            }
        )
    )


if __name__ == "__main__":
    main()
