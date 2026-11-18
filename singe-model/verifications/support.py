"""What every verification's Python side shares: its cases, comparing outputs with the expected ones, timing, and the report.

A model's `verify.py` holds the reference code itself, written as `transformers`, `sentence-transformers`, or `diffusers` would be used, so it reads beside `verify.rs`.
Each of its functions takes a snapshot, loads the reference in the type `verify.rs` loads Singe's pipeline in, and returns what runs a case's inputs with its parameters.
`Verification.check` runs it, compares its outputs with `cases.json`'s expected ones by the rules `support/compare.rs` compares Singe's by, and times it.
`verify.py --write` records the reference's outputs as the expected ones instead.
A script exits 1 when a case failed, and `NOTHING_RAN` when its filter matched no case or every matching case was skipped.
"""

import base64
import gc
import io
import json
import math
import os
import statistics
import sys
import time
import tomllib
import traceback
import wave
from pathlib import Path

import numpy
import torch

ROOT = Path(__file__).resolve().parent
# The references run on the GPU, as Singe's pipelines do, and the timing synchronizes and profiles CUDA.
DEVICE = "cuda"
TOLERANCE = 1e-3
# The runs a case is timed over, after one that warms it up.
TIMED_RUNS = 5
# The exit status of a script that ran no case, which `just verify` tells apart from a failure.
NOTHING_RAN = 3

# The references run without cuDNN, whose libraries on the path may not be the ones torch was built against.
torch.backends.cudnn.enabled = False

# Loading reports and progress bars would bury the results, and the reports name local paths.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from transformers.utils import logging  # noqa: E402

logging.set_verbosity_error()
logging.disable_progress_bar()


def media(value):
    """Rewrites each string naming a file under `media/` to its path, as cases name the clips and images they read."""
    if isinstance(value, str) and value.startswith("media/"):
        return str(ROOT / value)
    if isinstance(value, list):
        return [media(item) for item in value]
    if isinstance(value, dict):
        return {key: media(item) for key, item in value.items()}
    return value


def plain(value):
    """Turns NumPy and torch scalars into Python's, so outputs compare and serialize as JSON."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def profiled(work):
    """Runs `work`, returning what it gave, its wall time, and the GPU's work during it: kernel seconds, kernels launched, and copy seconds, from torch's profiler."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        start = time.perf_counter()
        value = work()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
    kernels = copies = 0.0
    launches = 0
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        seconds = event.time_range.elapsed_us() / 1e6
        if event.name.startswith(("Memcpy", "Memset")):
            copies += seconds
        else:
            kernels += seconds
            launches += 1
    return value, wall, {"kernels": kernels, "launches": launches, "copies": copies}


def duration(seconds):
    """A duration in milliseconds below ten seconds, and in seconds above, as `support/mod.rs` writes it."""
    return f"{seconds * 1e3:.1f} ms" if seconds < 10 else f"{seconds:.2f} s"


def report(load, load_gpu, first, run, run_gpu):
    busy = run_gpu["kernels"] + run_gpu["copies"]
    return (
        f"load {duration(load)} ({duration(load_gpu['copies'])} copying to the GPU), first run {duration(first)}, "
        f"run {duration(run)}: GPU {duration(busy)} in {run_gpu['launches']} kernels, host {duration(max(run - busy, 0.0))}"
    )


def repositories():
    """The Hugging Face repository of each snapshot name, from `snapshots.toml`."""
    return tomllib.loads((ROOT / "snapshots.toml").read_text())["snapshots"]


def snapshot_entry(name):
    """Snapshot `name`'s repository, revision, and the files to download, from `snapshots.toml`: an entry is the repository, or a table of its `repository`, the `revision` to read (such as the safetensors conversion bot's `refs/pr/N`), and the `files` patterns to keep instead of `KEPT`."""
    entry = repositories()[name]
    if isinstance(entry, str):
        return entry, "main", KEPT
    return entry["repository"], entry.get("revision", "main"), entry.get("files", KEPT)


def find_snapshot(name):
    """The directory of snapshot `name`: `SINGE_SNAPSHOT_<NAME>` when it is set, or else its repository's revision in the Hugging Face cache, as `support/snapshot.rs` finds it; `None` when neither has it."""
    if directory := os.environ.get(f"SINGE_SNAPSHOT_{name}"):
        return directory
    entry = repositories()[name]
    if isinstance(entry, dict) and "pipeline" in entry:
        return assembled(name, entry)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repository, revision, files = snapshot_entry(name)
    try:
        return snapshot_download(
            repository,
            revision=revision,
            allow_patterns=files,
            ignore_patterns=LEFT,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None


def assembled(name, entry):
    """A `diffusers` pipeline assembled from snapshots, as `support/snapshot.rs` assembles it: the `base` snapshot's components and the added `components` snapshots, linked under `target/tmp/snapshots/<NAME>` beside a `model_index.json` naming the `pipeline` class; `None` when a part is missing."""
    base = find_snapshot(entry["base"])
    added = {
        component: find_snapshot(part)
        for component, part in entry.get("components", {}).items()
    }
    if base is None or None in added.values():
        return None
    directory = ROOT.parents[1] / "target" / "tmp" / "snapshots" / name
    directory.mkdir(parents=True, exist_ok=True)
    index = json.loads((Path(base) / "model_index.json").read_text())
    index["_class_name"] = entry["pipeline"]
    parts = {path.name: path for path in Path(base).iterdir() if path.is_dir()}
    for component, part in added.items():
        index[component] = [
            "diffusers",
            json.loads((Path(part) / "config.json").read_text())["_class_name"],
        ]
        parts[component] = Path(part)
    # Components the base names but its download left out, such as a safety checker, are dropped.
    for component in [
        key
        for key, value in index.items()
        if isinstance(value, list) and key not in parts
    ]:
        del index[component]
    for component, part in parts.items():
        link = directory / component
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(part.resolve())
    (directory / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    return str(directory)


# What a snapshot download keeps: configs, tokenizers, chat templates, and safetensors weights, not the other formats' weights.
KEPT = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
LEFT = ["onnx/*", "openvino/*", "*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx"]


def write_media():
    """Writes the media cases read under `media/`, unless they are there: a LibriSpeech clip, a cat in three sizes, COCO's two cats, and a clip's frames, from Hugging Face's test datasets and COCO."""
    from huggingface_hub import hf_hub_download

    directory = ROOT / "media"
    directory.mkdir(exist_ok=True)
    clip = directory / "clip.wav"
    if not clip.exists():
        import pyarrow.parquet
        import soundfile

        path = hf_hub_download(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        row = pyarrow.parquet.read_table(path).to_pylist()[1]
        samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        soundfile.write(clip, samples, rate, subtype="PCM_16")
    second = directory / "clip-1s.wav"
    if not second.exists():
        import soundfile

        # The clip's first second, which keeps a codec's samples in `cases.json` few.
        samples, rate = soundfile.read(clip, dtype="int16")
        soundfile.write(second, samples[:rate], rate, subtype="PCM_16")
    high = directory / "clip-48k.wav"
    if not high.exists():
        import scipy.signal
        import soundfile

        # The clip at 48 kHz, the rate CLAP hears.
        samples, rate = soundfile.read(clip, dtype="int16")
        resampled = scipy.signal.resample_poly(
            samples.astype("float64"), 48_000 // rate, 1
        )
        soundfile.write(
            high,
            numpy.clip(numpy.round(resampled), -32768, 32767).astype("int16"),
            48_000,
            subtype="PCM_16",
        )
    coco = directory / "coco-cats.png"
    if not coco.exists():
        from huggingface_hub.utils import get_session
        from PIL import Image

        # Two cats and two remotes on a couch, the image `transformers`' detection examples read.
        # COCO's image host has no certificate of its own name, so the image is read over HTTPS from the S3 bucket behind it.
        response = get_session().get(
            "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000039769.jpg"
        )
        response.raise_for_status()
        Image.open(io.BytesIO(response.content)).convert("RGB").save(coco)
    spaghetti = directory / "spaghetti"
    if not spaghetti.exists():
        from PIL import Image

        # The clip `transformers`' VideoMAE tests read, already decoded to frames.
        path = hf_hub_download(
            "hf-internal-testing/spaghetti-video",
            "eating_spaghetti.npy",
            repo_type="dataset",
        )
        spaghetti.mkdir()
        for index, frame in enumerate(numpy.load(path)):
            Image.fromarray(frame).save(spaghetti / f"frame-{index:03}.png")
    short = directory / "spaghetti-8"
    if not short.exists():
        # The clip's first 8 frames, which video-to-video resizes to the size it is asked for.
        short.mkdir()
        for frame in sorted(spaghetti.glob("*.png"))[:8]:
            (short / frame.name).write_bytes(frame.read_bytes())
    sphere = directory / "sphere.png"
    if not sphere.exists():
        from PIL import Image

        # A shaded red ball on white, an object as Shap-E's image prior expects one.
        side, radius = 256, 90
        light = numpy.array([-0.4, -0.5, 0.77])
        rows, columns = numpy.mgrid[0:side, 0:side].astype(numpy.float64)
        x, y = (columns - side / 2) / radius, (rows - side / 2) / radius
        inside = x**2 + y**2 < 1
        z = numpy.sqrt(numpy.clip(1 - x**2 - y**2, 0, 1))
        shade = numpy.clip(x * light[0] + y * light[1] + z * light[2], 0, 1) * 0.8 + 0.2
        pixels = numpy.full((side, side, 3), 255, dtype=numpy.uint8)
        pixels[inside] = (
            (numpy.stack([shade * 230, shade * 40, shade * 40], axis=-1)[inside])
            .round()
            .astype(numpy.uint8)
        )
        Image.fromarray(pixels).save(sphere)
    invoice = directory / "invoice.png"
    if not invoice.exists():
        from PIL import Image, ImageDraw, ImageFont

        # A short invoice, as Donut's document questions read scanned forms.
        image = Image.new("RGB", (960, 1280), "white")
        draw = ImageDraw.Draw(image)
        lines = [
            ("INVOICE", 64),
            ("Invoice number: 4521", 36),
            ("Date: 12 March 2024", 36),
            ("Bill to: Northwind Traders", 36),
            ("Coffee beans, 2 kg ........ 24.00", 32),
            ("Delivery ................... 13.80", 32),
            ("TOTAL DUE: 37.80 EUR", 44),
        ]
        top = 80
        for text, size in lines:
            draw.text(
                (80, top), text, fill="black", font=ImageFont.load_default(size=size)
            )
            top += size + 60
        image.save(invoice)
    printed = directory / "printed-line.png"
    if not printed.exists():
        from PIL import Image, ImageDraw, ImageFont

        # A line of printed text on white, as TrOCR's printed-text models read receipts' lines.
        image = Image.new("RGB", (384, 48), "white")
        font = ImageFont.load_default(size=28)
        ImageDraw.Draw(image).text((8, 8), "TOTAL 12.50 EUR", fill="black", font=font)
        image.save(printed)
    sizes = [(448, 320), (320, 512), (128, 128)]
    if not all(
        (directory / f"cat-{width}x{height}.png").exists() for width, height in sizes
    ):
        from PIL import Image

        path = hf_hub_download(
            "huggingface/documentation-images",
            "pipeline-cat-chonk.jpeg",
            repo_type="dataset",
        )
        cat = Image.open(path).convert("RGB")
        for width, height in sizes:
            cat.resize((width, height), Image.BICUBIC).save(
                directory / f"cat-{width}x{height}.png"
            )


def download(names):
    from huggingface_hub import snapshot_download

    for name in names:
        entry = repositories()[name]
        if isinstance(entry, dict) and "pipeline" in entry:
            download([entry["base"], *entry.get("components", {}).values()])
            continue
        repository, revision, files = snapshot_entry(name)
        snapshot_download(
            repository,
            revision=revision,
            allow_patterns=files,
            ignore_patterns=LEFT,
        )
        print(f"{name} downloaded")


def rounded(value):
    """Keeps six significant digits of each float, far finer than any comparison's tolerance, so `cases.json` stays small."""
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


class Verification:
    """One model's cases, checked or written by `verify.py`."""

    def __init__(self, script):
        self.directory = Path(script).resolve().parent
        self.name = self.directory.relative_to(ROOT).as_posix()
        self.path = self.directory / "cases.json"
        self.cases = json.loads(self.path.read_text())
        self.write = "--write" in sys.argv[1:]
        filters = [
            argument for argument in sys.argv[1:] if not argument.startswith("-")
        ]
        self.filter = filters[0] if filters else ""
        self.passed = self.failed = self.skipped = self.written = 0

    def check(self, name, reference):
        """Runs case `name` through `reference`, then compares its outputs with the expected ones, or records them with `--write`.
        A case that raises fails, and the next one runs."""
        label = f"{self.name}/{name}"
        if self.filter not in label:
            return
        try:
            self.run(label, name, reference)
        except Exception as error:
            print(f"{label}: FAILED, {type(error).__name__}: {error}")
            traceback.print_exc()
            self.failed += 1
        finally:
            # Torch keeps the reference's memory cached, which the next model needs on a small GPU.
            gc.collect()
            torch.cuda.empty_cache()

    def run(self, label, name, reference):
        case = next(
            (case for case in self.cases["cases"] if case["name"] == name), None
        )
        if case is None:
            raise LookupError(f"cases.json has no case {name}")
        snapshot = find_snapshot(case["snapshot"])
        if snapshot is None:
            name = case["snapshot"]
            print(
                f"{label}: skipped, snapshot {name} is not downloaded (`just snapshot {name}`)"
            )
            self.skipped += 1
            return

        inputs, parameters = media(case["inputs"]), case.get("parameters") or {}
        run, load, load_gpu = profiled(lambda: reference(snapshot))
        # The first run warms the reference up, and its outputs are the ones compared or recorded.
        start = time.perf_counter()
        outputs = plain(run(inputs, parameters))
        first = time.perf_counter() - start
        # The timed runs are not profiled, so profiling slows none of them; one more run is, for what the GPU did in a run.
        times = []
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            run(inputs, parameters)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        _, _, run_gpu = profiled(lambda: run(inputs, parameters))
        timing = report(load, load_gpu, first, statistics.median(times), run_gpu)
        del run

        if self.write:
            case["expected"] = rounded(
                [recorded(case["task"], output) for output in outputs]
            )
            print(f"{label}: written\n    reference: {timing}")
            self.written += 1
            return
        results = [
            compare(case["task"], want, got)
            for want, got in zip(case["expected"], outputs)
        ]
        details = [detail for _, detail in results]
        if len(outputs) != len(case["expected"]):
            details.append(
                f"{len(outputs)} outputs for {len(case['expected'])} expected"
            )
        matched = len(outputs) == len(case["expected"]) and all(ok for ok, _ in results)
        print(
            f"{label}: {'ok' if matched else 'MISMATCH'}; {'; '.join(details)}\n    reference: {timing}"
        )
        if matched:
            self.passed += 1
        else:
            self.failed += 1

    def finish(self):
        """Writes the recorded outputs with `--write`, even when a case failed, reports the counts, and exits."""
        if self.write:
            self.path.write_text(
                json.dumps(self.cases, indent=2, ensure_ascii=False) + "\n"
            )
        print(f"{self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        if self.failed:
            sys.exit(1)
        if self.passed + self.written == 0:
            print("no case ran")
            sys.exit(NOTHING_RAN)
        sys.exit(0)


def recorded(task, output):
    """What `cases.json` keeps of an output: a mesh's summary, and other outputs whole."""
    if task in ("text-to-3d", "image-to-3d"):
        return mesh_summary(output)
    return output


# Comparisons, as `support/compare.rs` makes them.


def compare(task, want, got):
    if task in (
        "image-to-image",
        "unconditional-image-generation",
        "text-to-image",
        "image-text-to-image",
    ):
        return compare_images(want, got)
    if task in (
        "text-to-video",
        "image-to-video",
        "video-to-video",
        "image-text-to-video",
    ):
        return compare_frames(want, got)
    if task in ("text-to-3d", "image-to-3d"):
        return compare_meshes(want, got)
    if task == "depth-estimation":
        return compare_images(want["depth"], got["depth"])
    if task == "image-segmentation":
        return compare_masks(want, got)
    if task == "keypoint-detection":
        return compare_keypoints(want, got)
    if task == "keypoint-matching":
        return compare_matches(want, got)
    if task in ("audio-to-audio", "text-to-audio"):
        return compare_waveforms(want, got)
    if task == "mask-generation":
        regions = lambda output: [
            {"score": score, "mask": mask}
            for score, mask in zip(output["scores"], output["masks"])
        ]  # noqa: E731
        return compare_masks(regions(want), regions(got))
    if task in ("text-generation", "image-text-to-text", "video-text-to-text"):
        return compare_tokens(want, got)
    if task in ("feature-extraction", "image-feature-extraction"):
        return compare_features(want, got)
    if task == "automatic-speech-recognition":
        return compare_text(want, got)
    if task in GENERATED_TEXT:
        if not isinstance(want, dict):
            return want == got, json.dumps(got, ensure_ascii=False)
        # The keys the reference records, which Singe's output may extend.
        matched = all(key in got and got[key] == value for key, value in want.items())
        return matched, json.dumps(
            {key: got.get(key) for key in want}, ensure_ascii=False
        )
    if task in (
        "decision",
        "visual-document-retrieval",
        "time-series-forecasting",
        "reinforcement-learning",
        "tabular-classification",
        "tabular-regression",
    ):
        error = largest_difference(want, got)
        if error is None:
            return False, f"{task} differs: {got}, expected {want}"
        return error < TOLERANCE, f"largest difference {error:.1e}"
    return compare_scored(task, want, got)


# Tasks whose outputs are generated text, compared exactly, as greedy generation in F32 gives the reference's tokens.
GENERATED_TEXT = (
    "summarization",
    "translation",
    "image-to-text",
    "document-question-answering",
    "table-question-answering",
    "any-to-any",
    "audio-text-to-text",
    "visual-question-answering",
)


def largest_difference(want, got):
    """The largest difference of two outputs' numbers, relative above one, if everything else in them is the same, and `None` otherwise."""
    if isinstance(want, dict) and isinstance(got, dict):
        if want.keys() != got.keys():
            return None
        differences = [largest_difference(want[key], got[key]) for key in want]
    elif isinstance(want, list) and isinstance(got, list):
        if len(want) != len(got):
            return None
        differences = [largest_difference(a, b) for a, b in zip(want, got)]
    elif isinstance(want, bool) or isinstance(got, bool):
        return 0.0 if want == got else None
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
        # Relative to the reference's value above one, which records six significant digits.
        return abs(want - got) / max(abs(want), 1.0)
    else:
        return 0.0 if want == got else None
    return None if None in differences else max(differences, default=0.0)


def generation(model, tokens):
    """A generation's record: the tokens `generate` gave after the prompt, and the end tokens of the model's generation config, where a generation that leaves the end token out may stop early."""
    end = model.generation_config.eos_token_id
    return {
        "generated_token_ids": tokens,
        "end_token_ids": [end] if isinstance(end, int) else list(end or []),
    }


def compare_tokens(want, got):
    """The same tokens, or a non-empty start of the reference's that stops where the reference generated an end token, which `generate` keeps and Singe leaves out."""
    ends = want.get("end_token_ids", [])
    want, got = want["generated_token_ids"], got["generated_token_ids"]
    if not got:
        return False, f"no tokens, expected {len(want)}"
    same = 0
    while same < min(len(want), len(got)) and want[same] == got[same]:
        same += 1
    if same == len(got) < len(want) and want[same] in ends:
        return True, f"{same} of {len(want)} tokens, stopping at the end token"
    return same == len(want) == len(got), f"{same} of {len(want)} tokens"


def rows(value):
    array = numpy.asarray(value, dtype=numpy.float64)
    return array.reshape(-1, array.shape[-1])


def compare_features(want, got):
    want, got = rows(want), rows(got)
    if want.shape != got.shape:
        return False, "shapes differ"
    norms = numpy.maximum(
        numpy.linalg.norm(want, axis=1) * numpy.linalg.norm(got, axis=1), 1e-12
    )
    similarities = (want * got).sum(axis=1) / norms
    mean, lowest = float(similarities.mean()), float(similarities.min())
    return (
        mean > 0.999 and lowest > 0.99,
        f"mean cosine {mean:.6f}, lowest {lowest:.6f}",
    )


def decoded(url):
    """An image's pixels from its PNG data URL, as pipelines return images."""
    from PIL import Image

    data = base64.b64decode(url.split(",", 1)[1])
    return numpy.asarray(
        Image.open(io.BytesIO(data)).convert("RGB"), dtype=numpy.float64
    )


def data_url(pixels):
    """A PNG data URL of RGB bytes, `[H, W, 3]`, as Singe's pipelines write images."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# The least peak signal-to-noise ratio of an image against the reference's, in decibels.
LEAST_PSNR = 40.0


# The most of a mask's pixels that may differ from the reference's, where scores near a tie at a region's edge pick the other label.
MOST_DIFFERING = 2e-3


def compare_masks(want, got):
    """The same regions in order, with the same labels and scores, and masks differing in at most `MOST_DIFFERING` of their pixels."""
    strip = lambda segments: [
        {key: value for key, value in segment.items() if key != "mask"}
        for segment in segments
    ]  # noqa: E731
    if len(want) != len(got):
        return False, f"{len(got)} regions, expected {len(want)}"
    ok, detail = compare_scored("image-segmentation", strip(want), strip(got))
    if not ok:
        return ok, detail
    worst = 0.0
    for want_segment, got_segment in zip(want, got):
        want_mask, got_mask = (
            decoded(want_segment["mask"]),
            decoded(got_segment["mask"]),
        )
        if want_mask.shape != got_mask.shape:
            return False, f"mask size {got_mask.shape}, expected {want_mask.shape}"
        worst = max(worst, float((want_mask != got_mask).any(axis=2).mean()))
    return (
        worst <= MOST_DIFFERING,
        f"{len(want)} regions, masks differing in at most {worst:.2%} of pixels",
    )


def compare_keypoints(want, got):
    """The same keypoints within a hundredth of a pixel, scores within `TOLERANCE`, and the descriptors the reference recorded, its first keypoints', by cosine."""
    if len(want["keypoints"]) != len(got["keypoints"]):
        return (
            False,
            f"{len(got['keypoints'])} keypoints, expected {len(want['keypoints'])}",
        )
    points = float(
        numpy.abs(
            numpy.asarray(want["keypoints"]) - numpy.asarray(got["keypoints"])
        ).max(initial=0.0)
    )
    scores = float(
        numpy.abs(numpy.asarray(want["scores"]) - numpy.asarray(got["scores"])).max(
            initial=0.0
        )
    )
    count = len(want["descriptors"])
    ok, detail = compare_features(want["descriptors"], got["descriptors"][:count])
    return (
        ok and points < 1e-2 and scores < TOLERANCE,
        f"{len(want['keypoints'])} keypoints, largest offset {points:.1e}, largest score difference {scores:.1e}, {detail}",
    )


def compare_matches(want, got):
    """The same pairs of keypoints, in any order, since near-equal scores may sort either way, each with its score within `TOLERANCE`."""
    pair = lambda match: (
        tuple(match["keypoint_image_0"].values()),
        tuple(match["keypoint_image_1"].values()),
    )  # noqa: E731
    want_pairs = {pair(match): match["score"] for match in want}
    got_pairs = {pair(match): match["score"] for match in got}
    if want_pairs.keys() != got_pairs.keys():
        missing, extra = (
            len(want_pairs.keys() - got_pairs.keys()),
            len(got_pairs.keys() - want_pairs.keys()),
        )
        return (
            False,
            f"{len(got)} matches, expected {len(want)}: {missing} missing, {extra} extra",
        )
    error = max(
        (abs(want_pairs[key] - got_pairs[key]) for key in want_pairs), default=0.0
    )
    return (
        error < TOLERANCE,
        f"{len(want)} matches, largest score difference {error:.1e}",
    )


# The least signal-to-noise ratio of a waveform against the reference's, in decibels.
LEAST_SNR = 30.0


def compare_waveforms(want, got):
    """The same rate and length, and a signal-to-noise ratio of at least `LEAST_SNR`."""
    if want["sampling_rate"] != got["sampling_rate"] or len(want["audio"]) != len(
        got["audio"]
    ):
        return (
            False,
            f"{len(got['audio'])} samples at {got['sampling_rate']} Hz, expected {len(want['audio'])} at {want['sampling_rate']}",
        )
    want_samples, got_samples = (
        numpy.asarray(want["audio"]),
        numpy.asarray(got["audio"]),
    )
    noise = float(((want_samples - got_samples) ** 2).sum())
    signal = float((want_samples**2).sum())
    snr = float("inf") if noise == 0 else 10 * numpy.log10(signal / noise)
    return snr >= LEAST_SNR, f"SNR {snr:.1f} dB"


def image_psnr(want, got):
    """The peak signal-to-noise ratio of an image against the reference's, in decibels, or `None` when they cannot be compared, and what the comparison found."""
    want, got = decoded(want), decoded(got)
    if want.shape != got.shape:
        return None, f"size {got.shape[1::-1]}, expected {want.shape[1::-1]}"
    error = float(((want - got) ** 2).mean())
    psnr = float("inf") if error == 0 else float(10 * numpy.log10(255.0**2 / error))
    return psnr, f"PSNR {psnr:.1f} dB"


def compare_images(want, got):
    """The same size, and a peak signal-to-noise ratio of at least `LEAST_PSNR`."""
    psnr, detail = image_psnr(want, got)
    return psnr is not None and psnr >= LEAST_PSNR, detail


# Every how many vertices and colors a mesh's summary keeps.
MESH_STRIDE = 97


def mesh_summary(mesh):
    """A mesh's counts and every `MESH_STRIDE`th vertex and color, which `cases.json` keeps of a mesh too large to record whole."""
    return {
        "vertex_count": len(mesh["vertices"]),
        "face_count": len(mesh["faces"]),
        "vertices": mesh["vertices"][::MESH_STRIDE],
        "colors": mesh["colors"][::MESH_STRIDE],
    }


def compare_meshes(want, got):
    """Vertex and face counts within 0.1%, and each vertex the reference sampled at a vertex of the mesh within `TOLERANCE`, with that vertex's color within `TOLERANCE`.
    Marching cubes flips a cell where the field is near zero on either side of a rounding, which shifts the indices after it, so vertices are matched by place rather than by index."""
    vertices, colors, faces = got["vertices"], got["colors"], len(got["faces"])

    def close(count, expected):
        return abs(count - expected) <= expected * 1e-3

    if not close(len(vertices), want["vertex_count"]) or not close(
        faces, want["face_count"]
    ):
        return (
            False,
            f"{len(vertices)} vertices and {faces} faces, expected {want['vertex_count']} and {want['face_count']}",
        )

    # The mesh's vertices by cell of a grid as fine as the field's, to find each sampled vertex's nearest.
    def cell(point):
        return tuple(math.floor((value + 1.0) * 64.0) for value in point)

    cells = {}
    for index, vertex in enumerate(vertices):
        cells.setdefault(cell(vertex), []).append(index)
    farthest = color_error = 0.0
    for vertex, color in zip(want["vertices"], want["colors"]):
        x, y, z = cell(vertex)
        nearest = min(
            (
                (math.dist(vertices[index], vertex), index)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                for index in cells.get((x + dx, y + dy, z + dz), [])
            ),
            key=lambda found: found[0],
            default=None,
        )
        if nearest is None:
            return False, f"no vertex near the sampled one at {vertex}"
        distance, index = nearest
        farthest = max(farthest, distance)
        got_color = colors[index] if index < len(colors) else [math.nan] * 3
        color_error = max(
            [color_error, *(abs(g - w) for g, w in zip(got_color, color))]
        )
    return (
        farthest < TOLERANCE and color_error < TOLERANCE,
        f"{len(vertices)} vertices; sampled vertices within {farthest:.1e} of the mesh's, colors within {color_error:.1e}",
    )


def compare_frames(want, got):
    """A clip's frames, each compared as an image, with the worst frame's finding: the first that could not be compared, or else the one of the lowest PSNR."""
    if len(want) != len(got):
        return False, f"{len(got)} frames, expected {len(want)}"
    if not want:
        return False, "a clip of no frames"
    results = [
        image_psnr(want_frame, got_frame) for want_frame, got_frame in zip(want, got)
    ]
    index = min(
        range(len(results)),
        key=lambda index: -math.inf if results[index][0] is None else results[index][0],
    )
    matched = all(psnr is not None and psnr >= LEAST_PSNR for psnr, _ in results)
    return matched, f"{len(want)} frames; worst, frame {index}: {results[index][1]}"


def compare_text(want, got):
    want, got = want["text"].strip(), got["text"].strip()
    return want == got, repr(got)


def scores(value):
    if isinstance(value, list):
        return [score for item in value for score in scores(item)]
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            if key in ("score", "scores"):
                # Numbers only, as `support/compare.rs` reads them: semantic segmentation records no score.
                values = item if isinstance(item, list) else [item]
                found.extend(value for value in values if value is not None)
            else:
                found.extend(scores(item))
        return found
    return []


def without_scores(value):
    if isinstance(value, list):
        return [without_scores(item) for item in value]
    if isinstance(value, dict):
        return {
            key: without_scores(item)
            for key, item in value.items()
            if key not in ("score", "scores")
        }
    return value


def compare_scored(task, want, got):
    if without_scores(want) != without_scores(got) or len(scores(want)) != len(
        scores(got)
    ):
        return False, f"{task} differs: {got}, expected {want}"
    error = max((abs(w - g) for w, g in zip(scores(want), scores(got))), default=0.0)
    return error < TOLERANCE, f"largest score difference {error:.1e}"


def read_wav(path):
    """Reads a 16-bit PCM WAV file as samples in [-1, 1), its channels averaged, and its rate."""
    with wave.open(path) as clip:
        frames = clip.readframes(clip.getnframes())
        channels, rate = clip.getnchannels(), clip.getframerate()
    samples = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32) / 32768.0
    return samples.reshape(-1, channels).mean(axis=1), rate


if __name__ == "__main__":
    # `python support.py download NAME…` downloads snapshots, as `just snapshot` runs it, and `python support.py media` writes the media cases read, as `just verify` runs it.
    if sys.argv[1:2] == ["download"] and sys.argv[2:]:
        download(sys.argv[2:])
    elif sys.argv[1:] == ["media"]:
        write_media()
    else:
        sys.exit("usage: support.py download NAME… | support.py media")
