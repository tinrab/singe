"""RT-DETR and RT-DETRv2, detecting objects with `transformers`' object-detection pipeline, as `mod.rs` detects them in Singe."""

from transformers import pipeline

from support import DEVICE, Verification


def object_detection(snapshot, dtype):
    detect = pipeline("object-detection", model=snapshot, dtype=dtype, device=DEVICE)
    return lambda paths, parameters: [detect(path, **parameters) for path in paths]


verification = Verification(__file__)
verification.check("detection", object_detection)
verification.check("detection_v2", object_detection)
verification.finish()
