"""Real snapshots in the Hugging Face cache, downloaded when missing."""

from huggingface_hub import snapshot_download

# The real models Singe's ignored tests read, by the name `SINGE_SNAPSHOT_<NAME>` gives them.
MODELS = {
    "QWEN3": "Qwen/Qwen3-0.6B",
    "MINILM": "sentence-transformers/all-MiniLM-L6-v2",
    "TAESD": "madebyollin/taesd",
    "DINOV2": "facebook/dinov2-small",
    "WHISPER": "openai/whisper-tiny",
}


def snapshot(name, patterns=None):
    return snapshot_download(MODELS[name], allow_patterns=patterns)
