"""QA model registry (Sprint 4.6). Model identity lives here, not hardcoded
in the inference path -- mirrors reader.py dispatching reader models by
name. One model registered this sprint; a second is a registry entry + a
download, not a code change.

Files live under ~/cognitrace-data/models/<name>/ (off any sync-watched
path, S20 -- reusing datasets.DATA_DIR's convention), alongside the
datasets, not in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cognitrace.harness.datasets import DATA_DIR

MODELS_DIR = DATA_DIR.parent / "models"


@dataclass(frozen=True)
class QAModelSpec:
    name: str
    repo_id: str  # Hugging Face repo the artifact is exported from
    onnx_filename: str
    max_seq_len: int


QA_MODELS: dict[str, QAModelSpec] = {
    "minilm-squad2": QAModelSpec(
        name="minilm-squad2",
        repo_id="deepset/minilm-uncased-squad2",
        onnx_filename="model.onnx",
        max_seq_len=512,
    ),
}


def get_spec(name: str) -> QAModelSpec:
    """O(1). Raises KeyError (listing known names) for an unregistered model
    rather than returning None -- an unknown model is a caller bug, not a
    silent miss."""
    if name not in QA_MODELS:
        raise KeyError(f"unknown QA model {name!r}; known: {sorted(QA_MODELS)}")
    return QA_MODELS[name]


def model_dir(name: str) -> Path:
    """Directory holding `name`'s ONNX + tokenizer files. Reads MODELS_DIR
    at call time so tests can monkeypatch it. O(1)."""
    return MODELS_DIR / name
