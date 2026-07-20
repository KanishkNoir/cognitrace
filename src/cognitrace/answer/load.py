"""Resolve + load a registered QA model's ONNX session and tokenizer,
cached per name (Sprint 4.6). onnxruntime/tokenizers are imported INSIDE
`load_qa_model`, never at module top -- so `_resolve_paths` (and every
pure caller) works on a bare install without the extractive-qa extra.
"""

from __future__ import annotations

from pathlib import Path

from cognitrace.answer.models import QAModelSpec, get_spec, model_dir

_CACHE: dict[str, tuple] = {}


def _resolve_paths(name: str) -> tuple[QAModelSpec, Path, Path]:
    """Locate `name`'s artifact files, raising an actionable FileNotFoundError
    if either is absent (no onnxruntime needed to fail this way, so it is
    testable on a bare install). O(1) plus two stat() calls."""
    spec = get_spec(name)
    d = model_dir(name)
    onnx_path = d / spec.onnx_filename
    tok_path = d / "tokenizer.json"
    if not onnx_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            f"QA model {name!r} not found at {d}. "
            f"Run: cognitrace download --model {name}"
        )
    return spec, onnx_path, tok_path


def load_qa_model(name: str) -> tuple:
    """Return `(onnxruntime session, tokenizers.Tokenizer, QAModelSpec)`,
    cached per name (loading a 30-70MB ONNX graph is not free; the harness
    answers many questions against one model). Lazy-imports the extra so an
    install without it can still import this module. O(1) amortized after
    first load; first load is O(model file size)."""
    if name in _CACHE:
        return _CACHE[name]
    spec, onnx_path, tok_path = _resolve_paths(name)
    import onnxruntime  # lazy: keeps extractive-qa an optional extra
    from tokenizers import Tokenizer

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(str(tok_path))
    result = (session, tokenizer, spec)
    _CACHE[name] = result
    return result
