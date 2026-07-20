"""One-time ONNX export + parity check for a registered QA model
(Sprint 4.6, D2). No SQuAD2 QA model ships pre-built ONNX, so the artifact
is produced once from the PyTorch checkpoint and then verified against it.

`parity_check` (the comparison logic) is PURE pure-python -- no numpy -- so
its test runs on a bare install. `export_to_onnx` lazy-imports the heavy
toolchain (optimum, torch) and is a dev/build-time step, never a harness
runtime dependency; it raises ImportError (with the pip line) when the
toolchain is absent so the CLI can print actionable guidance.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from cognitrace.answer.models import get_spec, model_dir


def _flatten(x) -> Iterable[float]:
    if isinstance(x, (list, tuple)):
        for item in x:
            yield from _flatten(item)
    else:
        yield float(x)


def parity_check(reference, onnx_out, *, atol: float = 1e-3) -> tuple[bool, float]:
    """Max absolute elementwise difference between the reference (PyTorch)
    logits and the ONNX logits; passes iff within `atol` AND same element
    count. Pure-python (no numpy) so it runs on a bare install. This is the
    S6-style ONNX-parity discipline applied to the answerer -- the direct
    guard against the silent-export-failure class the project's corpus
    documents (the DeBERTa 'always predicts the same label' bug avoided in
    the model choice). O(K), K = total logits."""
    ref = list(_flatten(reference))
    got = list(_flatten(onnx_out))
    if len(ref) != len(got):
        return False, float("inf")
    max_diff = max((abs(a - b) for a, b in zip(ref, got)), default=0.0)
    return (max_diff <= atol), max_diff


def export_to_onnx(name: str, *, atol: float = 1e-3) -> str:
    """Export `name`'s PyTorch checkpoint to ONNX under model_dir(name),
    verify PyTorch-vs-ONNX logit parity on a sample, and return the ONNX
    file's sha256 (the artifact is treated as pinned by that hash, not by a
    promise the export is reproducible). Refuses to bless an artifact that
    fails parity. Dev/build-time only -- raises ImportError (with the pip
    line) if the toolchain is absent. Not unit-tested against the real model
    (needs the toolchain); the parity LOGIC it relies on is tested via
    `parity_check`. O(model size)."""
    try:
        import numpy as np
        import torch  # noqa: F401
        from optimum.onnxruntime import ORTModelForQuestionAnswering
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:  # noqa: BLE001 - re-raise with the actionable line
        raise ImportError(
            "ONNX export needs the build toolchain: pip install optimum torch transformers"
        ) from exc

    spec = get_spec(name)
    out_dir = model_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id)
    ort_model = ORTModelForQuestionAnswering.from_pretrained(spec.repo_id, export=True)
    ort_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # tokenizers-lib expects tokenizer.json; AutoTokenizer.save_pretrained
    # writes it for fast tokenizers (MiniLM is uncased BERT WordPiece -> fast).

    # Parity: same inputs through the torch reference and the exported ONNX.
    ref_model = AutoModelForQuestionAnswering.from_pretrained(spec.repo_id)
    ref_model.eval()
    enc = tokenizer("What is this?", "This is a parity probe sentence.",
                    return_tensors="pt", truncation="only_second", max_length=spec.max_seq_len)
    with torch.no_grad():
        ref_out = ref_model(**enc)
    onnx_out = ort_model(**{k: v for k, v in enc.items()})
    ok_start, d_start = parity_check(
        ref_out.start_logits.tolist(), np.asarray(onnx_out.start_logits).tolist(), atol=atol)
    ok_end, d_end = parity_check(
        ref_out.end_logits.tolist(), np.asarray(onnx_out.end_logits).tolist(), atol=atol)
    if not (ok_start and ok_end):
        raise RuntimeError(
            f"ONNX parity FAILED for {name}: start_diff={d_start}, end_diff={d_end} "
            f"(atol={atol}). Refusing to bless a non-parity artifact."
        )

    onnx_path = out_dir / spec.onnx_filename
    return hashlib.sha256(onnx_path.read_bytes()).hexdigest()
