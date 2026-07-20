"""Tests for the zero-LLM extractive answerer (Sprint 4.6). Pure scoring
tests run on a bare install; the one real-inference test skips when the
ONNX model artifact (or the extractive-qa extra) is absent."""

from __future__ import annotations

from cognitrace.answer.qa_head import ChunkScore, AnswerResult, select_answer


def test_select_answer_empty_evidence_abstains():
    r = select_answer([], model="minilm-squad2")
    assert r.abstained is True
    assert r.answer == ""
    assert r.source_turn_id is None


def test_select_answer_single_positive_delta_returns_span():
    r = select_answer(
        [ChunkScore(turn_id="t1", span_text="Paris", span_score=5.0, null_score=1.0)],
        model="minilm-squad2",
    )
    assert r.abstained is False
    assert r.answer == "Paris"
    assert r.source_turn_id == "t1"
    assert r.score == 4.0  # 5.0 - 1.0


def test_select_answer_uses_null_adjusted_delta_not_raw_span_score():
    # Chunk A has the higher RAW span score (8 > 5) but a worse null-adjusted
    # delta (8-7=1 < 5-1=4). The whole point of the SQuAD2 null head: B wins.
    chunks = [
        ChunkScore(turn_id="A", span_text="wrong", span_score=8.0, null_score=7.0),
        ChunkScore(turn_id="B", span_text="right", span_score=5.0, null_score=1.0),
    ]
    r = select_answer(chunks, model="minilm-squad2")
    assert r.answer == "right"
    assert r.source_turn_id == "B"


def test_select_answer_all_deltas_below_threshold_abstains():
    chunks = [
        ChunkScore(turn_id="A", span_text="x", span_score=2.0, null_score=3.0),  # delta -1
        ChunkScore(turn_id="B", span_text="y", span_score=1.0, null_score=1.0),  # delta 0
    ]
    r = select_answer(chunks, model="minilm-squad2")  # default threshold 0.0
    assert r.abstained is True
    assert r.answer == ""


def test_select_answer_tie_is_broken_deterministically_by_chunk_order():
    chunks = [
        ChunkScore(turn_id="first", span_text="a", span_score=3.0, null_score=1.0),  # delta 2
        ChunkScore(turn_id="second", span_text="b", span_score=4.0, null_score=2.0),  # delta 2
    ]
    r = select_answer(chunks, model="minilm-squad2")
    assert r.source_turn_id == "first"  # earlier chunk wins an exact tie


def test_select_answer_respects_custom_abstain_threshold():
    chunks = [ChunkScore(turn_id="t1", span_text="maybe", span_score=3.0, null_score=1.0)]  # delta 2
    r = select_answer(chunks, model="minilm-squad2", abstain_threshold=2.0)  # <= 2 abstains
    assert r.abstained is True


def test_select_answer_carries_model_name():
    r = select_answer([], model="some-other-model")
    assert r.model == "some-other-model"


# --- model registry + loader --------------------------------------------

import pytest

from cognitrace.answer import models as qa_models
from cognitrace.answer.load import _resolve_paths
from cognitrace.answer.models import QA_MODELS, QAModelSpec, get_spec, model_dir


def test_registry_has_the_minilm_squad2_spec():
    spec = get_spec("minilm-squad2")
    assert isinstance(spec, QAModelSpec)
    assert spec.repo_id == "deepset/minilm-uncased-squad2"
    assert spec.max_seq_len == 512


def test_get_spec_unknown_model_raises_keyerror_listing_known():
    with pytest.raises(KeyError) as exc:
        get_spec("no-such-model")
    assert "minilm-squad2" in str(exc.value)


def test_model_dir_is_under_the_models_root(tmp_path, monkeypatch):
    monkeypatch.setattr(qa_models, "MODELS_DIR", tmp_path)
    assert model_dir("minilm-squad2") == tmp_path / "minilm-squad2"


def test_resolve_paths_missing_artifact_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(qa_models, "MODELS_DIR", tmp_path)  # empty dir: nothing downloaded
    with pytest.raises(FileNotFoundError) as exc:
        _resolve_paths("minilm-squad2")
    msg = str(exc.value)
    assert "cognitrace download --model minilm-squad2" in msg


# --- ONNX shell + orchestrator (real inference; skips without the model) --

from cognitrace.answer.models import get_spec, model_dir


def _qa_runnable(name: str = "minilm-squad2") -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    spec = get_spec(name)
    d = model_dir(name)
    return (d / spec.onnx_filename).exists() and (d / "tokenizer.json").exists()


def test_answer_question_empty_evidence_abstains_without_loading_a_model():
    # No evidence => no reason to load the model at all; must not raise even
    # when the artifact is absent.
    from cognitrace.answer.qa_head import answer_question

    r = answer_question("anything?", [], model="minilm-squad2")
    assert r.abstained is True


@pytest.mark.skipif(not _qa_runnable(), reason="QA model/extra absent (cognitrace download --model minilm-squad2)")
def test_answer_question_extracts_a_span_from_real_evidence():
    from cognitrace.answer.qa_head import answer_question

    r = answer_question(
        "What is the capital of France?",
        [("t1", "Paris is the capital of France."),
         ("t2", "The Eiffel Tower is a landmark.")],
        model="minilm-squad2",
    )
    assert r.abstained is False
    assert "Paris" in r.answer
    assert r.source_turn_id == "t1"


# --- export parity check (pure) + download CLI ---------------------------

from cognitrace.answer.export import parity_check


def test_parity_check_identical_logits_pass():
    ok, diff = parity_check([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]])
    assert ok is True
    assert diff == 0.0


def test_parity_check_small_diff_within_atol_passes():
    ok, diff = parity_check([1.0, 2.0], [1.0005, 2.0005], atol=1e-3)
    assert ok is True


def test_parity_check_large_diff_fails():
    ok, diff = parity_check([1.0, 2.0], [1.0, 2.5], atol=1e-3)
    assert ok is False
    assert abs(diff - 0.5) < 1e-9


def test_parity_check_shape_mismatch_fails_with_inf():
    ok, diff = parity_check([1.0, 2.0, 3.0], [1.0, 2.0])
    assert ok is False
    assert diff == float("inf")


def test_download_model_absent_toolchain_prints_actionable_todo(tmp_path, monkeypatch, capsys):
    # No model present and (in a bare test env) no optimum/torch -> the CLI
    # must print how to get it and exit 0, never crash (mirrors the existing
    # LongMemEval manual-fetch message).
    from cognitrace.answer import models as qa_models
    from cognitrace.harness.cli import main

    monkeypatch.setattr(qa_models, "MODELS_DIR", tmp_path)
    rc = main(["download", "--model", "minilm-squad2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "minilm-squad2" in out
