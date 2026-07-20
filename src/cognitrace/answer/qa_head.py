"""The zero-LLM extractive answerer (Sprint 4.6, design_scaffold.md 4.6):
turn admitted evidence into an answer span with a small self-hosted ONNX
QA encoder -- no LLM in the answer path.

The load-bearing seam: `select_answer` is PURE (no model, no I/O) and holds
every scoring decision, so it is fully testable with synthetic numbers.
`run_qa`/`answer_question` (added in Task 4) are the impure ONNX shell and
lazy-import onnxruntime/tokenizers INSIDE their bodies, never at module
top -- keeping the pure path importable on a bare install.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkScore:
    """One evidence chunk's QA result: the best span it yielded, that span's
    raw score, and the chunk's own no-answer (null) score. The null score is
    what makes span_score comparable across chunks (see select_answer)."""
    turn_id: str
    span_text: str
    span_score: float
    null_score: float


@dataclass
class AnswerResult:
    answer: str  # empty string == abstained
    abstained: bool
    source_turn_id: str | None
    score: float  # the winning null-adjusted delta (or best delta on abstain)
    model: str


def select_answer(
    chunk_scores: list[ChunkScore], *, model: str, abstain_threshold: float = 0.0
) -> AnswerResult:
    """Pick the answer span across independently-scored evidence chunks by
    NULL-ADJUSTED DELTA (`span_score - null_score`), not raw span score.

    Raw per-chunk span logits are softmax-normalized only within their own
    chunk, so they are NOT comparable across chunks (Wang & Ng, Multi-passage
    BERT, EMNLP-IJCNLP 2019). Subtracting each chunk's own null (no-answer)
    score re-centers every chunk against its own baseline, making the
    cross-chunk max meaningful -- and directly reuses SQuAD2's trained
    no-answer head. If the best delta is <= `abstain_threshold` (default 0:
    "no chunk beats its own no-answer baseline"), abstain rather than emit a
    low-confidence guess. Ties are broken by chunk order (earlier wins), so
    the result never depends on set/dict iteration order. O(N), N = chunks.
    """
    if not chunk_scores:
        return AnswerResult(answer="", abstained=True, source_turn_id=None, score=0.0, model=model)
    best = chunk_scores[0]
    best_delta = best.span_score - best.null_score
    for cs in chunk_scores[1:]:
        delta = cs.span_score - cs.null_score
        if delta > best_delta:  # strict: earlier chunk wins an exact tie
            best, best_delta = cs, delta
    if best_delta <= abstain_threshold:
        return AnswerResult(answer="", abstained=True, source_turn_id=None, score=best_delta, model=model)
    return AnswerResult(
        answer=best.span_text, abstained=False, source_turn_id=best.turn_id,
        score=best_delta, model=model,
    )


_MAX_ANSWER_TOKENS = 30  # longest span we'll consider (SQuAD convention)


def run_qa(session, tokenizer, spec, question: str, chunk_text: str) -> tuple[str, float, float]:
    """Run one QA forward pass over a (question, chunk) pair and return
    `(span_text, span_score, null_score)`.

    Encodes as a sequence PAIR with context-only truncation (never truncate
    the question -- `only_second`), feeds token_type_ids when the graph
    expects them (BERT-family models do; the DeBERTa export bug this project
    avoided was literally a token_type_ids feed error). The null score is the
    [CLS] start+end logit (SQuAD2's trained no-answer signal). The best span
    is the highest start[i]+end[j] over context tokens with 1 <= (j-i) span
    length <= _MAX_ANSWER_TOKENS, sliced from the ORIGINAL chunk string via
    the tokenizer's offset_mapping -- never detokenized from wordpieces
    (lossy). Imports numpy lazily (onnxruntime pulls it in). O(L * A) for the
    span search, L = sequence length, A = _MAX_ANSWER_TOKENS.
    """
    import numpy as np

    tokenizer.enable_truncation(max_length=spec.max_seq_len, strategy="only_second")
    enc = tokenizer.encode(question, chunk_text)

    feed = {
        "input_ids": np.array([enc.ids], dtype=np.int64),
        "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
    }
    input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

    start_logits, end_logits = session.run(None, feed)
    start, end = start_logits[0], end_logits[0]
    null_score = float(start[0] + end[0])  # [CLS] at index 0

    seq_ids = enc.sequence_ids  # 0 = question, 1 = context, None = special
    offsets = enc.offsets
    n = len(start)
    best_score: float | None = None
    best_i = best_j = 0
    for i in range(1, n):
        if seq_ids[i] != 1:
            continue
        for j in range(i, min(i + _MAX_ANSWER_TOKENS, n)):
            if seq_ids[j] != 1:
                break
            s = float(start[i] + end[j])
            if best_score is None or s > best_score:
                best_score, best_i, best_j = s, i, j

    if best_score is None:  # no context tokens survived truncation
        return "", float("-inf"), null_score
    start_char = offsets[best_i][0]
    end_char = offsets[best_j][1]
    return chunk_text[start_char:end_char], best_score, null_score


def answer_question(
    question: str, evidence: list[tuple[str, str]], *,
    model: str = "minilm-squad2", abstain_threshold: float = 0.0,
) -> AnswerResult:
    """Orchestrate the zero-LLM answer path: one `run_qa` per evidence chunk
    (over its RAW value -- never a formatted display string), then pure
    `select_answer` over the collected ChunkScores. Empty evidence abstains
    WITHOUT loading a model (so it's safe on a bare/artifact-less install).
    `evidence` is `RetrievalResult.admitted_evidence`: (turn_id, raw_value)
    pairs. O(E * L * A), E = chunks, plus one model load (cached). No LLM,
    no network -- the encoder-only zero-LLM claim on the answer side."""
    if not evidence:
        return select_answer([], model=model, abstain_threshold=abstain_threshold)
    from cognitrace.answer.load import load_qa_model

    session, tokenizer, spec = load_qa_model(model)
    scores = []
    for turn_id, value in evidence:
        st, ss, ns = run_qa(session, tokenizer, spec, question, value)
        scores.append(ChunkScore(turn_id=turn_id, span_text=st, span_score=ss, null_score=ns))
    return select_answer(scores, model=model, abstain_threshold=abstain_threshold)
