"""Reader + judge: the two LLM roles the benchmark protocol itself requires.

The system under test only decides WHAT context reaches the reader. Reader
and judge models are pinned per eval run and recorded in the results file,
so every published number states its full protocol.

Grading is tiered (RESEARCH_BASIS §3): a deterministic matcher runs first
and may only return a confident YES — anything else falls through to the
LLM judge. Tier-0 rules must stay generic, dataset-agnostic transformations;
a rule that would not make sense on a dataset we have never seen does not
belong here (the equivalence-rule firewall).

Judge calls dispatch by model prefix so the judge can live in a different
model family than the reader ("claude-*" -> Anthropic, otherwise OpenAI) —
the cross-family rule.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_READER_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "gpt-4o"

_MAX_ATTEMPTS = 3
_BACKOFF_S = 2.0

_READER_SYSTEM = (
    "You answer questions about a user based on their conversation history. "
    "Use ONLY the provided memory/context. Answer concisely with the specific "
    "fact requested. If the context does not contain the answer, say exactly: "
    "I don't know."
)

_JUDGE_TEMPLATE = (
    "You are grading a QA system.\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model response: {response}\n\n"
    "Does the model response contain the gold answer (semantically — exact "
    "wording not required, but the specific fact must match, including any "
    "dates or quantities)? Reply with exactly one word: yes or no."
)

# Abstention grading must not be a free pass: a refusal is only correct on
# an unanswerable item, and a refusal PLUS a guess is a guess. An
# abstain-always system must look terrible in our tables.
_JUDGE_ABSTENTION_TEMPLATE = (
    "You are grading a QA system on an unanswerable question — the correct "
    "behavior is to clearly decline, stating the information is not "
    "available, WITHOUT also guessing an answer.\n"
    "Question: {question}\n"
    "Model response: {response}\n\n"
    "Did the model decline to answer without hallucinating or appending a "
    "guessed answer? Reply with exactly one word: yes or no."
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_fingerprints() -> dict[str, str]:
    """SHA-256 of every prompt template, for the run manifest."""
    return {
        "reader_system": _sha(_READER_SYSTEM),
        "judge": _sha(_JUDGE_TEMPLATE),
        "judge_abstention": _sha(_JUDGE_ABSTENTION_TEMPLATE),
    }


@dataclass
class ReadResult:
    text: str
    ok: bool
    error: str | None
    usage_in: int
    usage_out: int
    model: str


@dataclass
class JudgeResult:
    correct: bool
    tier: str  # "deterministic" | "llm"
    raw: str  # verdict text as returned (tier-0: the rule name)


def _openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (harness needs it for reader/judge).")
    return OpenAI(api_key=key)


def read_answer(question: str, context: str, model: str = DEFAULT_READER_MODEL,
                question_date: str | None = None, seed: int | None = None) -> ReadResult:
    """Ask the reader. On failure after bounded retries, return ok=False —
    the error is recorded as a status, never as a response (failure banners
    must not enter the answer column)."""
    user = f"Memory/context:\n{context}\n\n"
    if question_date:
        user += f"Current date: {question_date}\n"
    user += f"Question: {question}"
    last_err: str | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = _openai_client().chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": _READER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0,
                seed=seed,
            )
            usage = getattr(resp, "usage", None)
            return ReadResult(
                text=(resp.choices[0].message.content or "").strip(),
                ok=True,
                error=None,
                usage_in=getattr(usage, "prompt_tokens", 0) or 0,
                usage_out=getattr(usage, "completion_tokens", 0) or 0,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001 - typed as status, not answer
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_S * (2 ** attempt))
    return ReadResult(text="", ok=False, error=last_err, usage_in=0, usage_out=0, model=model)


# --- Tier 0: deterministic matcher (generic transformations only) ----------

_NORM_DROP = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(the|a|an)\b")


def _normalize(text: str) -> str:
    t = _NORM_DROP.sub(" ", text.lower())
    t = _ARTICLES.sub(" ", t)
    return _WS.sub(" ", t).strip()


def deterministic_match(gold: str, response: str) -> str | None:
    """Return a rule name on a confident YES; None means undecided (fall
    through to the LLM judge). Tier 0 never returns a NO — a conservative
    matcher cannot create false negatives, only save judge calls."""
    if not gold or not response:
        return None
    g, r = _normalize(gold), _normalize(response)
    if not g or not r:
        return None
    if g == r:
        return "exact"
    # Whole-phrase containment (the judge template's own semantics are
    # containment; require token boundaries so "10" can't match inside "104").
    if len(g) >= 3 and re.search(rf"(?:^| ){re.escape(g)}(?: |$)", r):
        return "containment"
    return None


# --- Judge -----------------------------------------------------------------

def _judge_completion(prompt: str, model: str) -> str:
    """Dispatch by model family so the judge can be cross-family."""
    if model.startswith("claude"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment issue
            raise SystemExit(
                "judge model is Anthropic but the 'anthropic' package is not "
                "installed (uv add anthropic)"
            ) from exc
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set (needed for a claude-* judge).")
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model, max_tokens=8, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    resp = _openai_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def judge_answer(question: str, gold: str, response: str,
                 is_abstention: bool = False, model: str = DEFAULT_JUDGE_MODEL) -> JudgeResult:
    if not is_abstention:
        rule = deterministic_match(gold, response)
        if rule is not None:
            return JudgeResult(correct=True, tier="deterministic", raw=rule)
    if is_abstention:
        prompt = _JUDGE_ABSTENTION_TEMPLATE.format(question=question, response=response)
    else:
        prompt = _JUDGE_TEMPLATE.format(question=question, gold=gold, response=response)
    verdict = _judge_completion(prompt, model).strip().lower()
    return JudgeResult(correct=verdict.startswith("yes"), tier="llm", raw=verdict)
