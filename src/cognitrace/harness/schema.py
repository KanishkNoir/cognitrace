"""Normalized benchmark schema.

Both LongMemEval and LoCoMo reduce to the same shape: a set of dated
sessions of (role, content) turns, and questions asked against them.
LongMemEval ships one haystack per question; LoCoMo shares one
conversation across ~200 questions — both become a list of Task.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" / "assistant" for LongMemEval; speaker name for LoCoMo
    content: str
    # Stable id when the dataset provides one (LoCoMo dia_id, e.g. "D1:3");
    # synthesized as "{session_id}:{index}" otherwise. Used for turn-level
    # evidence-recall diagnostics, never by the system under test.
    turn_id: str | None = None


@dataclass
class Session:
    session_id: str
    date: str | None  # ISO-ish string as provided by the dataset
    turns: list[Turn] = field(default_factory=list)


@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    category: str  # dataset-native category/ability label
    question_date: str | None = None
    # Session ids containing the evidence, when the dataset provides them
    # (used for retrieval-recall diagnostics, never by the system under test).
    evidence_session_ids: list[str] = field(default_factory=list)
    # Turn ids containing the evidence (LoCoMo `evidence` dia_ids). Same
    # firewall: diagnostics only — these must never reach a store or a
    # retrieval call.
    evidence_turn_ids: list[str] = field(default_factory=list)
    # LongMemEval abstention items (qid endswith "_abs"): correct behavior is
    # refusing to answer.
    is_abstention: bool = False


@dataclass
class Retrieval:
    """What a system hands the reader, plus what it surfaced to get there.

    `session_ids`/`turn_ids` are ranked (best first) and deduplicated; they
    feed evidence-recall@k. Systems that use everything (full-context) list
    everything.
    """

    context: str
    session_ids: list[str] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)


@dataclass
class Task:
    task_id: str
    dataset: str  # "longmemeval_s" | "longmemeval_m" | "locomo"
    sessions: list[Session]
    questions: list[QAItem]
