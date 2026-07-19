"""Reference ceiling: hand the reader the entire dated transcript."""

from __future__ import annotations

from cognitrace.harness.schema import QAItem, Retrieval, Session


class FullContext:
    def __init__(self, sessions: list[Session]):
        parts = []
        self._session_ids = [s.session_id for s in sessions]
        self._turn_ids = [t.turn_id for s in sessions for t in s.turns if t.turn_id]
        for s in sessions:
            header = f"=== Session {s.session_id}" + (f" ({s.date})" if s.date else "") + " ==="
            body = "\n".join(f"{t.role}: {t.content}" for t in s.turns)
            parts.append(f"{header}\n{body}")
        self._context = "\n\n".join(parts)

    def retrieve(self, question: QAItem) -> Retrieval:
        # Everything is in context: evidence-recall is 1.0 by construction,
        # which is the point of the ceiling — it isolates reader error.
        return Retrieval(
            context=self._context,
            session_ids=list(self._session_ids),
            turn_ids=list(self._turn_ids),
        )

    def context_for(self, question: QAItem) -> str:
        return self.retrieve(question).context


def build(sessions: list[Session]) -> FullContext:
    return FullContext(sessions)
