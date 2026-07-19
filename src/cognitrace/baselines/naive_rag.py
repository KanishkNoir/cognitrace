"""Naive-RAG floor: lexical top-k turns via stdlib SQLite FTS5 (BM25).

No embeddings, no chunk tuning — deliberately the dumbest defensible
retriever, so gains above it are attributable to the system, not the plumbing.
"""

from __future__ import annotations

import re
import sqlite3

from cognitrace.harness.schema import QAItem, Retrieval, Session

_TOKEN = re.compile(r"[A-Za-z0-9]+")

_EMPTY = "(no relevant memory found)"


def _fts_query(text: str) -> str | None:
    # FTS5 MATCH has operator syntax; quote each token and OR them.
    tokens = _TOKEN.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else None


class NaiveRag:
    def __init__(self, sessions: list[Session], top_k: int = 20):
        self.top_k = top_k
        self._db = sqlite3.connect(":memory:")
        # Porter stemming so "name" matches "named" — without it the lexical
        # floor whiffs on trivial morphology, which understates every system
        # compared against it. Still deliberately dumb: no synonyms, no
        # embeddings, no tuning.
        self._db.execute(
            "CREATE VIRTUAL TABLE turns USING fts5(session_id UNINDEXED, "
            "turn_id UNINDEXED, date UNINDEXED, role UNINDEXED, content, "
            "tokenize='porter unicode61')"
        )
        rows = [
            (s.session_id, t.turn_id or "", s.date or "", t.role, t.content)
            for s in sessions
            for t in s.turns
            if t.content.strip()
        ]
        self._db.executemany("INSERT INTO turns VALUES (?, ?, ?, ?, ?)", rows)
        self._db.commit()

    def retrieve(self, question: QAItem) -> Retrieval:
        query = _fts_query(question.question)
        if query is None:
            return Retrieval(context=_EMPTY)
        cur = self._db.execute(
            "SELECT session_id, turn_id, date, role, content FROM turns "
            "WHERE turns MATCH ? ORDER BY rank LIMIT ?",
            (query, self.top_k),
        )
        rows = cur.fetchall()
        if not rows:
            return Retrieval(context=_EMPTY)
        lines = []
        session_ids: list[str] = []
        turn_ids: list[str] = []
        for sid, tid, date, role, content in rows:
            lines.append(f"[session {sid}{' @ ' + date if date else ''}] {role}: {content}")
            if sid and sid not in session_ids:
                session_ids.append(sid)
            if tid:
                turn_ids.append(tid)
        return Retrieval(context="\n".join(lines), session_ids=session_ids, turn_ids=turn_ids)

    def context_for(self, question: QAItem) -> str:
        return self.retrieve(question).context


def build(sessions: list[Session], top_k: int = 20) -> NaiveRag:
    return NaiveRag(sessions, top_k=top_k)
