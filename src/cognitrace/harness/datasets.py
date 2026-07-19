"""Loaders that normalize LongMemEval and LoCoMo into Task objects.

Raw files live under data/ (gitignored):
  data/longmemeval/longmemeval_s.json   (and _m / _oracle)
  data/locomo/locomo10.json
See `cognitrace download` for how to fetch them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .schema import QAItem, Session, Task, Turn

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

_LOCOMO_SESSION_KEY = re.compile(r"^session_(\d+)$")
_LOCOMO_DIA_ID = re.compile(r"^D(\d+):\d+$")


def file_sha256(path: str | Path) -> str:
    """Dataset fingerprint for the run manifest (no number without its SHA)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_longmemeval(path: str | Path | None = None, variant: str = "s") -> list[Task]:
    """One Task per question: each item carries its own haystack of sessions."""
    path = Path(path) if path else DATA_DIR / "longmemeval" / f"longmemeval_{variant}.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for item in items:
        qid = str(item["question_id"])
        session_ids = [str(s) for s in item.get("haystack_session_ids", [])]
        dates = item.get("haystack_dates", [])
        sessions = []
        for i, raw_session in enumerate(item["haystack_sessions"]):
            sid = session_ids[i] if i < len(session_ids) else f"{qid}_s{i}"
            sessions.append(
                Session(
                    session_id=sid,
                    date=dates[i] if i < len(dates) else None,
                    turns=[
                        Turn(role=t["role"], content=t["content"], turn_id=f"{sid}:{j}")
                        for j, t in enumerate(raw_session)
                    ],
                )
            )
        qa = QAItem(
            qid=qid,
            question=item["question"],
            answer=str(item["answer"]),
            category=item.get("question_type", "unknown"),
            question_date=item.get("question_date"),
            evidence_session_ids=[str(s) for s in item.get("answer_session_ids", [])],
            is_abstention=qid.endswith("_abs"),
        )
        tasks.append(
            Task(task_id=qid, dataset=f"longmemeval_{variant}", sessions=sessions, questions=[qa])
        )
    return tasks


# LoCoMo category codes -> readable labels (category 5 is adversarial/
# unanswerable; most published comparisons drop it — we keep it, labeled).
_LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def _evidence_turn_ids(raw) -> list[str]:
    """Normalize LoCoMo's `evidence` field (list of dia_ids, occasionally a
    bare string or nested list) into a flat list of turn-id strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, list):
            out.extend(s.strip() for s in item if isinstance(s, str) and s.strip())
    return out


def load_locomo(path: str | Path | None = None) -> list[Task]:
    """One Task per conversation sample; all its QA items attached."""
    path = Path(path) if path else DATA_DIR / "locomo" / "locomo10.json"
    samples = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", len(tasks)))
        conv = sample["conversation"]
        sessions = []
        for key, value in conv.items():
            m = _LOCOMO_SESSION_KEY.match(key)
            if not m or not isinstance(value, list):
                continue
            n = int(m.group(1))
            sid = f"{sample_id}_session_{n}"
            turns = [
                Turn(
                    role=t.get("speaker", "unknown"),
                    content=t.get("text", ""),
                    turn_id=t.get("dia_id") or f"{sid}:{j}",
                )
                for j, t in enumerate(value)
            ]
            sessions.append(Session(session_id=sid, date=conv.get(f"session_{n}_date_time"), turns=turns))
        sessions.sort(key=lambda s: int(s.session_id.rsplit("_", 1)[-1]))
        questions = []
        for i, qa in enumerate(sample.get("qa", [])):
            cat = qa.get("category")
            turn_ids = _evidence_turn_ids(qa.get("evidence"))
            # Derive session-level evidence from dia_ids ("D3:7" -> session 3)
            # so LoCoMo gets session-recall diagnostics alongside turn-recall.
            session_ids: list[str] = []
            for tid in turn_ids:
                dm = _LOCOMO_DIA_ID.match(tid)
                if dm:
                    sid = f"{sample_id}_session_{int(dm.group(1))}"
                    if sid not in session_ids:
                        session_ids.append(sid)
            questions.append(
                QAItem(
                    qid=f"{sample_id}_q{i}",
                    question=qa["question"],
                    answer=str(qa.get("answer", "")),
                    category=_LOCOMO_CATEGORIES.get(cat, str(cat)),
                    evidence_session_ids=session_ids,
                    evidence_turn_ids=turn_ids,
                    is_abstention=cat == 5,
                )
            )
        tasks.append(Task(task_id=sample_id, dataset="locomo", sessions=sessions, questions=questions))
    return tasks
