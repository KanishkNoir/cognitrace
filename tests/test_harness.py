"""Loader + baseline smoke tests on miniature in-format fixtures (no network)."""

import json

from cognitrace.baselines import full_context, naive_rag
from cognitrace.harness import protocol
from cognitrace.harness.datasets import load_locomo, load_longmemeval
from cognitrace.harness.reader import deterministic_match, prompt_fingerprints

LONGMEMEVAL_FIXTURE = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What instrument does the user play?",
        "answer": "cello",
        "question_date": "2023/05/20 (Sat) 02:21",
        "haystack_session_ids": ["s_a", "s_b"],
        "haystack_dates": ["2023/04/10 (Mon) 11:00", "2023/04/12 (Wed) 09:30"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I started cello lessons last week."},
                {"role": "assistant", "content": "That's wonderful!"},
            ],
            [
                {"role": "user", "content": "What's a good pasta recipe?"},
                {"role": "assistant", "content": "Try cacio e pepe."},
            ],
        ],
        "answer_session_ids": ["s_a"],
    },
    {
        "question_id": "q2_abs",
        "question_type": "knowledge-update",
        "question": "What is the user's dog's name?",
        "answer": "N/A",
        "haystack_session_ids": ["s_c"],
        "haystack_dates": ["2023/04/15 (Sat) 10:00"],
        "haystack_sessions": [[{"role": "user", "content": "I love hiking."}]],
        "answer_session_ids": [],
    },
]

LOCOMO_FIXTURE = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Maya",
            "speaker_b": "Rob",
            "session_1_date_time": "1:00 pm on 5 May, 2023",
            "session_1": [
                {"speaker": "Maya", "dia_id": "D1:1", "text": "I adopted a beagle named Biscuit!"},
                {"speaker": "Rob", "dia_id": "D1:2", "text": "Congrats!"},
            ],
            "session_2_date_time": "4:10 pm on 20 June, 2023",
            "session_2": [
                {"speaker": "Maya", "dia_id": "D2:1", "text": "Biscuit chewed my headphones."},
            ],
        },
        "qa": [
            {"question": "What is Maya's dog's name?", "answer": "Biscuit", "category": 4,
             "evidence": ["D1:1"]},
            {"question": "When did Maya adopt her dog?", "answer": "May 2023", "category": 2,
             "evidence": ["D1:1"]},
        ],
    }
]


def test_load_longmemeval(tmp_path):
    p = tmp_path / "longmemeval_s.json"
    p.write_text(json.dumps(LONGMEMEVAL_FIXTURE), encoding="utf-8")
    tasks = load_longmemeval(p)
    assert len(tasks) == 2
    t1 = tasks[0]
    assert t1.dataset == "longmemeval_s"
    assert [s.session_id for s in t1.sessions] == ["s_a", "s_b"]
    assert t1.sessions[0].date == "2023/04/10 (Mon) 11:00"
    assert t1.questions[0].evidence_session_ids == ["s_a"]
    assert not t1.questions[0].is_abstention
    assert tasks[1].questions[0].is_abstention  # qid ends with _abs


def test_load_locomo(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    tasks = load_locomo(p)
    assert len(tasks) == 1
    task = tasks[0]
    assert len(task.sessions) == 2
    assert task.sessions[0].date == "1:00 pm on 5 May, 2023"
    assert task.sessions[0].turns[0].role == "Maya"
    assert len(task.questions) == 2
    assert task.questions[0].category == "single-hop"
    assert task.questions[1].category == "temporal"


def test_full_context_contains_everything(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    task = load_locomo(p)[0]
    ctx = full_context.build(task.sessions).context_for(task.questions[0])
    assert "Biscuit" in ctx and "headphones" in ctx
    assert "1:00 pm on 5 May, 2023" in ctx  # dates must reach the reader


def test_naive_rag_surfaces_relevant_turn(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    task = load_locomo(p)[0]
    rag = naive_rag.build(task.sessions, top_k=1)
    ctx = rag.context_for(task.questions[0])  # "What is Maya's dog's name?"
    assert "Biscuit" in ctx


def test_naive_rag_empty_query_is_safe(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    task = load_locomo(p)[0]
    rag = naive_rag.build(task.sessions)
    q = task.questions[0]
    q.question = "???"
    assert rag.context_for(q) == "(no relevant memory found)"


# --- evidence ids (the blocking defect for evidence-recall) -----------------

def test_locomo_evidence_turn_and_session_ids(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    task = load_locomo(p)[0]
    q = task.questions[0]
    assert q.evidence_turn_ids == ["D1:1"]
    # dia_id "D1:1" derives session-level evidence: conv-1_session_1
    assert q.evidence_session_ids == ["conv-1_session_1"]
    # turn ids come straight from dia_id
    assert task.sessions[0].turns[0].turn_id == "D1:1"


def test_longmemeval_turn_ids_synthesized(tmp_path):
    p = tmp_path / "longmemeval_s.json"
    p.write_text(json.dumps(LONGMEMEVAL_FIXTURE), encoding="utf-8")
    task = load_longmemeval(p)[0]
    assert task.sessions[0].turns[0].turn_id == "s_a:0"
    assert task.sessions[0].turns[1].turn_id == "s_a:1"


# --- retrieval interface (feeds evidence-recall) ----------------------------

def test_full_context_retrieve_lists_everything(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    task = load_locomo(p)[0]
    r = full_context.build(task.sessions).retrieve(task.questions[0])
    assert r.session_ids == ["conv-1_session_1", "conv-1_session_2"]
    assert "D1:1" in r.turn_ids and "D2:1" in r.turn_ids
    assert "Biscuit" in r.context


def test_naive_rag_retrieve_returns_ranked_ids(tmp_path):
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    task = load_locomo(p)[0]
    r = naive_rag.build(task.sessions, top_k=1).retrieve(task.questions[0])
    assert r.turn_ids == ["D1:1"]  # the Biscuit-adoption turn ranks first
    assert r.session_ids == ["conv-1_session_1"]


def test_evidence_recall_metric():
    assert protocol.evidence_recall(["D1:1", "D2:1"], ["D1:1"], k=5) == 1.0
    assert protocol.evidence_recall(["D2:1"], ["D1:1"], k=5) == 0.0
    assert protocol.evidence_recall(["D2:1", "D1:1"], ["D1:1"], k=1) == 0.0
    # No gold ids => not scored (None), never counted as zero.
    assert protocol.evidence_recall(["D1:1"], [], k=5) is None


# --- tiered grading ---------------------------------------------------------

def test_deterministic_match_is_conservative():
    assert deterministic_match("cello", "The user plays the cello.") == "containment"
    assert deterministic_match("Biscuit", "biscuit") == "exact"
    # Token boundary: "10" must not match inside "104".
    assert deterministic_match("10", "the answer is 104") is None
    # Undecided (paraphrase) falls to the LLM tier — never a deterministic NO.
    assert deterministic_match("May 2023", "sometime last spring") is None
    assert deterministic_match("", "anything") is None


# --- protocol governance ----------------------------------------------------

def test_protocol_file_matches_live_prompts():
    # Drift tripwire: editing a prompt template without re-pinning the
    # protocol file must fail CI, not silently relabel runs.
    proto = protocol.load_protocol()
    assert proto is not None, "protocol_v1.json missing"
    assert proto["prompts"] == prompt_fingerprints()


def test_estimate_firewall_on_partial_and_offprotocol_runs():
    proto = protocol.load_protocol()
    pinned = {
        "limit": 0,
        "reader_model": proto["reader_model"],
        "prompts": proto["prompts"],
    }
    assert protocol.classify_run(pinned) == proto["name"]
    assert protocol.classify_run({**pinned, "limit": 50}) == "estimate"
    assert protocol.classify_run({**pinned, "reader_model": "other"}) == "estimate"
    # Score half: off-protocol judge or answer key demotes a pinned run.
    m = {"protocol": proto["name"]}
    assert protocol.classify_score(m, proto["judge_model"], "original") == proto["name"]
    assert protocol.classify_score(m, "some-other-judge", "original") == "estimate"
    assert protocol.classify_score({"protocol": "estimate"}, proto["judge_model"], "original") == "estimate"


def test_verdict_cache_roundtrip(tmp_path):
    cache = protocol.VerdictCache(tmp_path / "verdicts.jsonl")
    k = protocol.VerdictCache.key("gpt-4o", "sha", "q?", "gold", "resp")
    assert cache.get(k) is None
    cache.put(k, True, "llm", "yes")
    assert cache.get(k)["correct"] is True
    # Reload from disk: regrade is bit-identical at zero model cost.
    cache2 = protocol.VerdictCache(tmp_path / "verdicts.jsonl")
    assert cache2.get(k)["correct"] is True
