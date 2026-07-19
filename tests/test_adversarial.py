"""Offline tests for the adversarial judge-validation generator and its
confusion-table/gate math. No network calls -- those are exercised only by
the deliberate, manually-invoked `cognitrace validate-judges` run."""

from __future__ import annotations

from cognitrace.harness import adversarial
from cognitrace.harness.datasets import load_locomo
from cognitrace.harness.schema import QAItem, Session, Task, Turn

_SMALL_COUNTS = {
    "exact_match": 2, "paraphrase": 2, "hedged_correct": 2, "wrong_but_topical": 2,
    "date_quantity_near_miss": 2, "flat_wrong": 2, "abstain_then_guess": 2,
}


def _fixture_tasks() -> list[Task]:
    def qa(qid, q, a, cat):
        return QAItem(qid=qid, question=q, answer=a, category=cat)

    questions = (
        [qa(f"sh{i}", f"single-hop question {i}", f"single-hop answer {i}", "single-hop") for i in range(6)]
        + [qa(f"tm{i}", f"temporal question {i}", d, "temporal")
           for i, d in enumerate(["May 2023", "7 June 2023", "2022", "10 March 2024", "June 2023", "1 July 2023"])]
        + [qa(f"mh{i}", f"multi-hop question {i}", f"multi-hop answer {i}", "multi-hop") for i in range(6)]
        + [qa(f"od{i}", f"open-domain question {i}", f"open-domain answer {i}", "open-domain") for i in range(6)]
        + [qa(f"adv{i}", f"unanswerable question {i}", "", "adversarial") for i in range(6)]
    )
    session = Session(session_id="s1", date="1 May 2023", turns=[Turn(role="A", content="x", turn_id="s1:0")])
    return [Task(task_id="conv-1", dataset="locomo", sessions=[session], questions=questions)]


def test_generator_is_deterministic_given_same_seed():
    tasks = _fixture_tasks()
    a = adversarial.generate_validation_set(tasks, seed=42, counts=_SMALL_COUNTS)
    b = adversarial.generate_validation_set(tasks, seed=42, counts=_SMALL_COUNTS)
    assert [(c.cls, c.qid, c.response, c.expected_correct) for c in a] == \
           [(c.cls, c.qid, c.response, c.expected_correct) for c in b]


def test_generator_different_seeds_can_differ():
    tasks = _fixture_tasks()
    a = adversarial.generate_validation_set(tasks, seed=1, counts=_SMALL_COUNTS)
    b = adversarial.generate_validation_set(tasks, seed=2, counts=_SMALL_COUNTS)
    assert [c.response for c in a] != [c.response for c in b]


def test_generator_produces_requested_counts_per_class():
    tasks = _fixture_tasks()
    cases = adversarial.generate_validation_set(tasks, seed=0, counts=_SMALL_COUNTS)
    assert len(cases) == sum(_SMALL_COUNTS.values())
    for cls, n in _SMALL_COUNTS.items():
        assert sum(1 for c in cases if c.cls == cls) == n


def test_expected_correct_labels_match_class_semantics():
    tasks = _fixture_tasks()
    cases = adversarial.generate_validation_set(tasks, seed=0, counts=_SMALL_COUNTS)
    by_class = {c.cls: [] for c in cases}
    for c in cases:
        by_class[c.cls].append(c)
    for c in by_class["exact_match"] + by_class["paraphrase"] + by_class["hedged_correct"]:
        assert c.expected_correct is True
        assert c.is_abstention is False
    for c in (by_class["wrong_but_topical"] + by_class["date_quantity_near_miss"]
              + by_class["flat_wrong"] + by_class["abstain_then_guess"]):
        assert c.expected_correct is False
    for c in by_class["abstain_then_guess"]:
        assert c.is_abstention is True


def test_exact_match_response_is_gold_verbatim():
    tasks = _fixture_tasks()
    cases = adversarial.generate_validation_set(tasks, seed=0, counts=_SMALL_COUNTS)
    for c in cases:
        if c.cls == "exact_match":
            assert c.response == c.gold


def test_date_quantity_near_miss_shifts_a_real_year_or_number():
    tasks = _fixture_tasks()
    counts = dict(_SMALL_COUNTS, date_quantity_near_miss=6)
    cases = [c for c in adversarial.generate_validation_set(tasks, seed=0, counts=counts)
             if c.cls == "date_quantity_near_miss"]
    assert len(cases) == 6
    # None should just echo the gold value verbatim inside the wrapper -- a
    # near-miss that equals the original isn't a near-miss.
    for c in cases:
        assert c.gold not in c.response or c.gold == ""


def test_wrong_but_topical_never_reuses_the_items_own_answer():
    tasks = _fixture_tasks()
    counts = dict(_SMALL_COUNTS, wrong_but_topical=10)
    cases = [c for c in adversarial.generate_validation_set(tasks, seed=0, counts=counts)
             if c.cls == "wrong_but_topical"]
    for c in cases:
        assert c.gold not in c.response


def test_abstain_then_guess_sources_from_adversarial_items():
    tasks = _fixture_tasks()
    cases = [c for c in adversarial.generate_validation_set(tasks, seed=0, counts=_SMALL_COUNTS)
             if c.cls == "abstain_then_guess"]
    for c in cases:
        assert c.qid.startswith("adv")
        assert c.gold == ""  # LoCoMo's own unanswerable gold is empty


def test_generator_runs_against_real_locomo_data():
    # End-to-end sanity against the actual downloaded dataset (Sprint 1.2),
    # not just the small fixture -- confirms real category names/volumes
    # satisfy every pool the generator needs.
    tasks = load_locomo()
    cases = adversarial.generate_validation_set(tasks, seed=0)
    assert len(cases) == sum(adversarial._DEFAULT_COUNTS.values())
    for cls in adversarial.CLASSES:
        assert sum(1 for c in cases if c.cls == cls) == adversarial._DEFAULT_COUNTS[cls]


# --- confusion tables + gate math (synthetic judge output, no network) ----

def _fake_results(rows_by_judge: dict[str, list[dict]]) -> dict[str, list[dict]]:
    return rows_by_judge


def test_confusion_table_computes_false_accept_and_reject_rates():
    rows = [
        {"cls": "wrong_but_topical", "qid": "q1", "expected_correct": False, "actual_correct": True, "tier": "llm", "raw": "yes"},
        {"cls": "wrong_but_topical", "qid": "q2", "expected_correct": False, "actual_correct": False, "tier": "llm", "raw": "no"},
        {"cls": "paraphrase", "qid": "q3", "expected_correct": True, "actual_correct": False, "tier": "llm", "raw": "no"},
        {"cls": "paraphrase", "qid": "q4", "expected_correct": True, "actual_correct": True, "tier": "llm", "raw": "yes"},
    ]
    tables = adversarial.confusion_tables({"judge-x": rows})
    assert tables["judge-x"]["wrong_but_topical"]["false_accept_rate"] == 0.5
    assert tables["judge-x"]["paraphrase"]["false_reject_rate"] == 0.5
    assert tables["judge-x"]["exact_match"]["n"] == 0
    assert tables["judge-x"]["exact_match"]["false_accept_rate"] is None


def test_gate_check_fails_on_high_false_accept():
    rows = [
        {"cls": "wrong_but_topical", "qid": f"q{i}", "expected_correct": False,
         "actual_correct": i < 3, "tier": "llm", "raw": ""} for i in range(10)
    ] + [
        {"cls": "date_quantity_near_miss", "qid": f"d{i}", "expected_correct": False,
         "actual_correct": False, "tier": "llm", "raw": ""} for i in range(10)
    ] + [
        {"cls": "paraphrase", "qid": f"p{i}", "expected_correct": True,
         "actual_correct": True, "tier": "llm", "raw": ""} for i in range(10)
    ] + [
        {"cls": "hedged_correct", "qid": f"h{i}", "expected_correct": True,
         "actual_correct": True, "tier": "llm", "raw": ""} for i in range(10)
    ]
    tables = adversarial.confusion_tables({"bad-judge": rows})
    verdicts = adversarial.gate_check(tables)
    assert verdicts["bad-judge"]["false_accept_max"] == 0.3  # 3/10 on wrong_but_topical
    assert not verdicts["bad-judge"]["passes_false_accept_gate"]
    assert verdicts["bad-judge"]["passes_false_reject_gate"]
    assert not verdicts["bad-judge"]["passes"]


def test_gate_check_passes_a_clean_judge():
    rows = []
    for cls in adversarial.FALSE_ACCEPT_GATE_CLASSES:
        rows += [{"cls": cls, "qid": f"{cls}{i}", "expected_correct": False,
                  "actual_correct": False, "tier": "llm", "raw": ""} for i in range(10)]
    for cls in adversarial.FALSE_REJECT_GATE_CLASSES:
        rows += [{"cls": cls, "qid": f"{cls}{i}", "expected_correct": True,
                  "actual_correct": True, "tier": "llm", "raw": ""} for i in range(10)]
    tables = adversarial.confusion_tables({"good-judge": rows})
    verdicts = adversarial.gate_check(tables)
    assert verdicts["good-judge"]["passes"]
