"""cognitrace CLI: download datasets, run an eval, score a results file.

Run and score are separate so a results file can be re-judged (or judged
with a different model, or against a corrected answer key) without
re-spending reader calls. Judge verdicts are cached to disk, so a regrade
is zero-model-cost.

Every results file opens with a Full Disclosure manifest; runs or scores
that deviate from protocol_v1.json are machine-labeled "estimate" and must
never enter a published table (the SPEC firewall).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from . import protocol
from .datasets import DATA_DIR, file_sha256, load_locomo, load_longmemeval
from .schema import Task

RESULTS_DIR = Path(os.environ.get("COGNITRACE_RESULTS_DIR") or (DATA_DIR.parent / "results"))

_LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

_ABSTAIN_MARKERS = ("i don't know", "i do not know", "not available", "no information",
                    "cannot determine", "can't determine", "not mentioned")

_MAX_IDS_PER_ROW = 100  # cap stored retrieved-id lists; full-context would bloat rows


def _cmd_download(args: argparse.Namespace) -> int:
    locomo_target = DATA_DIR / "locomo" / "locomo10.json"
    if locomo_target.exists():
        print(f"[skip] {locomo_target} already present")
    else:
        locomo_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[get ] LoCoMo <- {_LOCOMO_URL}")
        try:
            with urllib.request.urlopen(_LOCOMO_URL) as resp:
                locomo_target.write_bytes(resp.read())
            print(f"[ok  ] {locomo_target}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[fail] LoCoMo download: {exc}")
            print("       Fetch data/locomo10.json manually from github.com/snap-research/locomo")

    lme_dir = DATA_DIR / "longmemeval"
    if any(lme_dir.glob("longmemeval_*.json")):
        print(f"[skip] LongMemEval files present in {lme_dir}")
    else:
        lme_dir.mkdir(parents=True, exist_ok=True)
        print("[todo] LongMemEval is distributed via the authors' Drive/HF links:")
        print("       https://github.com/xiaowu0162/LongMemEval  (Setup section)")
        print(f"       Place longmemeval_s.json / _m / _oracle in {lme_dir}")
    return 0


def _dataset_path(dataset: str) -> Path:
    if dataset == "locomo":
        return DATA_DIR / "locomo" / "locomo10.json"
    variant = dataset.split("_", 1)[1] if "_" in dataset else "s"
    return DATA_DIR / "longmemeval" / f"longmemeval_{variant}.json"


def _load_tasks(dataset: str) -> list[Task]:
    if dataset == "locomo":
        return load_locomo()
    if dataset.startswith("longmemeval"):
        variant = dataset.split("_", 1)[1] if "_" in dataset else "s"
        return load_longmemeval(variant=variant)
    raise SystemExit(f"unknown dataset: {dataset}")


def _build_system(name: str, sessions, top_k: int):
    if name == "full-context":
        from cognitrace.baselines import full_context

        return full_context.build(sessions)
    if name == "naive-rag":
        from cognitrace.baselines import naive_rag

        return naive_rag.build(sessions, top_k=top_k)
    if name == "grep-agent":
        from cognitrace.baselines import grep_agent

        return grep_agent.build(sessions)
    raise SystemExit(f"unknown system: {name}")


def _run_once(args: argparse.Namespace, seed: int) -> Path:
    from . import reader

    tasks = _load_tasks(args.dataset)
    ds_path = _dataset_path(args.dataset)
    out = RESULTS_DIR / f"{args.dataset}.{args.system}.seed{seed}.{int(time.time())}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = protocol.build_manifest(
        dataset=args.dataset,
        dataset_path=str(ds_path),
        dataset_sha256=file_sha256(ds_path) if ds_path.exists() else "missing",
        system=args.system,
        system_params={"top_k": args.top_k},
        reader_model=args.reader_model,
        prompts=reader.prompt_fingerprints(),
        seed=seed,
        limit=args.limit,
        results_path=str(out),
    )

    n_done = n_err = 0
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"manifest": manifest}) + "\n")
        for task in tasks:
            system = _build_system(args.system, task.sessions, args.top_k)
            for qa in task.questions:
                if args.limit and n_done >= args.limit:
                    break
                t0 = time.perf_counter()
                if args.system == "grep-agent":
                    ans = system.answer(qa, model=args.reader_model, seed=seed)
                    row_extra = {
                        "context_chars": 0,
                        "llm_calls": ans.llm_calls,
                        "tool_calls": ans.tool_calls,
                        "retrieved_session_ids": ans.session_ids[:_MAX_IDS_PER_ROW],
                        "retrieved_turn_ids": ans.turn_ids[:_MAX_IDS_PER_ROW],
                        "usage_in": ans.usage_in, "usage_out": ans.usage_out,
                    }
                    text, ok, err = ans.text, ans.ok, ans.error
                else:
                    retrieval = system.retrieve(qa)
                    res = reader.read_answer(
                        qa.question, retrieval.context, model=args.reader_model,
                        question_date=qa.question_date, seed=seed,
                    )
                    row_extra = {
                        "context_chars": len(retrieval.context),
                        "llm_calls": 1,
                        "tool_calls": 0,
                        "retrieved_session_ids": retrieval.session_ids[:_MAX_IDS_PER_ROW],
                        "retrieved_turn_ids": retrieval.turn_ids[:_MAX_IDS_PER_ROW],
                        "usage_in": res.usage_in, "usage_out": res.usage_out,
                    }
                    text, ok, err = res.text, res.ok, res.error
                fh.write(json.dumps({
                    "qid": qa.qid, "category": qa.category,
                    "question": qa.question, "gold": qa.answer,
                    "is_abstention": qa.is_abstention,
                    "evidence_session_ids": qa.evidence_session_ids,
                    "evidence_turn_ids": qa.evidence_turn_ids,
                    "status": "ok" if ok else "error",
                    "error": err,
                    "response": text if ok else None,
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                    **row_extra,
                }) + "\n")
                n_done += 1
                n_err += 0 if ok else 1
                if n_done % 25 == 0:
                    print(f"  ... {n_done} questions")
            if args.limit and n_done >= args.limit:
                break
    label = manifest["protocol"]
    print(f"[ok] {n_done} responses ({n_err} errors) -> {out}  [protocol: {label}]")
    print(f"Next: cognitrace score {out}")
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    seeds = list(range(args.seeds)) if args.seeds else [args.seed]
    for seed in seeds:
        if len(seeds) > 1:
            print(f"[seed {seed}]")
        _run_once(args, seed)
    return 0


def _looks_like_abstention(response: str) -> bool:
    r = response.lower()
    return any(m in r for m in _ABSTAIN_MARKERS)


def _pct(hits: list[bool]) -> str:
    return f"{sum(hits)}/{len(hits)} = {100 * sum(hits) / max(len(hits), 1):.1f}%"


def _cmd_score(args: argparse.Namespace) -> int:
    from . import reader

    lines = Path(args.results).read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    manifest = head.get("manifest")
    if manifest is None:
        raise SystemExit("results file has no manifest (pre-manifest files must be re-run)")
    rows = [json.loads(ln) for ln in lines[1:] if ln.strip()]

    # Answer-key overlay (dual-key scoring): {qid: corrected_gold}.
    key_name = "original"
    if args.answer_key:
        overlay = json.loads(Path(args.answer_key).read_text(encoding="utf-8"))
        key_name = Path(args.answer_key).stem
        replaced = 0
        for row in rows:
            if row["qid"] in overlay:
                row["gold"] = str(overlay[row["qid"]])
                replaced += 1
        print(f"[key] '{key_name}': {replaced} gold answers overlaid")

    # Judge drift sentinel (tripwire for silent judge-model swaps).
    def _sentinel_judge(question, gold, response, is_abstention, model):
        return reader.judge_answer(question, gold, response,
                                   is_abstention=is_abstention, model=model).correct

    report = protocol.run_sentinel(_sentinel_judge, args.judge_model)
    if report is not None:
        state = "baseline recorded" if not report.baseline_recorded else (
            f"flip-rate {report.flip_rate:.1%} over {report.total}")
        print(f"[sentinel] {state}")
        if not report.ok and not args.ignore_sentinel:
            raise SystemExit(
                f"judge sentinel drift {report.flip_rate:.1%} > 2% — the judge has "
                "changed under us; halting. Re-validate, then re-baseline or pass "
                "--ignore-sentinel to score anyway (result will be an estimate)."
            )

    cache = protocol.VerdictCache(Path(args.results).with_suffix(".verdicts.jsonl"))
    prompts = manifest.get("prompts", {})
    judge_prompt_sha = f"{prompts.get('judge', '')}|{prompts.get('judge_abstention', '')}"

    scored = [r for r in rows if r.get("status", "ok") == "ok" and r.get("response") is not None]
    excluded = len(rows) - len(scored)
    per_cat: dict[str, list[bool]] = {}
    tier_counts = {"deterministic": 0, "llm": 0, "cached": 0}
    for i, row in enumerate(scored):
        k = protocol.VerdictCache.key(args.judge_model, judge_prompt_sha,
                                      row["question"], row["gold"], row["response"])
        hit = cache.get(k)
        if hit is not None:
            row["correct"], row["grade_tier"] = hit["correct"], hit["tier"]
            tier_counts["cached"] += 1
        else:
            res = reader.judge_answer(
                row["question"], row["gold"], row["response"],
                is_abstention=row.get("is_abstention", False),
                model=args.judge_model,
            )
            row["correct"], row["grade_tier"] = res.correct, res.tier
            cache.put(k, res.correct, res.tier, res.raw)
            tier_counts[res.tier] += 1
        per_cat.setdefault(row["category"], []).append(row["correct"])
        if (i + 1) % 25 == 0:
            print(f"  ... judged {i + 1}/{len(scored)}")

    label = protocol.classify_score(manifest, args.judge_model, key_name)
    if report is not None and not report.ok:
        label = "estimate"

    answerable = [r for r in scored if not r.get("is_abstention")]
    abstention = [r for r in scored if r.get("is_abstention")]
    total = [r["correct"] for r in scored]
    total_wo = [r["correct"] for r in answerable]

    print(f"\n{manifest.get('dataset')} / {manifest.get('system')} "
          f"(reader={manifest.get('reader_model')}, judge={args.judge_model}, "
          f"key={key_name}, seed={manifest.get('seed')})  [protocol: {label}]")
    if excluded:
        print(f"[warn] {excluded} rows excluded (reader errors) — not in any denominator")
    print(f"overall (with abstention):    {_pct(total)}")
    print(f"overall (without abstention): {_pct(total_wo)}")
    for cat in sorted(per_cat):
        print(f"  {cat:<24} {_pct(per_cat[cat])}")

    # Abstention calibration pair: an abstain-always system must look bad here.
    metrics: dict = {}
    if abstention:
        false_answer = [not r["correct"] for r in abstention]
        metrics["false_answer_rate_on_unanswerable"] = sum(false_answer) / len(false_answer)
        print(f"  abstention: false-answer rate  {sum(false_answer)}/{len(false_answer)}"
              f" = {100 * sum(false_answer) / len(false_answer):.1f}%")
    if answerable:
        false_abstain = [_looks_like_abstention(r["response"]) for r in answerable]
        metrics["false_abstain_rate_on_answerable"] = sum(false_abstain) / len(false_abstain)
        print(f"  abstention: false-abstain rate {sum(false_abstain)}/{len(false_abstain)}"
              f" = {100 * sum(false_abstain) / len(false_abstain):.1f}%")

    # Reader-independent evidence-recall (immune to judge leniency).
    for level, ret_key, gold_key in (
        ("session", "retrieved_session_ids", "evidence_session_ids"),
        ("turn", "retrieved_turn_ids", "evidence_turn_ids"),
    ):
        for k in (5, 10):
            vals = [
                protocol.evidence_recall(r.get(ret_key) or [], r.get(gold_key) or [], k)
                for r in scored
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                metrics[f"{level}_recall@{k}"] = sum(vals) / len(vals)
                print(f"  evidence {level}-recall@{k:<3} {100 * sum(vals) / len(vals):.1f}% "
                      f"(over {len(vals)} questions with gold ids)")

    print(f"  grading tiers: {tier_counts}")

    graded = Path(args.results).with_suffix(".graded.json")
    graded.write_text(json.dumps({
        "manifest": manifest,
        "judge_model": args.judge_model,
        "answer_key": key_name,
        "protocol": label,
        "excluded_rows": excluded,
        "metrics": metrics,
        "tier_counts": tier_counts,
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"[ok] per-question grades -> {graded}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from cognitrace.store.doctor import run_doctor
    from cognitrace.store.rebuild import record_replay_baseline
    from cognitrace.store.schema import open_store

    conn = open_store(Path(args.store_path))
    if args.record_baseline:
        sha = record_replay_baseline(conn)
        print(f"[ok] replay baseline recorded: {sha}")
    violations = run_doctor(conn)
    if not violations:
        print(f"[ok] {args.store_path}: all invariants clean")
        return 0
    for v in violations:
        print(f"[{v.code}] {v.detail}")
    print(f"[fail] {len(violations)} invariant violation(s)")
    return 1


def _cmd_validate_judges(args: argparse.Namespace) -> int:
    from cognitrace.harness import adversarial

    tasks = _load_tasks("locomo")
    cases = adversarial.generate_validation_set(tasks, seed=args.seed)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    print(f"[validate-judges] {len(cases)} cases x {len(judges)} judge(s) "
          f"= {len(cases) * len(judges)} judge calls")

    results = adversarial.run_validation(cases, judges)
    tables = adversarial.confusion_tables(results)
    verdicts = adversarial.gate_check(tables)

    out = RESULTS_DIR / f"judge_validation.seed{args.seed}.{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed, "n_cases": len(cases), "judges": judges,
        "confusion_tables": tables, "gate_verdicts": verdicts, "raw_results": results,
    }, indent=1), encoding="utf-8")

    for jm in judges:
        print(f"\n=== {jm} ===")
        for cls in adversarial.CLASSES:
            row = tables[jm][cls]
            fa = f"{row['false_accept_rate']:.1%}" if row["false_accept_rate"] is not None else "n/a"
            fr = f"{row['false_reject_rate']:.1%}" if row["false_reject_rate"] is not None else "n/a"
            print(f"  {cls:<24} n={row['n']:<3} false-accept={fa:<7} false-reject={fr}")
        v = verdicts[jm]
        gate = "PASS" if v["passes"] else "FAIL"
        fa_s = f"{v['false_accept_max']:.1%}" if v["false_accept_max"] is not None else "n/a"
        fr_s = f"{v['false_reject_max']:.1%}" if v["false_reject_max"] is not None else "n/a"
        print(f"  gate: false-accept-max={fa_s} false-reject-max={fr_s} -> {gate}")
    print(f"\n[ok] full results -> {out}")
    return 0


def _cmd_failures(args: argparse.Namespace) -> int:
    from cognitrace.store.ingest import list_dead_letters
    from cognitrace.store.schema import open_store

    conn = open_store(Path(args.store_path))
    dead = list_dead_letters(conn)
    if not dead:
        print(f"[ok] {args.store_path}: no dead letters")
        return 0
    for row in dead:
        print(f"[job {row['job_id']}] turn={row['turn_id']} error={row['error']}")
    print(f"[warn] {len(dead)} dead-lettered turn(s) — capture miss, visible in FDR")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cognitrace")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch benchmark datasets into data/")

    run = sub.add_parser("run", help="run a system over a dataset")
    run.add_argument("dataset", choices=["locomo", "longmemeval_s", "longmemeval_m", "longmemeval_oracle"])
    run.add_argument("system", choices=["full-context", "naive-rag", "grep-agent"])
    run.add_argument("--limit", type=int, default=0, help="max questions (0 = all; nonzero => estimate)")
    run.add_argument("--top-k", type=int, default=20)
    run.add_argument("--reader-model", default="gpt-4o-mini")
    run.add_argument("--seed", type=int, default=0, help="single-run seed (recorded in manifest)")
    run.add_argument("--seeds", type=int, default=0,
                     help="run N seeded passes (0..N-1); pinned numbers need 5")

    score = sub.add_parser("score", help="judge a results file (tiered, cached, regradeable)")
    score.add_argument("results")
    score.add_argument("--judge-model", default="gpt-4o")
    score.add_argument("--answer-key", default=None,
                       help="JSON {qid: corrected_gold} overlay (e.g. Penfield-corrected key)")
    score.add_argument("--ignore-sentinel", action="store_true",
                       help="score despite sentinel drift (forces estimate label)")

    doctor = sub.add_parser("doctor", help="run store invariants I1-I7 against a conversation store file")
    doctor.add_argument("store_path")
    doctor.add_argument("--record-baseline", action="store_true",
                        help="record a fresh replay baseline before checking I7 (first run only)")

    failures = sub.add_parser("failures", help="list dead-lettered extraction jobs in a store file")
    failures.add_argument("store_path")

    validate_judges = sub.add_parser(
        "validate-judges", help="adversarial judge validation over real LoCoMo gold (Sprint 1.5, S14)"
    )
    validate_judges.add_argument("--judges", default="gpt-4o",
                                 help="comma-separated candidate judge models")
    validate_judges.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    return {
        "download": _cmd_download, "run": _cmd_run, "score": _cmd_score,
        "doctor": _cmd_doctor, "failures": _cmd_failures,
        "validate-judges": _cmd_validate_judges,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
