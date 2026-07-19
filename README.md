# CogniTrace

**Encoder-driven conversational memory, benchmarked in the open.**

CogniTrace takes the research line behind
[CogniKernel](https://github.com/KanishkNoir/cognikernel) — memory
construction as *classification, not generation*: typed events, latest-wins
supersession, hybrid lexical-primary retrieval, small local encoder models
instead of LLM extraction calls — and points it at the public long-term-memory
benchmarks: **LongMemEval** and **LoCoMo**, where systems like Mem0, Zep, and
Supermemory report their numbers.

## The claim under test

Every leading memory system on these leaderboards spends **LLM calls per
conversation turn** to extract and update memories — that is their per-user
cost, latency, and privacy bill. CogniTrace constructs memory with **encoders
only** (ONNX, local CPU, milliseconds, $0 per turn). The benchmark protocol
still requires an LLM *reader* to phrase the final answer and an LLM *judge*
to grade it — nobody escapes that; it's the protocol — but the memory system
itself never generates.

The target is the **cost/accuracy Pareto frontier**: encoder-only
construction within striking distance of LLM-extraction systems. The focus
category is **temporal reasoning** — historically the weakest column for the
incumbents — hence the name.

## Status: Phase 0 — harness before system

No memory system exists here yet, deliberately. Phase 0 exits on a
three-part contract, not scalar-matching: (1) **anchor reproduction** —
full-context LoCoMo lands inside a pre-registered 72.90 ± 4 band under the
anchor config (the peer-reviewed Mem0-paper number, arXiv 2504.19413); a
miss is published as a root-caused adverse finding, never silently tuned
away; (2) **ordering reproduction** — ceiling > floor in every category,
temporal weakest; (3) **our own multi-seed ceiling/floor under the hardened
cross-family protocol** becomes the permanent reference row. Baselines:
full-context (ceiling), naive-RAG (floor), and a filesystem-grep agent (the
null memory system — it scores ~74% on LoCoMo; beating naive-RAG but not
grep would mean nothing). Every number ships with a Full Disclosure
manifest (reader, judge, prompts, seeds, dataset SHAs) in a rerunnable
results file; any off-protocol number is machine-labeled an estimate.

```sh
uv sync
uv run cognitrace download          # LoCoMo auto; LongMemEval prints its manual step
export OPENAI_API_KEY=...           # reader + judge

uv run cognitrace run locomo full-context --limit 50   # ceiling, small slice
uv run cognitrace run locomo naive-rag   --limit 50    # floor
uv run cognitrace score results/<file>.jsonl           # per-category accuracy
```

## Roadmap

1. **Phase 0 — harness** (this): normalized loaders (LongMemEval S/M,
   LoCoMo with gold evidence turn-ids), dual answer keys
   (Penfield-corrected headline + original for comparability),
   adversarially validated cross-family judge with a drift sentinel,
   full-context + naive-RAG + filesystem-grep baselines, per-category
   scoring including abstention (with/without, plus the false-answer /
   false-abstain calibration pair) and reader-independent evidence-recall.
2. **Phase 1 — the spine**: vendor CogniKernel's event-sourced store and
   supersession; typed events (`FACT`, `PREFERENCE`, `EVENT_DATED`,
   `RELATIONSHIP`) with an `asserts_change` flag — supersession is an
   *edge* (subject key + provenance + time), not an event type;
   **tri-temporal schema from day one** (`event_time`, `mention_time`,
   `valid_from`/`valid_to`); 1a: lexical-primary retrieval (FTS5 BM25,
   phased ranking, token-budget admission with feature logging);
   1b: dense arm + RRF behind the judgment-list gate.
3. **Phase 2 — the research payload**: dialogue-domain heads gated on the
   annotation study's per-category agreement ceilings, supersession
   retrained on conversational updates, calibrated abstention published as
   coverage-accuracy curves.
4. **Phase 3 — climb by category**: per-ability scoreboard, attack the
   weakest column each iteration; abstention as a structural edge —
   encoders are *calibratable* (temperature-scaled, versioned), and the
   calibration's transfer to benchmark register is measured, not assumed.

## Layout

```
src/cognitrace/
  harness/     schema, dataset loaders, reader/judge, CLI (run|score|download)
  baselines/   full-context (ceiling), naive-rag (FTS5 BM25 floor)
tests/         loader + baseline smoke tests (no network)
data/          benchmark datasets (gitignored)
results/       per-run response + graded files (gitignored)
```
