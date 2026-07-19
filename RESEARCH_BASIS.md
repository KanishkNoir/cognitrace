# CogniTrace research basis

*Every design decision in this project derives from a measured finding in the
CogniKernel research corpus (`CogniKernel/research/`, private). This document
is the mapping. Nothing gets built that doesn't trace to a row here.
The full corpus harvest — five deep-read sweeps across ~60 docs — lives in
[`research/cognikernel_corpus_synthesis.md`](research/cognikernel_corpus_synthesis.md);
its §8 lists the design deltas adopted on top of this document. The
panel-adjudicated design decisions superseding parts of this document live in
[`research/design_scaffold_2026-07.md`](research/design_scaffold_2026-07.md).*

---

## 0. The uncomfortable finding we enter with

Our own research says these leaderboards measure the wrong thing:
**recall benchmarks saturate and fail to predict action-level behavior**
(`paper_outline_2026-07.md` §2 — an agent can recite a decision and still
violate it; the counterfactual grid was built precisely because LoCoMo-style
QA can't see that). We enter the leaderboards anyway, with eyes open, for two
reasons:

1. **Positioning** — LongMemEval/LoCoMo are where Mem0/Zep/Supermemory are
   compared; absence reads as inability.
2. **The critique is only citable if we've run the gauntlet** — "your
   benchmark saturates" lands differently from a system that scores
   competitively on it AND publishes the diagnostics the leaderboard omits.

So CogniTrace's deliverable is a *score plus the columns the leaderboard
doesn't report*: memory-construction cost (LLM calls per turn: ours is 0),
tokens per query, evidence-recall (did retrieval surface the gold session,
independent of reader luck), abstention calibration, and update/staleness
handling. That extra table is the research contribution; the score is the
entry ticket.

---

## 1. Encoder-side findings → CogniTrace design

| # | Finding (source) | Measured evidence | CogniTrace design consequence |
|---|---|---|---|
| E1 | **Encoders beat small LLMs at extraction-as-classification** (`extraction_paradigm_review.md` Spike B) | LFM2.5-230M (the sub-500M extraction champion) lost to the 33M encoder on accuracy AND meta-framing, at ~700× latency | Encoder-only construction is evidence-backed, not ideology. No SLM fallback path; the "no LLM constructs memory" claim is the thesis under test |
| E2 | **Measure the label ceiling before training** (`encoder_strengthening.md` P0) | Aggregate eval accuracy was *above* the 0.625 inter-annotator ceiling — the ruler was lying; real headroom was concentrated per-class (THREAD: 0.96 ceiling vs 0.13 F1) | Phase 2 starts with a dialogue-fact annotation study: per-category agreement ceilings on LoCoMo/LongMemEval-style facts BEFORE any head is trained. Track per-class F1 vs ceiling, never aggregate accuracy |
| E3 | **Role + previous-turn context breaks the framing ceiling** (`encoder_strengthening.md` P2; shipped, +0.068 macroF1) | Sentence-only heads can't see "this QUOTES a prior decision"; composing `[role] prev ⟶ current` fixed exactly that class | Dialogue heads take composed speaker+context input from day one. LoCoMo is two-speaker dialogue — reported speech ("Rob said he might quit") is the same meta-framing problem we already solved |
| E4 | **Class prior drives capture balance** (`encoder_strengthening.md` P3) | Matching training NOISE proportion to deployment moved false_capture 0.68→0.40 with no architecture change | Measure the deployment prior (facts per dialogue turn in LoCoMo ≪ decisions per coding transcript) and resample training data to it before comparing anything |
| E5 | **Calibration is a downstream property, not cosmetics** (`encoder_strengthening.md` P1) | Head confidence IS the event weight; miscalibration (0.78 on errors vs 0.88 on correct) distorts ranking and budget | Temperature-scale every head; calibrated confidence feeds retrieval rank AND the abstention decision — which LongMemEval grades directly. Abstention should be our structural edge: encoder heads are *calibratable* (raw logits recorded immutably; temperature scaling versioned in extractor config, applied at materialization), and the calibration's transfer to benchmark register is measured, not assumed |
| E6 | **Synthetic-heavy training generalizes narrowly** (`extraction_paradigm_review.md` §1) | 99%-synthetic corpus was the cleanest explanation for the meta-framing plateau; humanization + real-distribution active learning were the levers | Train dialogue heads on real benchmark-adjacent register from the start (MSC, synthetic *humanized* dialogues — **zero LoCoMo data**: locomo10.json is ten conversations and the public file IS the benchmark, so any "train split" is direct contamination); pre-registered data-lineage doc with overlap checks against both eval sets |

## 2. Framework-side findings → CogniTrace design

| # | Finding (source) | Measured evidence | CogniTrace design consequence |
|---|---|---|---|
| F1 | **Supersession fails when extraction fails** (`rootcause_supersession_extraction.md`) | The bcrypt→argon2id chain never linked because no topic-bearing decision was ever extracted; four independent gates each would have blocked it | Extraction must emit *subject-keyed* facts (entity + attribute), not trigger-matched sentences. Supersession links on shared subject **across types**. And: never drop facts by speaker role — in dialogue, both speakers' turns carry facts |
| F2 | **Supersession chains are the most critical AND most fragile fact class** (+0.81 lift, 0/8 corruption-resist; `paper_outline_2026-07.md`) | Auto-memory served stale values from un-updated notes; stale memory is followed into output 75–85% of the time regardless of system | LongMemEval's *knowledge-update* category is our home turf — latest-wins chains with explicit validity. Target it as a headline per-category number |
| F3 | **Temporal validity ≠ recency** (`memory_decay_research.md`; Zep's temporal-KG results) | Session-based hyperbolic decay with protected categories preserved old-but-binding constraints; event-time vs mention-time distinction is where incumbents score worst (2025 peer-reviewed: Mem0 49%, Zep 63.8% temporal; 2026 self-reports claim 76–95 but independent runs land the same systems at 58–80 overall — see `research/external_landscape_2026-07.md` §1) | Schema carries `event_time` (when it happened) separate from `mention_time` (when it was said) and `superseded_at`. Temporal questions resolve against validity intervals, not embedding similarity. This is the name-sake bet |
| F4 | **Two complementary memory channels** (`publication_draft_2026-07.md` §3.1) | Event memory carries what the artifact can't encode; structural memory carries its current shape (2–10× fewer reads) | Dialogue analog of the skeleton: a compact **entity graph** of the speaker's world (people, places, jobs, dates — encoder NER, no LLM), injected alongside event memory; answers "who/what is related" without transcript re-reads. Multi-hop questions traverse it |
| F5 | **Compact injection wins** (`injection_format/ckl_v2_cipher.md`, `format_sandbox/`) | 1,360-token block vs 6,000+ industry norm; 71% CKL reduction with no compliance loss; U-curve section ordering | Report **tokens-per-query** as a first-class leaderboard column (Mem0 self-reports ~6.9k; we should undercut it hard). Reuse the budgeted drop-to-fit compressor |
| F6 | **Lexical-primary retrieval with dense as a fused signal** (CogniKernel retrieval stack) | BM25 ∪ dense → RRF beat pure-vector in our setting; type-restricted pools rescued prohibitions from topical crowd-out | Same spine, vendored. Type-restricted pool analog, two-tier: rule-resolved time anchors become a first-phase validity-interval SQL filter (rank lexically within the window); soft/inferred temporality is a second-phase feature plus a validity-pool arm in the fusion union — boost, never a serial hard gate (serial hard gates are how the bcrypt chain died) |
| F7 | **Reliability spine is part of the result** (CI gate, fail-open, idempotent replay) | The engineering discipline is why the live-arm numbers were reproducible | Same Definition of Done: harness + system fully rerunnable by a stranger from the repo |

## 3. Eval-framework findings → harness design

These come from `paper_outline_2026-07.md` §3 (the bias-defense section) and
`benchmarking/methodology.md`, and they apply *verbatim* to how we run
LongMemEval/LoCoMo — where self-reported, protocol-unstated numbers are
exactly the credibility problem (see the public Zep-vs-Mem0 dispute):

- **Cross-family judge** — reader and judge from different vendors, judge does
  narrow classification with ground truth in context, verdicts cached to disk.
- **Tiered grading** — deterministic string/date match first, LLM judge only
  for the remainder; judge-overturn rate monitored (>30% = broken grader).
- **Publish adverse findings** — per-category numbers including the ones we
  lose; the floor (naive RAG) and ceiling (full context) always in the table.
- **Non-saturating diagnostics** — evidence-recall@k against the datasets'
  gold session/dialogue ids (retrieval quality independent of reader), and
  corruption-style probes on the knowledge-update category (does the system
  serve the stale value when the update is removed? — the counterfactual
  instrument, ported).
- **Protocol pinned in the results file** — reader model, judge model, prompts,
  dataset variant, all recorded per run; no number leaves the repo without it.

## 4. What is genuinely new research here (not ported)

1. **Dialogue-domain typed schema** — `FACT / PREFERENCE / EVENT_DATED /
   RELATIONSHIP` plus an `asserts_change` flag replaces the coding taxonomy
   (UPDATE is a supersession *edge* — subject key + provenance + time — not
   an event type); requires the E2 annotation study to establish ceilings
   first.
2. **Temporal validity intervals** (F3) — the schema and retrieval work that
   CogniKernel only gestured at (session decay ≠ event-time reasoning).
   Columns land in Phase 1 (rule-resolved population); heads light them up
   in Phase 2.
3. **Encoder NER entity graph** (F4) — structural memory for a life instead
   of a codebase.
4. **Abstention as calibrated refusal** (E5) — thresholded head confidence →
   "I don't know", graded by LongMemEval's abstention category.

## 5. Sequencing (research-first ordering)

1. **Harness + baselines** (exists as skeleton) — hit the anchor band
   (full-context LoCoMo 72.90 ± 4, pre-registered) with full-context /
   naive-RAG / filesystem-grep baselines; add evidence-recall diagnostics.
   *No system code until the ruler is trusted (the P0 lesson, E2).*
2. **Annotation study** — per-category agreement ceilings on dialogue facts.
3. **Spine port + lexical-only system** — vendored store/retrieval/
   supersession with subject-keyed facts (F1); establishes OUR floor.
4. **Dialogue heads** (E2–E6) + **temporal validity** (F3) + **entity graph**
   (F4) — measured per-category against the floor, weakest column first.
5. **Publish** — score table + the honest columns (§0), rerunnable.
