# Store contracts (Sprint 3.6)

Promises this store makes, and the ones it deliberately does not — written
down before Phase 1a retrieval gets built on top, per the engineering-
doctrine convergence on shipping evidence with the claim (and the Rust
honesty-about-non-promises norm).

## Promises

- **Replay determinism.** `rebuild_from_raw` reproduces `memory_records` and
  `supersessions` byte-for-byte from `memory_events` alone (never by
  re-running the extractor over `raw_evidence` — S6 forbids recomputing
  recorded encoder outputs on replay). It is an explicit, deliberate,
  MUTATING call — the CLI `rebuild --from-raw` command, or a one-time
  `record_replay_baseline`. `doctor`'s I7 check is READ-ONLY: it hashes
  whatever is *already* materialized and compares to the recorded
  baseline; it never re-derives. A health check must never itself be a
  write.
- **Baseline semantics.** The replay baseline is a frozen snapshot taken
  once, not a live recomputation. Any ingest after recording it legitimately
  changes the materialized state, so I7 will (correctly) flag a mismatch —
  re-run `record_replay_baseline` after any ingest batch that should become
  the new reference point. I7's job is catching *unexplained* drift
  (corruption, a botched migration, a bypassed invariant) between baseline
  recordings, not detecting that new data was ingested.
- **Atomicity.** Every `ingest_turn` call is one transaction: either
  `raw_evidence` + `extraction_jobs` + `memory_events` + derived tables all
  land, or none of them do. A caught extraction failure still commits
  (as a dead-letter row) — that is a successful, atomic outcome, not a
  partial one.
- **Isolation.** One SQLite file per conversation (S7); there is no
  cross-conversation shared state to isolate.
- **Idempotency.** Re-ingesting the same `(turn_id, extractor_version)` pair
  is a no-op that returns the original result, never a duplicate row.
- **Schema evolution.** `meta.schema_version` is bumped on any DDL change;
  a store file's version is checked before ingest/doctor run against it
  (checked at the CLI layer, Sprint 3.2/4.7).
- **Promotion protocol.** An `extractor_version` may not be used by
  `ingest_turn` until `verify_batch1_parity` has set
  `batch1_parity_verified_at` — enforced by `UnverifiedExtractorError`, not
  just documented.
- **Freshness states.** A `memory_records` row is *live* (`valid_to IS
  NULL`), *superseded* (`valid_to` set, `superseded_by` set), or — for
  `EVENT_DATED` — *episodic* (`supersedable = 0`, never closes). There is no
  fourth "unknown" state in Sprint 3; that is a Phase 1a/1b question once
  the temporal resolver and subject-key normalizer exist.

## Non-promises

- **No score stability across extractor versions.** `rebuild --reextract`
  is expected to produce different `memory_events` under a new
  `extractor_version` — that is reinterpretation, not replay, and ships
  SPEC-labeled, never silently merged with pinned numbers.
- **No latency SLOs before the floor is measured** (A6). Sprint 3 makes no
  timing promise; Sprint 4.8 measures the intrinsic floor first.
- **No embedding byte-stability.** The `embeddings` table is a sidecar
  populated from Phase 1b onward; nothing here promises embeddings survive
  a re-embed under a new model/onnxruntime version.
- **No subject-key canonicalization stability.** Sprint 3's supersession
  rule is exact-string subject-key matching — a deliberately naive
  placeholder proving the mechanism, not the real normalizer (Sprint 4.2).
  Its collision/split behavior is expected to change once that lands.
- **No multi-writer support.** One connection, one writer, per conversation
  file. Concurrent writers to the same file are out of scope, not merely
  untested.
