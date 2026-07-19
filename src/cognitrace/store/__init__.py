"""Event-sourced per-conversation store (Phase 1a spine, S3-S8).

One SQLite file per conversation (S7): cross-contamination is structurally
impossible, hermetic reset is deleting the file, and parallel ingest across
conversations has zero writer contention. `raw_evidence` + `memory_events`
are the append-only ground truth (S5/S6); `memory_records`, `supersessions`,
`records_fts`, and `embeddings` are droppable materialized views that
`rebuild_from_raw` reconstructs deterministically from recorded events —
never by re-running the extractor, which is what makes replay bit-identical
(S6's replay-determinism contract).
"""
