"""Protocol governance: Full Disclosure manifests, the estimate firewall,
the judge verdict cache, and the judge drift sentinel.

Rules implemented here (research/design_scaffold_2026-07.md S14):
  - No FDR, not a result: every results file opens with a manifest that
    names everything needed to re-run it (models, prompt SHAs, dataset SHA,
    harness git SHA, seed).
  - SPEC estimate firewall: any run/score whose effective config deviates
    from the checked-in protocol file is machine-labeled "estimate".
  - Verdict cache: judge verdicts are cached to disk keyed by the full
    judging context, so re-scoring (dual answer keys, corrected keys,
    challenge-by-regrade) costs zero judge calls.
  - Sentinel: before scoring, re-judge a frozen set of (question, gold,
    response, expected) tuples and halt on verdict drift — the tripwire for
    silent API-side judge model swaps.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_FILE = REPO_ROOT / "protocol_v1.json"
SENTINEL_FILE = REPO_ROOT / "tests" / "judge_sentinel.jsonl"
SENTINEL_BASELINE_FILE = REPO_ROOT / "tests" / "judge_sentinel_baseline.json"

# Fields of the protocol file that must match the effective config for a
# number to carry the pinned label. Anything else -> "estimate".
_PINNED_FIELDS = ("reader_model", "judge_model", "prompts")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_info() -> dict:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except OSError:
            return None

    sha = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def load_protocol() -> dict | None:
    if PROTOCOL_FILE.exists():
        return json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    return None


def build_manifest(*, dataset: str, dataset_path: str, dataset_sha256: str,
                   system: str, system_params: dict, reader_model: str,
                   prompts: dict[str, str], seed: int, limit: int,
                   results_path: str | None = None) -> dict:
    """The run-time half of the FDR. Judge fields are added at score time
    (run and score are separate so regrade never re-spends reader calls)."""
    from .datasets import is_sync_watched  # local import: avoids a module cycle

    manifest = {
        "fdr": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_info(),
        "dataset": dataset,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha256,
        "dataset_path_sync_watched": is_sync_watched(dataset_path) if dataset_path else None,
        "system": system,
        "system_params": system_params,
        "reader_model": reader_model,
        "prompts": prompts,
        "seed": seed,
        "limit": limit,
    }
    if results_path is not None:
        manifest["results_path_sync_watched"] = is_sync_watched(results_path)
    manifest["protocol"] = classify_run(manifest)
    return manifest


def classify_run(manifest: dict) -> str:
    """Estimate firewall, run half: partial runs and off-protocol readers
    are estimates no matter what the scorer later does."""
    proto = load_protocol()
    if proto is None:
        return "estimate"
    if manifest.get("limit"):
        return "estimate"
    if manifest.get("reader_model") != proto.get("reader_model"):
        return "estimate"
    if manifest.get("prompts") != proto.get("prompts"):
        return "estimate"
    return str(proto.get("name", "pinned-v1"))


def classify_score(manifest: dict, judge_model: str, answer_key: str) -> str:
    """Estimate firewall, score half: the label can only stay pinned if the
    run half was pinned AND the judge/key match the protocol."""
    label = manifest.get("protocol", "estimate")
    if label == "estimate":
        return "estimate"
    proto = load_protocol() or {}
    if judge_model != proto.get("judge_model"):
        return "estimate"
    if answer_key not in (proto.get("answer_keys") or [answer_key]):
        return "estimate"
    return label


class VerdictCache:
    """Disk cache of judge verdicts. Key covers everything that could change
    a verdict; a hit is bit-identical regrading."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["k"]] = row

    @staticmethod
    def key(judge_model: str, prompt_sha: str, question: str, gold: str, response: str) -> str:
        return sha256_text("\x1f".join((judge_model, prompt_sha, question, gold, response)))

    def get(self, k: str) -> dict | None:
        return self._data.get(k)

    def put(self, k: str, correct: bool, tier: str, raw: str) -> None:
        row = {"k": k, "correct": correct, "tier": tier, "raw": raw}
        self._data[k] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


@dataclass
class SentinelReport:
    total: int
    flips: int
    flip_rate: float
    baseline_recorded: bool

    @property
    def ok(self) -> bool:
        # >2% flip-rate vs the recorded baseline halts scoring (S14).
        return not self.baseline_recorded or self.flip_rate <= 0.02


def run_sentinel(judge_fn, judge_model: str) -> SentinelReport | None:
    """Re-judge the frozen sentinel set. `judge_fn(question, gold, response,
    is_abstention, model) -> bool`. Returns None when no sentinel file exists
    (Phase 0 curation pending)."""
    if not SENTINEL_FILE.exists():
        return None
    items = [
        json.loads(ln)
        for ln in SENTINEL_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    baseline: dict[str, bool] = {}
    if SENTINEL_BASELINE_FILE.exists():
        baseline = json.loads(SENTINEL_BASELINE_FILE.read_text(encoding="utf-8")).get(judge_model, {})
    verdicts: dict[str, bool] = {}
    flips = 0
    for item in items:
        verdict = judge_fn(
            item["question"], item.get("gold", ""), item["response"],
            item.get("is_abstention", False), judge_model,
        )
        verdicts[item["id"]] = verdict
        if item["id"] in baseline and baseline[item["id"]] != verdict:
            flips += 1
    if not baseline:
        # First run against this judge: record the baseline.
        all_baselines = {}
        if SENTINEL_BASELINE_FILE.exists():
            all_baselines = json.loads(SENTINEL_BASELINE_FILE.read_text(encoding="utf-8"))
        all_baselines[judge_model] = verdicts
        SENTINEL_BASELINE_FILE.write_text(json.dumps(all_baselines, indent=1), encoding="utf-8")
        return SentinelReport(total=len(items), flips=0, flip_rate=0.0, baseline_recorded=False)
    compared = sum(1 for i in verdicts if i in baseline)
    rate = flips / compared if compared else 0.0
    return SentinelReport(total=compared, flips=flips, flip_rate=rate, baseline_recorded=True)


def evidence_recall(retrieved: list[str], gold: list[str], k: int) -> float | None:
    """Fraction of gold ids present in the top-k retrieved ids. None when the
    dataset provides no gold ids for this question (not scored, not zero)."""
    if not gold:
        return None
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


# --- Statistical gate (design_scaffold.md A4: two noise floors) ------------
#
# Paired within-harness comparisons (same questions, same seeds, same judge)
# use a paired bootstrap: resample question indices with replacement, so
# each resample keeps A/B paired per question. A win needs BOTH a 95% CI
# excluding zero AND >=3-point magnitude — bootstrap significance alone is
# not enough at typical eval sizes (n~150-1000) to rule out a real-but-tiny
# effect that ships as tuning noise a week later.
#
# Cross-lab comparisons (our number vs a number from someone else's harness,
# judge, prompts, seeds) cannot be paired at all - confounds are unmeasured
# - so the older, blunter 10-point floor stands instead.

_CROSS_LAB_FLOOR_PTS = 10.0


@dataclass
class PairedGateResult:
    n: int
    diff_pts: float  # mean(a) - mean(b), in percentage points
    ci_lo_pts: float
    ci_hi_pts: float
    n_resamples: int
    magnitude_pts: float

    @property
    def significant(self) -> bool:
        # 95% CI excludes zero.
        return self.ci_lo_pts > 0.0 or self.ci_hi_pts < 0.0

    @property
    def magnitude_ok(self) -> bool:
        return abs(self.diff_pts) >= self.magnitude_pts

    @property
    def passed(self) -> bool:
        return self.significant and self.magnitude_ok


def paired_bootstrap_gate(
    a_correct: list[bool],
    b_correct: list[bool],
    *,
    n_resamples: int = 10000,
    magnitude_pts: float = 3.0,
    seed: int = 0,
) -> PairedGateResult:
    """A4's paired-within-harness floor. `a_correct`/`b_correct` are
    per-question correctness for the SAME ordered set of questions (index i
    in both lists must be the same question) - the pairing is what lets the
    bootstrap cancel per-question difficulty instead of averaging it away."""
    n = len(a_correct)
    if n == 0 or len(b_correct) != n:
        raise ValueError("paired_bootstrap_gate requires two equal-length, aligned lists")
    diffs = [(1 if a else 0) - (1 if b else 0) for a, b in zip(a_correct, b_correct)]
    observed_pts = 100.0 * sum(diffs) / n
    rng = random.Random(seed)
    resample_means: list[float] = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        resample_means.append(100.0 * total / n)
    resample_means.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    return PairedGateResult(
        n=n,
        diff_pts=observed_pts,
        ci_lo_pts=resample_means[lo_idx],
        ci_hi_pts=resample_means[hi_idx],
        n_resamples=n_resamples,
        magnitude_pts=magnitude_pts,
    )


def cross_lab_gate(our_pts: float, other_pts: float, floor_pts: float = _CROSS_LAB_FLOOR_PTS) -> bool:
    """A4's blunt cross-lab floor: unpaired numbers from different harnesses
    need a bigger gap before we call it a difference at all."""
    return abs(our_pts - other_pts) >= floor_pts


# --- Pre-registered anchor band (Sprint 2.2; RESEARCH_BASIS §5.1) ----------
#
# The anti-tuning commitment only works if the band is on the record BEFORE
# the run that gets compared against it - written here, checked in, and
# read by the scoring code, not typed into a results writeup after the
# number is already known.

ANCHOR_BANDS: dict[str, dict[str, float]] = {
    "locomo_full_context": {"center": 72.90, "tolerance": 4.0},
}


def anchor_band(name: str) -> dict[str, float]:
    if name not in ANCHOR_BANDS:
        raise KeyError(f"no pre-registered anchor band named {name!r}")
    return ANCHOR_BANDS[name]


def anchor_band_ok(name: str, observed_pts: float) -> bool:
    band = anchor_band(name)
    return abs(observed_pts - band["center"]) <= band["tolerance"]
