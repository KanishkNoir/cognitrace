"""Measure real collision/split rates for the v0 subject-key normalizer
against real dialogue (Sprint 4.2) -- "the F1-shaped hole gets a number,
not a hope."

Method: LoCoMo dialogue often self-reveals identity in one breath ("my
sister Nina"). Wherever that happens, the relation phrase and the proper
name are known -- by construction, from the text itself -- to be the same
entity. That gives free, dataset-native ground truth for two failure
modes of the naive `speaker:relation` key, with no human annotation
required:

  collision -- the same (speaker, relation) key reveals >=2 DISTINCT
               names across the conversation (two different real people
               would collide onto one key).
  split     -- a (speaker, relation) key reveals exactly one name
               somewhere, but is ALSO mentioned bare (no name) elsewhere
               in the conversation (the same real person would get two
               different keys: "speaker:relation" and the name itself).

Important limitation, stated plainly: this can only see collisions/splits
for relations that get named SOMEWHERE in the conversation. A relation
that is never named is invisible to this method -- so these rates are a
floor on the true problem, not a ceiling. Real coreference measurement
needs the annotation study (A1/A2); this is the number available before
that exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cognitrace.harness.schema import Session
from cognitrace.subject.normalizer import find_relation_name_comentions, find_relation_phrases


@dataclass
class SubjectKeyReport:
    pairs_with_reveal: int
    collisions: int
    splits: int
    total_relation_mentions: int
    collision_examples: list[tuple[str, str, list[str]]] = field(default_factory=list)
    split_examples: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def collision_rate(self) -> float | None:
        return self.collisions / self.pairs_with_reveal if self.pairs_with_reveal else None

    @property
    def split_rate(self) -> float | None:
        return self.splits / self.pairs_with_reveal if self.pairs_with_reveal else None


def _possessor(possessive: str, speaker: str, other_speaker: str) -> str | None:
    # Lowercased to match what normalize_subject() itself would produce as
    # a key (_canon lowercases) -- these rates must reflect the real keys.
    if possessive in ("my", "our"):
        return speaker.lower()
    if possessive == "your":
        return other_speaker.lower()
    return None  # his/her/their: whose relation this is needs coreference


def measure_conversation(sessions: list[Session]) -> SubjectKeyReport:
    speakers = list(dict.fromkeys(t.role for s in sessions for t in s.turns))
    revealed: dict[tuple[str, str], set[str]] = {}
    bare_mentions: dict[tuple[str, str], int] = {}
    total_mentions = 0

    for session in sessions:
        for turn in session.turns:
            other = next((s for s in speakers if s != turn.role), turn.role)
            comentions = find_relation_name_comentions(turn.content)
            named_this_turn: set[tuple[str, str]] = set()
            for possessive, relation, name in comentions:
                possessor = _possessor(possessive, turn.role, other)
                if possessor is None:
                    continue
                key = (possessor, relation)
                revealed.setdefault(key, set()).add(name)
                named_this_turn.add(key)
                total_mentions += 1

            for possessive, relation in find_relation_phrases(turn.content):
                possessor = _possessor(possessive, turn.role, other)
                if possessor is None:
                    continue
                key = (possessor, relation)
                if key not in named_this_turn:
                    bare_mentions[key] = bare_mentions.get(key, 0) + 1
                    total_mentions += 1

    collisions = 0
    splits = 0
    collision_examples: list[tuple[str, str, list[str]]] = []
    split_examples: list[tuple[str, str, str]] = []
    for key, names in revealed.items():
        if len(names) >= 2:
            collisions += 1
            collision_examples.append((key[0], key[1], sorted(names)))
        elif key in bare_mentions:
            splits += 1
            split_examples.append((key[0], key[1], next(iter(names))))

    return SubjectKeyReport(
        pairs_with_reveal=len(revealed),
        collisions=collisions,
        splits=splits,
        total_relation_mentions=total_mentions,
        collision_examples=collision_examples,
        split_examples=split_examples,
    )


def measure_dataset(tasks) -> SubjectKeyReport:
    """Aggregate across every conversation (each Task's own sessions)."""
    total = SubjectKeyReport(0, 0, 0, 0)
    for task in tasks:
        r = measure_conversation(task.sessions)
        total.pairs_with_reveal += r.pairs_with_reveal
        total.collisions += r.collisions
        total.splits += r.splits
        total.total_relation_mentions += r.total_relation_mentions
        total.collision_examples.extend(r.collision_examples)
        total.split_examples.extend(r.split_examples)
    return total
