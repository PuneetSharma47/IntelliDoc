"""Multi-extractor arbitration.

Two extractors that agree are worth more than the better one alone; two that
disagree tell you exactly which field to send to a human. This module merges
candidate lists from several extractors:

* identical values from different extractors get the strongest score plus an
  agreement bonus, and the field is reported as ``agreed``;
* every distinct value is kept as an alternative so the reviewer sees the
  competing readings instead of one arbitrary winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import FieldName
from .base import Candidate, ExtractionContext, Extractor, pick_best


@dataclass
class EnsembleResult:
    candidates: list[Candidate] = field(default_factory=list)
    agreed_fields: set[FieldName] = field(default_factory=set)
    disagreements: dict[FieldName, list] = field(default_factory=dict)


def _key(candidate: Candidate) -> str:
    value = candidate.value
    if isinstance(value, float):
        return f"{round(value, 2):.2f}"
    return str(value).strip().lower()


class EnsembleExtractor:
    """Runs each member extractor over the same context and merges candidates."""

    name = "ensemble"

    def __init__(self, members: list[Extractor], agreement_bonus: float = 0.06) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member extractor")
        self.members = members
        self.agreement_bonus = agreement_bonus

    def run(self, ctx: ExtractionContext) -> EnsembleResult:
        per_member: dict[str, list[Candidate]] = {}
        for member in self.members:
            per_member[member.name] = member.extract(ctx)

        merged: dict[FieldName, dict[str, Candidate]] = {}
        sources: dict[FieldName, dict[str, set[str]]] = {}
        for member_name, candidates in per_member.items():
            for cand in candidates:
                bucket = merged.setdefault(cand.field, {})
                keys = sources.setdefault(cand.field, {})
                key = _key(cand)
                keys.setdefault(key, set()).add(member_name)
                existing = bucket.get(key)
                if existing is None or cand.score > existing.score:
                    bucket[key] = cand
                else:
                    existing.provenance.rule = f"{existing.provenance.rule}+{member_name}"

        result = EnsembleResult()
        for fname, bucket in merged.items():
            for key, cand in bucket.items():
                supporters = sources[fname][key]
                if len(supporters) > 1:
                    cand.score = min(0.99, cand.score + self.agreement_bonus)
                    cand.note = f"agreed by {', '.join(sorted(supporters))}"
                    result.agreed_fields.add(fname)
                result.candidates.append(cand)
            ranked = sorted(bucket.values(), key=lambda c: -c.score)
            if (
                len(ranked) > 1
                and abs(ranked[0].score - ranked[1].score) < 0.02
                and _key(ranked[0]) != _key(ranked[1])
                and len(sources[fname][_key(ranked[0])]) == len(sources[fname][_key(ranked[1])])
            ):
                result.disagreements[fname] = [ranked[0].value, ranked[1].value]
        return result

    def extract(self, ctx: ExtractionContext) -> list[Candidate]:  # pragma: no cover
        return self.run(ctx).candidates


def group_by_field(candidates: list[Candidate]) -> dict[FieldName, list[Candidate]]:
    grouped: dict[FieldName, list[Candidate]] = {}
    for cand in candidates:
        grouped.setdefault(cand.field, []).append(cand)
    return grouped


def best_per_field(candidates: list[Candidate]) -> dict[FieldName, Candidate | None]:
    return {fname: pick_best(cands) for fname, cands in group_by_field(candidates).items()}
