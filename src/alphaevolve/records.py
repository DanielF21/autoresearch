"""What an AlphaEvolve attempt records beyond the harness's attempt files.

The first line of an attempt's ``transcript.jsonl`` is its lineage: the island
it was sampled for, the parent and the programs its prompt showed, the meta
prompt it used and any it wrote, and the block texts of the program it made.
Everything else a candidate is, the harness already writes: the patch, the
usage, and the referee's measurement. The database and the meta prompt
population are rebuilt from these two records, so nothing else is state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from autoresearch import history
from autoresearch.types import Attempt, AttemptRef

LINEAGE_KIND = "lineage"


class RecordError(RuntimeError):
    pass


@dataclass(frozen=True)
class Lineage:
    island: int
    parent: str
    parent_number: int
    top: tuple[str, ...]
    diverse: tuple[str, ...]
    inspirations: tuple[str, ...]
    meta: str
    meta_generated: str = ""
    texts: tuple[str, ...] | None = None
    error: str = ""

    def to_line(self) -> str:
        return json.dumps(
            {
                "kind": LINEAGE_KIND,
                "island": self.island,
                "parent": self.parent,
                "parent_number": self.parent_number,
                "top": list(self.top),
                "diverse": list(self.diverse),
                "inspirations": list(self.inspirations),
                "meta": self.meta,
                "meta_generated": self.meta_generated,
                "texts": None if self.texts is None else list(self.texts),
                "error": self.error,
            },
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lineage:
        texts = d.get("texts")
        return cls(
            island=int(d["island"]),
            parent=str(d["parent"]),
            parent_number=int(d["parent_number"]),
            top=tuple(str(x) for x in d.get("top", [])),
            diverse=tuple(str(x) for x in d.get("diverse", [])),
            inspirations=tuple(str(x) for x in d.get("inspirations", [])),
            meta=str(d["meta"]),
            meta_generated=str(d.get("meta_generated", "")),
            texts=None if texts is None else tuple(str(t) for t in texts),
            error=str(d.get("error", "")),
        )


def read_lineage(paths: history.RunPaths, ref: AttemptRef) -> Lineage:
    path = paths.attempt(ref) / history.TRANSCRIPT_JSONL
    first = path.read_text().split("\n", 1)[0] if path.exists() else ""
    try:
        record = json.loads(first)
    except json.JSONDecodeError as e:
        raise RecordError(f"{path} does not open with a lineage record") from e
    if not isinstance(record, dict) or record.get("kind") != LINEAGE_KIND:
        raise RecordError(f"{path} does not open with a lineage record")
    return Lineage.from_dict(record)


def fitness(attempt: Attempt) -> float:
    """The referee's held out geomean for a real speedup, and 0 for anything else."""
    m = attempt.measurement
    if m is None or not m.clears_noise or m.speedup is None:
        return 0.0
    return m.speedup


def meta_id(number: int) -> str:
    """The id of the meta prompt attempt ``number`` wrote."""
    return f"m{number:04d}"
