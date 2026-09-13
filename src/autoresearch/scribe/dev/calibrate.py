"""Before a discrimination score is trusted: can the judge tell anything, and what does it see?

Three kinds of case, each one body placed in the candidate slot among real holdout
decoys:

- ``real``: a real holdout body. A judge reading style picks it at about chance.
- ``slop``: a body from a frozen slop set. A useful judge picks it well above chance.
- ``topic``: a real body about performance. Drafts are always about performance, so
  a judge that picks these above chance is reading the topic, not the writing.

No threshold is built in. The summary puts each kind's rate next to chance, and
what counts as good enough is decided by a person looking at it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.model.protocol import ChatModel
from autoresearch.scribe.corpus import Corpus, CorpusPR
from autoresearch.scribe.judges.discrimination import discriminate
from autoresearch.scribe.layout import read_json, write_json_once
from autoresearch.scribe.loop import Caps
from autoresearch.scribe.method import WriterMethod

PERF = re.compile(r"(?i)\b(perf|speed\w*|faster|slow\w*|optimi[sz]\w*|benchmark\w*)")
KINDS = ("real", "slop", "topic")


def load_slop(d: Path) -> list[tuple[str, str, str]]:
    """Each ``*.md`` file: a first line ``# title``, then the body."""
    out: list[tuple[str, str, str]] = []
    for p in sorted(d.glob("*.md")):
        text = p.read_text()
        first, _, rest = text.partition("\n")
        if not first.startswith("# ") or not rest.strip():
            raise ValueError(f"{p}: expected '# title' on the first line and a body after it")
        out.append((p.stem, first[2:].strip(), rest.strip()))
    return out


@dataclass(frozen=True)
class Case:
    kind: str
    id: str
    title: str
    body: str
    decoys: tuple[tuple[str, str], ...]


def _rotated(prs: list[CorpusPR], offset: int, count: int) -> tuple[tuple[str, str], ...]:
    if not prs:
        return ()
    return tuple(
        (prs[(offset + i) % len(prs)].title, prs[(offset + i) % len(prs)].body)
        for i in range(min(count, len(prs)))
    )


def is_perf(pr: CorpusPR) -> bool:
    return bool(PERF.search(pr.title) or any(PERF.search(label) for label in pr.labels))


def plan_cases(
    corpus: Corpus, slop: list[tuple[str, str, str]], trials: int, decoys: int
) -> list[Case]:
    holdout = list(corpus.holdout)
    if len(holdout) < 2:
        raise ValueError("calibration needs at least two holdout bodies")
    m = min(decoys, len(holdout) - 1)
    cases: list[Case] = []
    for t in range(trials):
        real = holdout[t % len(holdout)]
        pool = [p for p in holdout if p.number != real.number]
        cases.append(
            Case("real", f"real_{real.number}_{t}", real.title, real.body, _rotated(pool, t, m))
        )
        if slop:
            sid, title, body = slop[t % len(slop)]
            cases.append(Case("slop", f"slop_{sid}_{t}", title, body, _rotated(holdout, t, m)))
    perf = [p for p in corpus.prs if is_perf(p)]
    for t in range(min(trials, len(perf))):
        pr = perf[t]
        pool = [p for p in holdout if p.number != pr.number]
        cases.append(
            Case("topic", f"topic_{pr.number}_{t}", pr.title, pr.body, _rotated(pool, t, m))
        )
    return cases


def calibrate(
    out: Path,
    model: ChatModel,
    method: WriterMethod,
    corpus: Corpus,
    slop: list[tuple[str, str, str]],
    *,
    trials: int,
    seed: int,
    caps: Caps,
) -> dict[str, Any]:
    tallies: dict[str, dict[str, float]] = {}
    for i, case in enumerate(plan_cases(corpus, slop, trials, method.decoys)):
        path = out / case.kind / f"{case.id}.json"
        if path.exists():
            rec = read_json(path)
        else:
            disc = discriminate(
                model,
                method.prompts["judge_discrimination"],
                (case.title, case.body),
                list(case.decoys),
                shuffles=method.shuffles,
                seed=seed * 1000 + i,
                caps=caps,
                cache_key=f"scribe-calibrate-{case.kind}",
            )
            rec = {"kind": case.kind, "id": case.id, **disc.to_dict()}
            write_json_once(path, rec)
        t = tallies.setdefault(case.kind, {"cases": 0, "valid": 0, "picked": 0, "chance_sum": 0.0})
        t["cases"] += 1
        t["valid"] += int(rec["valid"])
        t["picked"] += int(rec["picked"])
        t["chance_sum"] += float(rec["chance"]) * int(rec["valid"])

    summary: dict[str, Any] = {
        "method_hash": method.hash,
        "corpus_hash": corpus.hash,
        "kinds": {},
    }
    for kind in KINDS:
        if kind not in tallies:
            continue
        t = tallies[kind]
        valid = int(t["valid"])
        summary["kinds"][kind] = {
            "cases": int(t["cases"]),
            "valid_trials": valid,
            "picked": int(t["picked"]),
            "rate": None if not valid else t["picked"] / valid,
            "chance": None if not valid else t["chance_sum"] / valid,
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
