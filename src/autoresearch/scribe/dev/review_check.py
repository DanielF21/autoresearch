"""The reviewer against patches whose mergeability was measured.

The labels come from ``artifacts/generality.md`` section 1. The check counts what
the reviewer said and whether its reasons name the inputs at risk. It does not
score from the verdict alone: every labelled regressor adds an import and no safe
patch does, so "rejects new imports" would match every label without understanding
a single cost model. A verdict counts as agreeing only when the reason is right.
"""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.model.protocol import ChatModel
from autoresearch.scribe import runread
from autoresearch.scribe.layout import read_json, sha256_bytes, write_json_once
from autoresearch.scribe.loop import Caps
from autoresearch.scribe.method import WriterMethod
from autoresearch.scribe.repo import RepoCache
from autoresearch.scribe.review import Verdict, review_candidate
from autoresearch.scribe.write import prepare_candidate, review_context, save_loop

LABELS = ("safe", "regresses")
IMPORT_WORDS = ("import", "numpy", "dependency")


@dataclass(frozen=True)
class Label:
    run: Path
    attempt: int
    label: str
    source: str


@dataclass(frozen=True)
class LabelSet:
    labels: tuple[Label, ...]
    risk_keywords: tuple[str, ...]
    hash: str


def load_labels(path: Path, root: Path) -> LabelSet:
    data = tomllib.loads(path.read_text())
    labels: list[Label] = []
    for item in data.get("items", []):
        if item["label"] not in LABELS:
            raise ValueError(f"{path}: label must be one of {LABELS}")
        run = Path(item["run"])
        labels.append(
            Label(
                run if run.is_absolute() else root / run,
                int(item["attempt"]),
                item["label"],
                item["source"],
            )
        )
    keywords = tuple(str(k).lower() for k in data.get("risk_keywords", []))
    if not labels or not keywords:
        raise ValueError(f"{path}: needs [[items]] and risk_keywords")
    return LabelSet(tuple(labels), keywords, sha256_bytes(path.read_bytes()))


def verdict_text(v: Verdict) -> str:
    parts = [v.cost_before, v.cost_after, v.reasons, *v.inputs_at_risk]
    parts += [c.claim for c in v.concerns]
    return " ".join(parts).lower()


def review_check(
    out: Path,
    labels: LabelSet,
    method: WriterMethod,
    reviewer: ChatModel,
    caps: Caps,
    cache: RepoCache,
    *,
    repeats: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in labels.labels:
        target = runread.read_target(item.run)
        c = next(c for c in runread.load_candidates(item.run) if c.number == item.attempt)
        prepared = prepare_candidate(cache, target, c)
        ctx = review_context(cache, target, prepared)
        counts: Counter[str] = Counter()
        risk_named = cost_changed = import_only = agrees = 0
        for r in range(1, repeats + 1):
            d = out / f"{target.run_id}_{c.dirname}" / f"repeat_{r}"
            if (d / "verdict.json").exists():
                rec = read_json(d / "verdict.json")
            else:
                outcome = review_candidate(
                    reviewer,
                    method,
                    c,
                    prepared.facts,
                    ctx,
                    caps,
                    cache_key=f"scribe-review-check-{target.run_id}-{c.dirname}",
                    mode="diff_only",
                )
                save_loop(d, "reviewer", outcome.loop)
                rec = outcome.to_dict()
                write_json_once(d / "verdict.json", rec)
            if not rec["valid"] or rec["verdict"] is None:
                counts["invalid"] += 1
                continue
            v = Verdict.from_dict(rec["verdict"])
            counts[v.verdict] += 1
            text = verdict_text(v)
            named = any(k in text for k in labels.risk_keywords)
            risk_named += named
            cost_changed += v.cost_changed
            import_only += (not named) and any(w in text for w in IMPORT_WORDS)
            if item.label == "regresses":
                agrees += v.verdict != "mergeable" and named
            else:
                agrees += v.verdict == "mergeable"
        rows.append(
            {
                "run": target.run_id,
                "attempt": c.number,
                "label": item.label,
                "source": item.source,
                "verdicts": dict(counts),
                "cost_model_changed": cost_changed,
                "risk_inputs_named": risk_named,
                "import_reason_without_risk": import_only,
                "agrees_for_the_right_reason": agrees,
                "repeats": repeats,
            }
        )
    summary = {"method_hash": method.hash, "labels_hash": labels.hash, "rows": rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
