"""Run writer methods over a fixed dev set and record comparable scores.

Each method runs the production draft loop, unchanged, on each dev set candidate.
A score row records the method, corpus and dev set hashes, so rows are compared
only when all three match. A row that exists is never recomputed.
"""

from __future__ import annotations

import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.scribe import runread
from autoresearch.scribe.config import ScribeConfig
from autoresearch.scribe.corpus import Corpus
from autoresearch.scribe.layout import append_jsonl, read_jsonl, sha256_bytes
from autoresearch.scribe.method import WriterMethod
from autoresearch.scribe.repo import RepoCache
from autoresearch.scribe.write import (
    RoleCaps,
    Roles,
    prepare_candidate,
    total_usage,
    write_for_candidate,
    writer_inputs,
)

SCORES = "scores.jsonl"


@dataclass(frozen=True)
class DevSet:
    name: str
    hash: str
    items: tuple[tuple[Path, int], ...]


def load_devset(path: Path, root: Path) -> DevSet:
    data = tomllib.loads(path.read_text())
    items: list[tuple[Path, int]] = []
    for block in data.get("items", []):
        run = Path(block["run"])
        run = run if run.is_absolute() else root / run
        items += [(run, int(n)) for n in block["attempts"]]
    if not items:
        raise ValueError(f"{path}: no [[items]]")
    return DevSet(str(data.get("name", path.stem)), sha256_bytes(path.read_bytes()), tuple(items))


def run_dev(
    out: Path,
    devset: DevSet,
    methods: list[WriterMethod],
    cfg: ScribeConfig,
    corpus: Corpus,
    roles: Roles,
    cache: RepoCache,
) -> list[dict[str, Any]]:
    scores = out / SCORES
    done = {(r["method_hash"], r["run"], r["attempt"]) for r in read_jsonl(scores)}
    rows: list[dict[str, Any]] = []
    for method in methods:
        for run_dir, number in devset.items:
            if (method.hash, str(run_dir), number) in done:
                continue
            target = runread.read_target(run_dir)
            candidates = runread.load_candidates(run_dir)
            matches = [c for c in candidates if c.number == number]
            if not matches or not matches[0].patch:
                raise ValueError(f"{run_dir} has no patch for attempt {number}")
            prepared = prepare_candidate(cache, target, matches[0])
            d = out / "variants" / method.hash[:12] / f"{target.run_id}_{matches[0].dirname}"
            outcome = write_for_candidate(
                d,
                writer_inputs(run_dir, target, candidates, prepared, corpus, cache),
                corpus,
                method,
                roles,
                RoleCaps.of(cfg),
                seed=cfg.corpus.seed,
                allow_uncalibrated=True,
            )
            best = outcome.best
            row = {
                "method": method.name,
                "method_dir": str(method.dir),
                "method_hash": method.hash,
                "devset": devset.name,
                "devset_hash": devset.hash,
                "corpus_hash": corpus.hash,
                "run": str(run_dir),
                "attempt": number,
                "dir": str(d),
                "status": outcome.status,
                "gates_ok": bool(best and best.gates_ok),
                "critique_rate": None if best is None else best.rate,
                "final_rate": None if outcome.final is None else outcome.final.rate,
                "final_chance": None if outcome.final is None else outcome.final.chance,
                "usage": total_usage(d),
            }
            append_jsonl(scores, row)
            rows.append(row)
    return rows


def compare_table(rows: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["method_hash"][:12], r["corpus_hash"][:8]), []).append(r)
    lines = [
        "method | hash | corpus | drafts | gates ok | mean final rate | mean chance",
        "---|---|---|---|---|---|---",
    ]
    for (name, h, corpus_hash), rs in sorted(groups.items()):
        rates = [r["final_rate"] for r in rs if r["final_rate"] is not None]
        chances = [r["final_chance"] for r in rs if r["final_chance"] is not None]
        lines.append(
            f"{name} | {h} | {corpus_hash} | {len(rs)} | {sum(r['gates_ok'] for r in rs)} | "
            f"{statistics.mean(rates):.2f} | {statistics.mean(chances):.2f}"
            if rates and chances
            else f"{name} | {h} | {corpus_hash} | {len(rs)} | {sum(r['gates_ok'] for r in rs)} | n/a | n/a"
        )
    return "\n".join(lines)
