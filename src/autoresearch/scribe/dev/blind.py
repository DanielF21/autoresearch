"""Blind ranking: finished drafts mixed with real merged bodies, ranked by a person.

The packet shows every body anonymised the way the judges see them, under a
neutral item number. The key that says which is which is written outside the
packet directory. After the ranking is filled in, ``unblind`` reports where the
drafts landed among the real bodies and whether the judge's rates order the drafts
the way the person did. That comparison decides whether the judges are trusted.
"""

from __future__ import annotations

import random
import statistics
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from autoresearch.scribe.corpus import Corpus
from autoresearch.scribe.judges.common import anonymise
from autoresearch.scribe.layout import (
    read_json,
    read_jsonl,
    stamp,
    write_json_once,
    write_text_once,
)

KEYS_DIR = "blind_keys"


def build_packet(
    dev_dir: Path, corpus: Corpus, *, real_count: int, seed: int, now: datetime | None = None
) -> Path:
    machine: list[tuple[str, str, str, float | None]] = []
    for r in read_jsonl(dev_dir / "scores.jsonl"):
        if r["status"] not in ("accepted", "uncalibrated"):
            continue
        d = Path(r["dir"])
        if not (d / "final_body.md").exists():
            continue
        ident = f"machine:{r['method_hash'][:12]}:{Path(r['run']).name}:{r['attempt']}"
        title = (d / "final_title.txt").read_text().strip()
        machine.append((ident, title, (d / "final_body.md").read_text(), r.get("final_rate")))
    if not machine:
        raise ValueError(f"no finished drafts in {dev_dir / 'scores.jsonl'}")

    items = [(i, t, b) for i, t, b, _ in machine]
    items += [(f"real:{pr.number}", pr.title, pr.body) for pr in list(corpus.holdout)[:real_count]]
    random.Random(seed).shuffle(items)
    labels = [f"{n:02d}" for n in range(1, len(items) + 1)]

    name = stamp(now)
    packet = dev_dir / "blind" / name
    parts = [
        "# Blind ranking",
        "",
        f"{len(items)} pull request descriptions. Some were merged in the repository and some "
        "were drafted by the Scribe. Rank every item in ranking.toml from 1, most clearly "
        f"written by a person, to {len(items)}. Use each rank once.",
    ]
    for label, (_, title, body) in zip(labels, items, strict=True):
        parts += [
            "",
            "---",
            "",
            f"## Item {label}",
            "",
            f"Title: {anonymise(title)}",
            "",
            anonymise(body),
        ]
    write_text_once(packet / "packet.md", "\n".join(parts) + "\n")
    write_text_once(
        packet / "ranking.toml",
        "# 1 is most clearly written by a person. Every item gets a different rank.\n[ranking]\n"
        + "".join(f'"{label}" = 0\n' for label in labels),
    )
    write_json_once(
        dev_dir / KEYS_DIR / f"{name}.json",
        {
            "items": {label: ident for label, (ident, _, _) in zip(labels, items, strict=True)},
            "judge_final_rates": {i: rate for i, _, _, rate in machine},
        },
    )
    return packet


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if den == 0 else num / den


def unblind(packet: Path) -> dict[str, Any]:
    dev_dir = packet.parent.parent
    key = read_json(dev_dir / KEYS_DIR / f"{packet.name}.json")
    ranking = tomllib.loads((packet / "ranking.toml").read_text()).get("ranking", {})
    labels = sorted(key["items"])
    ranks = {label: int(ranking.get(label, 0)) for label in labels}
    if sorted(ranks.values()) != list(range(1, len(labels) + 1)):
        raise ValueError(f"rank every item exactly once, 1 to {len(labels)}")

    ident = key["items"]
    machine = {
        ident[label]: ranks[label] for label in labels if ident[label].startswith("machine:")
    }
    real = {ident[label]: ranks[label] for label in labels if ident[label].startswith("real:")}
    rated = [(key["judge_final_rates"].get(i), r) for i, r in machine.items()]
    rated_pairs = [(float(rate), float(r)) for rate, r in rated if rate is not None]
    result: dict[str, Any] = {
        "items": len(labels),
        "machine_ranks": machine,
        "real_ranks": real,
        "mean_machine_rank": statistics.mean(machine.values()) if machine else None,
        "mean_real_rank": statistics.mean(real.values()) if real else None,
        "best_machine_rank": min(machine.values()) if machine else None,
        "machine_ranked_above_some_real": sum(
            1 for r in machine.values() if real and r < max(real.values())
        ),
        # Positive when drafts the judge picked out less often were also ranked more human.
        "judge_rate_vs_rank_spearman": spearman(
            [p[0] for p in rated_pairs], [p[1] for p in rated_pairs]
        ),
    }
    write_json_once(packet / "agreement.json", result)
    return result
