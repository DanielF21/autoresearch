"""A writer method: the prompts, lint patterns and knobs that hill climbing tunes.

A method is a directory, ``method.toml`` plus one markdown file per prompt, and it
is identified by the hash of its contents. Every score and every draft records
that hash, so two scores are comparable only when they came from the same bytes.
Production loads a method from ``scribe/methods/``; experiments live under
``configs/scribe/methods/``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.scribe.layout import sha256_bytes

PROMPTS = ("reviewer", "compare", "writer", "judge_discrimination", "judge_novelty")
REVIEW_MODES = ("diff_only", "full")
PACKAGED = Path(__file__).parent / "methods"


class MethodError(ValueError):
    pass


@dataclass(frozen=True)
class SlopPattern:
    name: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class WriterMethod:
    name: str
    dir: Path
    hash: str
    review_mode: str
    k: int
    iterations: int
    patience: int
    exemplars: int
    angles: tuple[str, ...]
    decoys: int
    shuffles: int
    accept_rate: float | None
    lint_max_corpus_rate: float
    require_input_coverage: bool
    slop: tuple[SlopPattern, ...]
    prompts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dir": str(self.dir),
            "hash": self.hash,
            "review_mode": self.review_mode,
            "k": self.k,
            "iterations": self.iterations,
            "patience": self.patience,
            "exemplars": self.exemplars,
            "decoys": self.decoys,
            "shuffles": self.shuffles,
            "accept_rate": self.accept_rate,
        }


def method_hash(path: Path) -> str:
    parts: list[bytes] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            parts.append(p.relative_to(path).as_posix().encode() + b"\x00" + p.read_bytes())
    return sha256_bytes(b"\x01".join(parts))


def _positive(table: dict[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise MethodError(f"{where}.{key} must be a positive integer")
    return value


def load_method(path: Path) -> WriterMethod:
    toml_path = path / "method.toml"
    if not toml_path.exists():
        raise MethodError(f"no method.toml in {path}")
    data = tomllib.loads(toml_path.read_text())
    review = data.get("review", {})
    write = data.get("write", {})
    judge = data.get("judge", {})
    checks = data.get("checks", {})

    mode = review.get("mode", "diff_only")
    if mode not in REVIEW_MODES:
        raise MethodError(f"review.mode must be one of {REVIEW_MODES}")
    angles = write.get("angles", [])
    if not angles or not all(isinstance(a, str) and a for a in angles):
        raise MethodError("write.angles must list at least one angle")
    accept = judge.get("accept_rate")
    if accept is not None and not (isinstance(accept, int | float) and 0.0 <= accept <= 1.0):
        raise MethodError("judge.accept_rate must be between 0 and 1, or absent")

    slop: list[SlopPattern] = []
    for item in data.get("slop", []):
        try:
            slop.append(SlopPattern(str(item["name"]), re.compile(str(item["regex"]), re.M)))
        except (KeyError, re.error) as e:
            raise MethodError(f"bad slop pattern {item!r}: {e}") from e

    prompts: dict[str, str] = {}
    for name in PROMPTS:
        p = path / f"{name}.md"
        if not p.exists():
            raise MethodError(f"method {path} has no {name}.md")
        prompts[name] = p.read_text()

    return WriterMethod(
        name=str(data.get("name", path.name)),
        dir=path,
        hash=method_hash(path),
        review_mode=mode,
        k=_positive(write, "k", "write"),
        iterations=_positive(write, "iterations", "write"),
        patience=_positive(write, "patience", "write"),
        exemplars=_positive(write, "exemplars", "write"),
        angles=tuple(angles),
        decoys=_positive(judge, "decoys", "judge"),
        shuffles=_positive(judge, "shuffles", "judge"),
        accept_rate=None if accept is None else float(accept),
        lint_max_corpus_rate=float(checks.get("lint_max_corpus_rate", 0.0)),
        require_input_coverage=bool(checks.get("require_input_coverage", True)),
        slop=tuple(slop),
        prompts=prompts,
    )
