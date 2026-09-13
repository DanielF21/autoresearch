"""The Scribe's own config, separate from any run config.

A run's ``config.toml`` is frozen and byte compared on resume, so nothing about
the Scribe can live there. Every value in a scribe TOML is an untuned knob until
a dev run or a calibration says otherwise, and the file's comments must say which.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLES = ("reviewer", "writer", "judge")
CORPUS_N_RANGE = (10, 20)


class ScribeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RoleModel:
    """One model role's settings and caps. Mirrors the worker's, so SailChatModel takes it."""

    model: str
    reasoning_effort: str
    completion_window: str
    turn_timeout: int
    max_turns: int
    max_seconds: int
    max_input_tokens: int


@dataclass(frozen=True)
class CorpusConfig:
    n: int
    pool: int
    exclude_authors: tuple[str, ...]
    holdout_n: int
    seed: int
    template_path: str


@dataclass(frozen=True)
class DevConfig:
    review_repeats: int
    calibration_trials: int
    blind_real_count: int


@dataclass(frozen=True)
class ScribeConfig:
    output_root: Path
    cache_root: Path
    method: Path
    min_inputs: int
    corpus: CorpusConfig
    models: dict[str, RoleModel]
    dev: DevConfig
    source: Path

    def role(self, name: str) -> RoleModel:
        return self.models[name]


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ScribeConfigError(f"[{key}] is required")
    return value


def _int(table: dict[str, Any], key: str, where: str, minimum: int = 0) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ScribeConfigError(f"{where}.{key} must be an integer >= {minimum}")
    return value


def _str(table: dict[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ScribeConfigError(f"{where}.{key} must be a non empty string")
    return value


def _path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else base / p


def parse_scribe_config(text: str, source: Path, root: Path) -> ScribeConfig:
    """``root`` is what relative paths resolve against: the repository root."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ScribeConfigError(f"{source}: {e}") from e

    scribe = _table(data, "scribe")
    corpus = _table(data, "corpus")
    filters = _table(data, "filters")
    dev = _table(data, "dev")
    models_table = _table(data, "models")

    n = _int(corpus, "n", "corpus", 1)
    lo, hi = CORPUS_N_RANGE
    if not lo <= n <= hi:
        raise ScribeConfigError(f"corpus.n must be between {lo} and {hi}, got {n}")
    pool = _int(corpus, "pool", "corpus", n)
    holdout_n = _int(corpus, "holdout_n", "corpus", 1)
    if holdout_n >= n:
        raise ScribeConfigError("corpus.holdout_n must leave at least one exemplar")
    exclude = corpus.get("exclude_authors", [])
    if not isinstance(exclude, list) or not all(isinstance(a, str) for a in exclude):
        raise ScribeConfigError("corpus.exclude_authors must be a list of logins")

    models: dict[str, RoleModel] = {}
    for role in ROLES:
        t = models_table.get(role)
        if not isinstance(t, dict):
            raise ScribeConfigError(f"[models.{role}] is required")
        where = f"models.{role}"
        models[role] = RoleModel(
            model=_str(t, "model", where),
            reasoning_effort=_str(t, "reasoning_effort", where),
            completion_window=_str(t, "completion_window", where),
            turn_timeout=_int(t, "turn_timeout", where, 1),
            max_turns=_int(t, "max_turns", where, 1),
            max_seconds=_int(t, "max_seconds", where, 1),
            max_input_tokens=_int(t, "max_input_tokens", where, 1),
        )

    return ScribeConfig(
        output_root=_path(_str(scribe, "output_root", "scribe"), root),
        cache_root=_path(_str(scribe, "cache_root", "scribe"), root),
        method=_path(_str(scribe, "method", "scribe"), root),
        min_inputs=_int(filters, "min_inputs", "filters", 1),
        corpus=CorpusConfig(
            n=n,
            pool=pool,
            exclude_authors=tuple(exclude),
            holdout_n=holdout_n,
            seed=_int(corpus, "seed", "corpus"),
            template_path=str(corpus.get("template_path", ".github/PULL_REQUEST_TEMPLATE.md")),
        ),
        models=models,
        dev=DevConfig(
            review_repeats=_int(dev, "review_repeats", "dev", 1),
            calibration_trials=_int(dev, "calibration_trials", "dev", 1),
            blind_real_count=_int(dev, "blind_real_count", "dev", 1),
        ),
        source=source,
    )


def load_scribe_config(path: Path, root: Path | None = None) -> ScribeConfig:
    return parse_scribe_config(path.read_text(), path, root or Path.cwd())
