"""Run configuration: one TOML file per run, validated into frozen records.

The file is the only place a run's settings live. The orchestrator copies the
file bytes into the run directory unchanged, so a run can always be traced back
to exactly what it was configured with.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOX_SIZES = ("s", "m", "l")


class ConfigError(ValueError):
    """A config file that parsed but does not describe a runnable experiment."""


@dataclass(frozen=True)
class TargetSpec:
    """One piece of code plus the one benchmark that measures it."""

    name: str
    repo: str
    sha: str
    hot_file: str
    test_file: str
    graph: str
    call: str
    allow: tuple[str, ...]
    deny: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo": self.repo,
            "sha": self.sha,
            "hot_file": self.hot_file,
            "test_file": self.test_file,
            "graph": self.graph,
            "call": self.call,
            "allow": list(self.allow),
            "deny": list(self.deny),
        }


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    reasoning_effort: str
    completion_window: str
    max_turns: int
    max_seconds: int
    max_input_tokens: int
    turn_timeout: int


@dataclass(frozen=True)
class RefereeConfig:
    pairs: int
    threshold: float
    repeats_per_launch: int
    min_clean_pairs: int
    hash_seeds: tuple[int, ...]


@dataclass(frozen=True)
class BoxConfig:
    worker_size: str
    referee_size: str
    control_size: str
    disk_gib: int


@dataclass(frozen=True)
class StorageConfig:
    volume: str
    mount: str


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    width: int
    rounds: int
    target: TargetSpec
    worker: WorkerConfig
    referee: RefereeConfig
    boxes: BoxConfig
    storage: StorageConfig
    source_text: str

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.source_text.encode()).hexdigest()[:12]

    @property
    def runs_root(self) -> Path:
        return Path(self.storage.mount) / "runs"

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] section")
    return value


def _require(section: dict[str, Any], name: str, key: str) -> Any:
    if key not in section:
        raise ConfigError(f"[{name}] is missing '{key}'")
    return section[key]


def _positive_int(section: dict[str, Any], name: str, key: str) -> int:
    value = _require(section, name, key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"[{name}].{key} must be an integer of at least 1, got {value!r}")
    return value


def _str(section: dict[str, Any], name: str, key: str) -> str:
    value = _require(section, name, key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"[{name}].{key} must be a non empty string")
    return value


def _str_list(section: dict[str, Any], name: str, key: str) -> tuple[str, ...]:
    value = _require(section, name, key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"[{name}].{key} must be a list of strings")
    return tuple(value)


def _size(section: dict[str, Any], name: str, key: str) -> str:
    value = _str(section, name, key)
    if value not in BOX_SIZES:
        raise ConfigError(f"[{name}].{key} must be one of {BOX_SIZES}, got {value!r}")
    return value


def parse_config(text: str) -> RunConfig:
    """Parse and validate TOML text. Raises ConfigError with the field named."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"not valid TOML: {e}") from e

    run = _section(raw, "run")
    target = _section(raw, "target")
    worker = _section(raw, "worker")
    referee = _section(raw, "referee")
    boxes = _section(raw, "boxes")
    storage = _section(raw, "storage")

    threshold = _require(referee, "referee", "threshold")
    if not isinstance(threshold, int | float) or threshold <= 1.0:
        raise ConfigError("[referee].threshold must be a number above 1.0")
    pairs = _positive_int(referee, "referee", "pairs")
    min_clean = referee.get("min_clean_pairs", max(1, (2 * pairs) // 3))
    if not isinstance(min_clean, int) or min_clean < 1 or min_clean > pairs:
        raise ConfigError("[referee].min_clean_pairs must be between 1 and pairs")
    seeds_raw = referee.get("hash_seeds", [0, 1, 2, 3, 4])
    if (
        not isinstance(seeds_raw, list)
        or not seeds_raw
        or not all(isinstance(s, int) for s in seeds_raw)
    ):
        raise ConfigError("[referee].hash_seeds must be a non empty list of integers")

    return RunConfig(
        run_id=_str(run, "run", "run_id"),
        width=_positive_int(run, "run", "width"),
        rounds=_positive_int(run, "run", "rounds"),
        target=TargetSpec(
            name=_str(target, "target", "name"),
            repo=_str(target, "target", "repo"),
            sha=_str(target, "target", "sha"),
            hot_file=_str(target, "target", "hot_file"),
            test_file=_str(target, "target", "test_file"),
            graph=_str(target, "target", "graph"),
            call=_str(target, "target", "call"),
            allow=_str_list(target, "target", "allow"),
            deny=_str_list(target, "target", "deny"),
        ),
        worker=WorkerConfig(
            model=_str(worker, "worker", "model"),
            reasoning_effort=_str(worker, "worker", "reasoning_effort"),
            completion_window=_str(worker, "worker", "completion_window"),
            max_turns=_positive_int(worker, "worker", "max_turns"),
            max_seconds=_positive_int(worker, "worker", "max_seconds"),
            max_input_tokens=_positive_int(worker, "worker", "max_input_tokens"),
            turn_timeout=_positive_int(worker, "worker", "turn_timeout"),
        ),
        referee=RefereeConfig(
            pairs=pairs,
            threshold=float(threshold),
            repeats_per_launch=_positive_int(referee, "referee", "repeats_per_launch"),
            min_clean_pairs=min_clean,
            hash_seeds=tuple(seeds_raw),
        ),
        boxes=BoxConfig(
            worker_size=_size(boxes, "boxes", "worker_size"),
            referee_size=_size(boxes, "boxes", "referee_size"),
            control_size=_size(boxes, "boxes", "control_size"),
            disk_gib=int(boxes.get("disk_gib", 32)),
        ),
        storage=StorageConfig(
            volume=_str(storage, "storage", "volume"),
            mount=_str(storage, "storage", "mount"),
        ),
        source_text=text,
    )


def load_config(path: Path) -> RunConfig:
    return parse_config(path.read_text())
