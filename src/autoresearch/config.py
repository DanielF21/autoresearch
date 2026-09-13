"""Run configuration: one TOML file per run, validated into frozen records.

The file is the only place a run's settings live. The orchestrator copies the
file bytes into the run directory unchanged, so a run can always be traced back
to exactly what it was configured with.

The ``[target]`` section describes the code under test completely: what to
import and under what name, where the package sits in the tree, what to install,
how each input is built, what call is timed, how its result is checked, and
which tests are the module's and which are the whole suite. Nothing about a
target lives in code. See ``TargetSpec``.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOX_SIZES = ("s", "m", "l")

# How a [target] written before 2026-09-12 is read. Those configs had no
# ``package`` key, spelled an input as one expression bound to ``G`` with the
# package bound to ``nx``, named the module suite ``test_file`` and took the
# full suite to be the first path segment of ``hot_file``, and relied on the
# image installing these three packages. A frozen config.toml inside an
# existing run directory is never rewritten, so this table is what keeps those
# runs readable and resumable. It is a record reader, not a default: a config
# that names ``package`` and still uses the old keys is refused by name.
LEGACY_NETWORKX_TARGET = {
    "package": "networkx",
    "alias": "nx",
    "package_root": ".",
    "pip": ("numpy", "scipy", "pandas"),
    "input_name": "G",
}


class ConfigError(ValueError):
    """A config file that parsed but does not describe a runnable experiment."""


@dataclass(frozen=True)
class BenchmarkInput:
    """One input the target is timed on, and the floor that input's noise sets.

    ``setup`` is Python statements, run once per launch in a namespace that
    holds the package under the target's alias and ``ROOT``, a ``pathlib.Path``
    of the tree under test. The target's ``call`` is then evaluated in that
    namespace. An input is a namespace, not a value.

    The floor is per input because it is a property of how long the call takes,
    not of the patch. A one millisecond call's ratio scatters about twice as
    wide as a 1.4 second one, so a single floor is either too loose for the slow
    input or too tight for the fast one. See artifacts/generality.md section 8.
    ``None`` means not yet calibrated: the calibrate and check commands accept
    such an input, a run refuses it by name.
    """

    name: str
    setup: str
    noise_floor: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "setup": self.setup, "noise_floor": self.noise_floor}


@dataclass(frozen=True)
class SuitePaths:
    """The two pytest targets the referee runs, both relative to the repo root.

    ``module`` is the hot module's own tests, run by the worker while it works
    and by the referee first. ``full`` is the whole suite, run by the referee
    only. They are two scopes because a measurement needs both to say the tests
    pass; a target with one suite names the same path twice.
    """

    module: str
    full: str

    def to_dict(self) -> dict[str, Any]:
        return {"module": self.module, "full": self.full}


@dataclass(frozen=True)
class TargetSpec:
    """One piece of code plus the set of inputs it is measured on.

    The package is imported by path from a git worktree and never installed,
    which is what lets two trees be timed from one box. ``package_root`` is the
    directory put on ``sys.path`` for that, relative to the repo root: ``"."``
    for a flat layout, ``"src"`` for a src layout. ``hot_file`` is the file the
    worker is pointed at and whose cProfile self time the referee checks ran.

    It was one input until 2026-09-12. Timing a single point in input space
    taught the agents to specialise to that point: four of the seven best
    patches were up to 4x slower than stock on sparse graphs while being 40x to
    80x faster on the one graph the referee timed. artifacts/generality.md has
    the measurements.
    """

    name: str
    repo: str
    sha: str
    package: str
    alias: str
    package_root: str
    hot_file: str
    call: str
    tests: SuitePaths
    inputs: tuple[BenchmarkInput, ...]
    allow: tuple[str, ...]
    deny: tuple[str, ...]
    pip: tuple[str, ...] = ()
    apt: tuple[str, ...] = ()
    fingerprint: str = ""
    docs: tuple[str, ...] = ()

    @property
    def primary(self) -> BenchmarkInput:
        """The input listed first. Ordering only: every input is timed alike."""
        return self.inputs[0]

    @property
    def uncalibrated(self) -> tuple[str, ...]:
        """Names of inputs with no noise floor yet. Empty is what a run needs."""
        return tuple(i.name for i in self.inputs if i.noise_floor is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo": self.repo,
            "sha": self.sha,
            "package": self.package,
            "alias": self.alias,
            "package_root": self.package_root,
            "hot_file": self.hot_file,
            "call": self.call,
            "tests": self.tests.to_dict(),
            "inputs": [i.to_dict() for i in self.inputs],
            "allow": list(self.allow),
            "deny": list(self.deny),
            "pip": list(self.pip),
            "apt": list(self.apt),
            "fingerprint": self.fingerprint,
            "docs": list(self.docs),
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
    # System prompt versions by worker slot: slot w runs prompts[w % len(prompts)].
    # One name means every worker is told the same thing, which is every run
    # before the four prompt experiment.
    prompts: tuple[str, ...] = ("v1",)


@dataclass(frozen=True)
class RefereeConfig:
    """How the referee times. The noise floor is not here: it is per input."""

    pairs: int
    repeats_per_launch: int
    min_clean_pairs: int
    hash_seeds: tuple[int, ...]
    # Passes of ``pairs`` retried on an input with fewer than ``min_clean_pairs``
    # clean, before the input is recorded as untimed. The cheap inputs are the
    # ones that fail, so a retry costs seconds; the width experiment lost an
    # input on 13 of 912 attempts with one retry.
    timing_retries: int = 2


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
class ObserveConfig:
    """Live tracing of worker attempts. Absent section means off.

    A view only: the run directory stays the record, so nothing here changes
    what a run produces.
    """

    enabled: bool = False
    session_prefix: str = ""


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
    observe: ObserveConfig = ObserveConfig()

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


def _prompt_versions(worker: dict[str, Any]) -> tuple[str, ...]:
    """``[worker].prompts``: absent means v1 for every slot. Each name must be a
    version the worker package has, and it is named here so a typo fails the
    config, not the first attempt of a paid run."""
    # Imported here: the prompt package imports this module for the target types.
    from autoresearch.worker.prompt import VERSIONS

    raw = worker.get("prompts", ["v1"])
    if not isinstance(raw, list) or not raw or not all(isinstance(v, str) and v for v in raw):
        raise ConfigError("[worker].prompts must be a non empty list of prompt version names")
    unknown = [v for v in raw if v not in VERSIONS]
    if unknown:
        raise ConfigError(
            f"[worker].prompts names unknown versions {unknown}; known: {sorted(VERSIONS)}"
        )
    return tuple(raw)


def _str_list(section: dict[str, Any], name: str, key: str) -> tuple[str, ...]:
    value = _require(section, name, key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"[{name}].{key} must be a list of strings")
    return tuple(value)


def _optional_str_list(section: dict[str, Any], name: str, key: str) -> tuple[str, ...]:
    return _str_list(section, name, key) if key in section else ()


def _size(section: dict[str, Any], name: str, key: str) -> str:
    value = _str(section, name, key)
    if value not in BOX_SIZES:
        raise ConfigError(f"[{name}].{key} must be one of {BOX_SIZES}, got {value!r}")
    return value


def _noise_floor(entry: dict[str, Any], where: str) -> float | None:
    """Absent means not yet calibrated. Present must be a number above 1.0."""
    if "noise_floor" not in entry:
        return None
    floor = entry["noise_floor"]
    if not isinstance(floor, int | float) or isinstance(floor, bool) or floor <= 1.0:
        raise ConfigError(f"[{where}].noise_floor must be a number above 1.0")
    return float(floor)


def _benchmark_inputs(target: dict[str, Any], legacy: bool) -> tuple[BenchmarkInput, ...]:
    """The [[target.inputs]] tables, in file order. The first one is the primary.

    Order is kept because it is what the worker is shown and what the tables in
    analyze.py column by. Names must be unique: they key every timing in
    measurement.json, and a duplicate would silently merge two inputs.
    """
    raw = _require(target, "target", "inputs")
    if not isinstance(raw, list) or not raw:
        raise ConfigError("[[target.inputs]] must appear at least once")
    inputs: list[BenchmarkInput] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        where = f"target.inputs[{i}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"[[{where}]] must be a table")
        name = _str(entry, where, "name")
        if name in seen:
            raise ConfigError(f"[[target.inputs]] name {name!r} appears twice")
        seen.add(name)
        if legacy:
            setup = f"{LEGACY_NETWORKX_TARGET['input_name']} = {_str(entry, where, 'graph')}"
        else:
            if "graph" in entry:
                raise ConfigError(
                    f"[{where}].graph is the old spelling of an input; write it as "
                    f"'setup', statements that bind the names the call uses"
                )
            setup = _str(entry, where, "setup")
        inputs.append(
            BenchmarkInput(name=name, setup=setup, noise_floor=_noise_floor(entry, where))
        )
    return tuple(inputs)


def _tests(target: dict[str, Any]) -> SuitePaths:
    raw = _require(target, "target", "tests")
    if not isinstance(raw, dict):
        raise ConfigError("[target.tests] must be a table with 'module' and 'full'")
    return SuitePaths(
        module=_str(raw, "target.tests", "module"), full=_str(raw, "target.tests", "full")
    )


def _target(target: dict[str, Any]) -> TargetSpec:
    legacy = "package" not in target
    if legacy:
        if "tests" in target or "setup" in target:
            raise ConfigError("[target] has no 'package' but uses new keys; name the package")
        hot_file = _str(target, "target", "hot_file")
        tests = SuitePaths(module=_str(target, "target", "test_file"), full=hot_file.split("/")[0])
        package = str(LEGACY_NETWORKX_TARGET["package"])
        alias = str(LEGACY_NETWORKX_TARGET["alias"])
        package_root = str(LEGACY_NETWORKX_TARGET["package_root"])
        pip = tuple(LEGACY_NETWORKX_TARGET["pip"])
        apt: tuple[str, ...] = ()
        fingerprint = ""
    else:
        if "test_file" in target:
            raise ConfigError("[target].test_file moved to [target.tests].module")
        hot_file = _str(target, "target", "hot_file")
        tests = _tests(target)
        package = _str(target, "target", "package")
        alias = _str(target, "target", "alias")
        package_root = str(target.get("package_root", "."))
        if not package_root:
            raise ConfigError("[target].package_root must be a non empty relative path")
        pip = _optional_str_list(target, "target", "pip")
        apt = _optional_str_list(target, "target", "apt")
        fingerprint = str(target.get("fingerprint", ""))
    if not alias.isidentifier():
        raise ConfigError(f"[target].alias must be a Python identifier, got {alias!r}")
    return TargetSpec(
        name=_str(target, "target", "name"),
        repo=_str(target, "target", "repo"),
        sha=_str(target, "target", "sha"),
        package=package,
        alias=alias,
        package_root=package_root,
        hot_file=hot_file,
        call=_str(target, "target", "call"),
        tests=tests,
        inputs=_benchmark_inputs(target, legacy),
        allow=_str_list(target, "target", "allow"),
        deny=_str_list(target, "target", "deny"),
        pip=pip,
        apt=apt,
        fingerprint=fingerprint,
        docs=_optional_str_list(target, "target", "docs"),
    )


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

    if "noise_floor" in referee:
        raise ConfigError(
            "[referee].noise_floor moved to [[target.inputs]].noise_floor, one per input"
        )
    pairs = _positive_int(referee, "referee", "pairs")
    min_clean = referee.get("min_clean_pairs", max(1, (2 * pairs) // 3))
    if not isinstance(min_clean, int) or min_clean < 1 or min_clean > pairs:
        raise ConfigError("[referee].min_clean_pairs must be between 1 and pairs")
    retries = referee.get("timing_retries", 2)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ConfigError("[referee].timing_retries must be an integer of at least 0")
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
        target=_target(target),
        worker=WorkerConfig(
            model=_str(worker, "worker", "model"),
            reasoning_effort=_str(worker, "worker", "reasoning_effort"),
            completion_window=_str(worker, "worker", "completion_window"),
            max_turns=_positive_int(worker, "worker", "max_turns"),
            max_seconds=_positive_int(worker, "worker", "max_seconds"),
            max_input_tokens=_positive_int(worker, "worker", "max_input_tokens"),
            turn_timeout=_positive_int(worker, "worker", "turn_timeout"),
            prompts=_prompt_versions(worker),
        ),
        referee=RefereeConfig(
            pairs=pairs,
            repeats_per_launch=_positive_int(referee, "referee", "repeats_per_launch"),
            min_clean_pairs=min_clean,
            hash_seeds=tuple(seeds_raw),
            timing_retries=retries,
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
        observe=_observe(raw.get("observe")),
    )


def _observe(section: Any) -> ObserveConfig:
    """No section means tracing off. A section must say so explicitly."""
    if section is None:
        return ObserveConfig()
    if not isinstance(section, dict):
        raise ConfigError("[observe] must be a section")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("[observe].enabled must be true or false")
    prefix = section.get("session_prefix", "")
    if not isinstance(prefix, str):
        raise ConfigError("[observe].session_prefix must be a string")
    return ObserveConfig(enabled=enabled, session_prefix=prefix)


def load_config(path: Path) -> RunConfig:
    return parse_config(path.read_text())
