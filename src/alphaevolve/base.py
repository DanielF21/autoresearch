"""A run's fixed inputs: the base files, the blocks, the prompt prefix, the budget.

``prepare`` makes them once, when a run directory is new, and writes them under
``evolve/`` in it; every later launch reads them back, so a resumed run uses
the same blocks and a byte identical prefix. Order matters for what it costs:
the profile document and the budget are checked before a box is created, and
the box that reads the base files is terminated before anything else happens.

The base files come from a referee box's clone at the base commit, never from
the machine the run is launched on, and each is checked against the blob id
git gives it.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from alphaevolve import budget, prompt
from alphaevolve.blocks import Block, candidate_paths, derive
from alphaevolve.config import EvolveConfig, EvolveRunConfig
from autoresearch import history
from autoresearch.boxes.image import REPO_DIR, WORK_DIR
from autoresearch.boxes.protocol import BoxError, BoxFactory
from autoresearch.config import RunConfig
from autoresearch.orchestrator.round import load_docs
from autoresearch.referee.referee import SETUP_TIMEOUT

EVOLVE_DIR = "evolve"
BASE_SUBDIR = "base"
BLOCKS_FILE = "blocks.json"
BUDGET_FILE = "budget.json"
PREFIX_FILE = "prefix.md"
SYSTEM_FILE = "system.md"


class PrepareError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvolveState:
    sources: dict[str, str]
    blocks: tuple[Block, ...]
    system: str
    prefix: str
    budget: int

    @property
    def base_texts(self) -> tuple[str, ...]:
        return tuple(b.text for b in self.blocks)


def blob_sha(data: bytes) -> str:
    """The id git gives a file with these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_sources(boxes: BoxFactory, config: RunConfig, paths: Sequence[str]) -> dict[str, str]:
    box = boxes.create(name=f"evolve-base-{config.run_id}", role="referee")
    try:
        out: dict[str, str] = {}
        for i, path in enumerate(paths):
            spec = shlex.quote(f"{config.target.sha}:{path}")
            dest = f"{WORK_DIR}/evolve_base_{i}"
            r = box.run(
                f"mkdir -p {WORK_DIR} && cd {REPO_DIR} && git rev-parse {spec} "
                f"&& git show {spec} > {dest}",
                timeout=SETUP_TIMEOUT,
            )
            if not r.ok:
                raise BoxError(f"could not read {path} at the base commit: {r.stderr[-500:]}")
            lines = r.stdout.strip().splitlines()
            want = lines[-1].strip() if lines else ""
            data = box.read(dest)
            if blob_sha(data) != want:
                raise BoxError(f"{path} read back as blob {blob_sha(data)}, but git names {want!r}")
            out[path] = data.decode()
        return out
    finally:
        try:
            box.terminate()
        except BoxError as e:
            print(
                f"box {box.name} ({box.box_id}) may still be running: {e}. "
                "autoresearch reap lists every live box.",
                file=sys.stderr,
                flush=True,
            )


def resolve_budget(evolve: EvolveConfig, runs_root: Path) -> int:
    if evolve.budget_tokens is not None:
        return evolve.budget_tokens
    return budget.harness_budget(runs_root / evolve.budget_from_run)


def prepare(
    paths: history.RunPaths, config: EvolveRunConfig, boxes: BoxFactory, runs_root: Path
) -> EvolveState:
    """The run's fixed inputs, made on first launch and read back on every later one."""
    if (paths.root / EVOLVE_DIR / BLOCKS_FILE).exists():
        return load(paths)
    cfg = config.run
    target = cfg.target
    docs = load_docs(paths)
    flat_name = f"profile_{target.primary.name}_flat.txt"
    flat = dict(docs).get(flat_name)
    if flat is None:
        raise PrepareError(
            f"the run has no {flat_name}; autoresearch profile writes it, and the config's "
            "[target].docs must name it"
        )
    wanted = candidate_paths(flat, target.allow, target.deny)
    if not wanted:
        raise PrepareError(f"{flat_name} names no function inside the allowed files")
    tokens = resolve_budget(config.evolve, runs_root)
    sources = fetch_sources(boxes, cfg, wanted)
    blocks = derive(flat, sources, target.allow, target.deny, config.evolve.max_code_length)
    used = {b.path: sources[b.path] for b in blocks}
    system = prompt.system_prompt()
    prefix = prompt.constant_prefix(target, target.sha, cfg.referee.pairs, docs, used, blocks)
    state = EvolveState(sources=used, blocks=blocks, system=system, prefix=prefix, budget=tokens)
    _write(paths, state, config.evolve)
    return state


def _write(paths: history.RunPaths, state: EvolveState, evolve: EvolveConfig) -> None:
    root = paths.root / EVOLVE_DIR
    for path, text in state.sources.items():
        target = root / BASE_SUBDIR / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (root / SYSTEM_FILE).write_text(state.system)
    (root / PREFIX_FILE).write_text(state.prefix)
    (root / BUDGET_FILE).write_text(
        json.dumps({"tokens": state.budget, "from_run": evolve.budget_from_run}, indent=2) + "\n"
    )
    # Last, because its presence is what says the rest is complete.
    (root / BLOCKS_FILE).write_text(
        json.dumps([b.to_dict() for b in state.blocks], indent=2) + "\n"
    )


def load(paths: history.RunPaths) -> EvolveState:
    root = paths.root / EVOLVE_DIR
    blocks = tuple(Block.from_dict(d) for d in json.loads((root / BLOCKS_FILE).read_text()))
    sources: dict[str, str] = {}
    for path in sorted({b.path for b in blocks}):
        file = root / BASE_SUBDIR / path
        if not file.exists():
            raise PrepareError(f"{file} is missing; the run's evolve directory is incomplete")
        sources[path] = file.read_text()
    return EvolveState(
        sources=sources,
        blocks=blocks,
        system=(root / SYSTEM_FILE).read_text(),
        prefix=(root / PREFIX_FILE).read_text(),
        budget=int(json.loads((root / BUDGET_FILE).read_text())["tokens"]),
    )
