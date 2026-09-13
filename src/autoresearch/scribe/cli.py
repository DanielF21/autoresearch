"""``uv run python -m autoresearch.scribe RUN_DIR``: filter a run's candidates, then one
model picks one and writes its pull request.

Filtering and fetching merged pull requests call no model. When no candidate passes
the filter, the command prints why and stops before any model call.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from autoresearch.model.protocol import ChatModel, ModelError
from autoresearch.scribe import candidates
from autoresearch.scribe.prs import (
    GhError,
    GhRunner,
    SubprocessGh,
    fetch_merged_prs,
    github_repo,
    load_prs,
    save_prs,
)
from autoresearch.scribe.scribe import ScribeError, run, worker_config

DEFAULT_OUT = Path("runs/scribe")
DEFAULT_N = 20


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoresearch.scribe", description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--repo", help="owner/name; default: the run's target repo")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="merged pull requests to show")
    p.add_argument("--prs", help="a saved prs.json to reuse instead of fetching")
    p.add_argument("--model", help="default: the run's worker model")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p


def main(
    argv: list[str] | None = None,
    *,
    model: ChatModel | None = None,
    gh: GhRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    target = candidates.read_target(run_dir)
    kept, excluded = candidates.filter_candidates(candidates.load_candidates(run_dir))
    for number, reasons in excluded.items():
        print(f"attempt {number:04d} excluded: {'; '.join(reasons)}")
    if not kept:
        print("No candidate passed the filter. No model was called.")
        return 1
    print(f"Passed the filter: {', '.join(f'{c.number:04d}' for c in kept)}")

    try:
        if args.prs:
            merged = load_prs(Path(args.prs))
        else:
            repo = args.repo or github_repo(target.repo)
            merged = fetch_merged_prs(gh or SubprocessGh(), repo, args.n)
    except (GhError, ValueError) as e:
        print(f"stopped: {e}")
        return 1

    out = Path(args.out) / target.run_id / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    excluded_json = {f"{n:04d}": list(reasons) for n, reasons in excluded.items()}
    (out / "excluded.json").write_text(json.dumps(excluded_json, indent=2) + "\n")
    save_prs(out / "prs.json", merged)

    if model is None:
        from autoresearch import env
        from autoresearch.model.sail_model import SailChatModel

        env.load_dotenv()
        model = SailChatModel(worker_config(run_dir, args.model))
    try:
        draft = run(model, target, kept, merged, out)
    except (ModelError, ScribeError) as e:
        print(f"stopped: {e}\n{out}")
        return 1

    print(out)
    if draft.pick is None:
        print(f"No candidate picked: {draft.reasons}")
    else:
        print(f"Picked attempt {draft.pick}: {draft.reasons}\n\n{draft.title}\n\n{draft.body}")
    return 0
