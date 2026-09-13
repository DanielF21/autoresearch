"""The Scribe's command line: ``uv run python -m autoresearch.scribe``.

Commands that call a model print an upper bound on model requests, from the turn
caps, and stop unless ``--go`` is passed. No command prints a cost estimate:
cost is unknown until measured, and every loop records its tokens on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from autoresearch.scribe.config import ScribeConfig, load_scribe_config
from autoresearch.scribe.layout import stamp
from autoresearch.scribe.method import WriterMethod, load_method

if TYPE_CHECKING:
    from autoresearch.scribe.write import Roles

DEFAULT_CONFIG = Path("configs/scribe/scribe.toml")


def _config(args: argparse.Namespace) -> ScribeConfig:
    return load_scribe_config(Path(args.config), Path.cwd())


def _method(args: argparse.Namespace, cfg: ScribeConfig) -> WriterMethod:
    return load_method(Path(args.method) if getattr(args, "method", None) else cfg.method)


def _bound(parts: dict[str, int], go: bool) -> bool:
    """Print the request bound. True when the command may proceed."""
    total = sum(parts.values())
    detail = ", ".join(f"{role} {n}" for role, n in parts.items())
    print(f"At most {total} model requests ({detail}), from the turn caps.")
    print("Cost is unknown until measured; every loop records its tokens.")
    if not go:
        print("Nothing was sent. Pass --go to run.")
    return go


def _roles(cfg: ScribeConfig) -> Roles:
    """Built only after --go, so a dry run never needs credentials or the Sail SDK."""
    from autoresearch import env
    from autoresearch.scribe.model import chat_model
    from autoresearch.scribe.write import Roles

    env.load_dotenv()
    return Roles(
        reviewer=chat_model(cfg.role("reviewer")),
        writer=chat_model(cfg.role("writer")),
        judge=chat_model(cfg.role("judge")),
    )


def draft_bound(cfg: ScribeConfig, method: WriterMethod, drafts: int) -> dict[str, int]:
    w, j = cfg.role("writer"), cfg.role("judge")
    per = method.iterations * method.k
    return {
        "writer": drafts * per * w.max_turns,
        "judge": drafts * (per * (method.shuffles + 1) + method.shuffles) * j.max_turns,
    }


# ----- commands -------------------------------------------------------------------------


def cmd_corpus_fetch(args: argparse.Namespace) -> int:
    from autoresearch.scribe.corpus import SubprocessGh, fetch_corpus
    from autoresearch.scribe.corpus_stats import corpus_stats

    cfg = _config(args)
    out = fetch_corpus(
        SubprocessGh(),
        repo=args.repo,
        n=cfg.corpus.n,
        pool=cfg.corpus.pool,
        exclude_authors=cfg.corpus.exclude_authors,
        holdout_n=cfg.corpus.holdout_n,
        seed=cfg.corpus.seed,
        template_path=cfg.corpus.template_path,
        out_root=cfg.output_root / "corpus",
        stats=corpus_stats,
    )
    print(out)
    return 0


def cmd_corpus_stats(args: argparse.Namespace) -> int:
    from autoresearch.scribe.corpus import load_corpus

    corpus = load_corpus(Path(args.corpus_dir))
    print(
        f"{corpus.repo}: {len(corpus.prs)} PRs, {len(corpus.exemplars)} exemplars, {len(corpus.holdout)} holdout"
    )
    print(json.dumps(corpus.stats, indent=2, sort_keys=True))
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    from autoresearch.scribe import runread
    from autoresearch.scribe.filters import filter_candidate
    from autoresearch.scribe.render import fmt_ratio
    from autoresearch.scribe.repo import PatchApplyError, RepoCache
    from autoresearch.scribe.write import prepare_candidate

    cfg = _config(args)
    run_dir = Path(args.run_dir)
    target = runread.read_target(run_dir)
    cache = RepoCache(cfg.cache_root)
    for c in runread.load_candidates(run_dir):
        f = filter_candidate(c, cfg.min_inputs)
        line = f"{c.dirname}  geomean {fmt_ratio(c.speedup)}  worst {fmt_ratio(c.worst_speedup)}"
        if not f.reviewable:
            print(f"{line}  excluded: {'; '.join(f.reasons)}")
            continue
        try:
            facts = prepare_candidate(cache, target, c).facts
        except PatchApplyError as e:
            print(f"{line}  does not apply: {e}")
            continue
        print(
            f"{line}  {'writable' if f.writable else 'review only'}  lines {facts.changed_lines}"
            f"  imports {list(facts.new_imports)}  guards {list(facts.new_condition_numbers)}"
        )
    return 0


def cmd_write(args: argparse.Namespace) -> int:
    from autoresearch.scribe import runread
    from autoresearch.scribe.corpus import load_corpus
    from autoresearch.scribe.repo import RepoCache
    from autoresearch.scribe.write import (
        new_session_dir,
        run_session,
        session_summary,
        writable_candidates,
    )

    cfg = _config(args)
    method = _method(args, cfg)
    run_dir = Path(args.run_dir)
    n = len(writable_candidates(run_dir, cfg.min_inputs))
    parts = {"reviewer": (n + 1) * cfg.role("reviewer").max_turns, **draft_bound(cfg, method, 1)}
    print(f"{n} writable candidate(s) in {run_dir}; method {method.name} {method.hash[:12]}")
    if not _bound(parts, args.go):
        return 0
    corpus = load_corpus(Path(args.corpus))
    target = runread.read_target(run_dir)
    session = Path(args.session) if args.session else new_session_dir(cfg, target.run_id, method)
    run_session(
        session,
        run_dir,
        cfg,
        method,
        corpus,
        _roles(cfg),
        RepoCache(cfg.cache_root),
        allow_uncalibrated=args.uncalibrated,
    )
    print(json.dumps(session_summary(session), indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from autoresearch.scribe.write import session_summary

    print(json.dumps(session_summary(Path(args.session_dir)), indent=2))
    return 0


def _dev_dir(cfg: ScribeConfig, label: str | None) -> Path:
    return cfg.output_root / "dev" / (label or stamp())


def cmd_dev_calibrate(args: argparse.Namespace) -> int:
    from autoresearch.scribe.corpus import load_corpus
    from autoresearch.scribe.dev.calibrate import calibrate, load_slop, plan_cases
    from autoresearch.scribe.loop import Caps

    cfg = _config(args)
    method = _method(args, cfg)
    corpus = load_corpus(Path(args.corpus))
    slop = load_slop(Path(args.slop))
    trials = args.trials or cfg.dev.calibration_trials
    cases = plan_cases(corpus, slop, trials, method.decoys)
    if not _bound({"judge": len(cases) * method.shuffles * cfg.role("judge").max_turns}, args.go):
        return 0
    roles = _roles(cfg)
    summary = calibrate(
        _dev_dir(cfg, args.label) / "calibration",
        roles.judge,
        method,
        corpus,
        slop,
        trials=trials,
        seed=cfg.corpus.seed,
        caps=Caps.of(cfg.role("judge")),
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_dev_run(args: argparse.Namespace) -> int:
    from autoresearch.scribe.corpus import load_corpus
    from autoresearch.scribe.dev.harness import compare_table, load_devset, run_dev
    from autoresearch.scribe.repo import RepoCache

    cfg = _config(args)
    devset = load_devset(Path(args.devset), Path.cwd())
    methods = [load_method(Path(m)) for m in args.methods.split(",")]
    bound = draft_bound(cfg, methods[0], len(devset.items) * len(methods))
    if not _bound(bound, args.go):
        return 0
    out = _dev_dir(cfg, args.label)
    rows = run_dev(
        out,
        devset,
        methods,
        cfg,
        load_corpus(Path(args.corpus)),
        _roles(cfg),
        RepoCache(cfg.cache_root),
    )
    print(out)
    print(compare_table(rows))
    return 0


def cmd_dev_review_check(args: argparse.Namespace) -> int:
    from autoresearch.scribe.dev.review_check import load_labels, review_check
    from autoresearch.scribe.loop import Caps
    from autoresearch.scribe.repo import RepoCache

    cfg = _config(args)
    method = _method(args, cfg)
    labels = load_labels(Path(args.labels), Path.cwd())
    repeats = args.repeats or cfg.dev.review_repeats
    reviewer = cfg.role("reviewer")
    if not _bound({"reviewer": len(labels.labels) * repeats * reviewer.max_turns}, args.go):
        return 0
    roles = _roles(cfg)
    summary = review_check(
        _dev_dir(cfg, args.label) / "review_check",
        labels,
        method,
        roles.reviewer,
        Caps.of(reviewer),
        RepoCache(cfg.cache_root),
        repeats=repeats,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_dev_blind(args: argparse.Namespace) -> int:
    from autoresearch.scribe.corpus import load_corpus
    from autoresearch.scribe.dev.blind import build_packet

    cfg = _config(args)
    packet = build_packet(
        Path(args.dev_dir),
        load_corpus(Path(args.corpus)),
        real_count=cfg.dev.blind_real_count,
        seed=cfg.corpus.seed,
    )
    print(
        f"{packet / 'packet.md'}\nFill in {packet / 'ranking.toml'}, then run dev unblind {packet}"
    )
    return 0


def cmd_dev_unblind(args: argparse.Namespace) -> int:
    from autoresearch.scribe.dev.blind import unblind

    print(json.dumps(unblind(Path(args.packet_dir)), indent=2))
    return 0


# ----- parser ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoresearch.scribe", description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="command")

    corpus = sub.add_parser("corpus", help="the style corpus").add_subparsers(dest="corpus_command")
    c = corpus.add_parser("fetch", help="freeze a repo's last merged PRs by people (gh, no model)")
    c.add_argument("--repo", required=True, help="owner/name")
    c.set_defaults(func=cmd_corpus_fetch)
    c = corpus.add_parser("stats", help="print a frozen corpus's stats")
    c.add_argument("corpus_dir")
    c.set_defaults(func=cmd_corpus_stats)

    c = sub.add_parser("facts", help="filters and code facts for every attempt in a run (no model)")
    c.add_argument("run_dir")
    c.set_defaults(func=cmd_facts)

    c = sub.add_parser("write", help="production: select a candidate and draft its PR")
    c.add_argument("run_dir")
    c.add_argument("--corpus", required=True)
    c.add_argument("--method")
    c.add_argument("--session", help="resume this session directory")
    c.add_argument(
        "--uncalibrated", action="store_true", help="keep a draft from a method with no accept_rate"
    )
    c.add_argument("--go", action="store_true")
    c.set_defaults(func=cmd_write)

    c = sub.add_parser("show", help="what a session decided")
    c.add_argument("session_dir")
    c.set_defaults(func=cmd_show)

    dev = sub.add_parser("dev", help="tuning the method").add_subparsers(dest="dev_command")
    c = dev.add_parser("calibrate", help="judge rates on real, slop and topic cases")
    c.add_argument("--corpus", required=True)
    c.add_argument("--slop", required=True, help="directory of '# title' markdown files")
    c.add_argument("--method")
    c.add_argument("--trials", type=int)
    c.add_argument("--label")
    c.add_argument("--go", action="store_true")
    c.set_defaults(func=cmd_dev_calibrate)
    c = dev.add_parser("run", help="run methods over a dev set")
    c.add_argument("--corpus", required=True)
    c.add_argument("--devset", required=True)
    c.add_argument("--methods", required=True, help="comma separated method directories")
    c.add_argument("--label")
    c.add_argument("--go", action="store_true")
    c.set_defaults(func=cmd_dev_run)
    c = dev.add_parser("review_check", help="the reviewer on labelled patches, diff only")
    c.add_argument("--labels", required=True)
    c.add_argument("--method")
    c.add_argument("--repeats", type=int)
    c.add_argument("--label")
    c.add_argument("--go", action="store_true")
    c.set_defaults(func=cmd_dev_review_check)
    c = dev.add_parser("blind", help="build a blind ranking packet from a dev run")
    c.add_argument("dev_dir")
    c.add_argument("--corpus", required=True)
    c.set_defaults(func=cmd_dev_blind)
    c = dev.add_parser("unblind", help="score a filled in ranking")
    c.add_argument("packet_dir")
    c.set_defaults(func=cmd_dev_unblind)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stdout)
        return 2
    result: int = func(args)
    return result
