"""Command line entry point. Subcommands are registered as their modules land."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch", description=__doc__)
    parser.add_subparsers(dest="command", metavar="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
