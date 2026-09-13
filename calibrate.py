#!/usr/bin/env python3
"""Kept so ``uv run calibrate.py <config> --rounds 7`` in the frozen t1 configs still works.

The command is ``autoresearch calibrate``, which also writes the floors into the config.
"""

import sys

from autoresearch.cli import main

if __name__ == "__main__":
    sys.exit(main(["calibrate", *sys.argv[1:]]))
