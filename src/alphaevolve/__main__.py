"""``python -m alphaevolve``: the entry point a control box calls."""

from __future__ import annotations

import sys

from alphaevolve.cli import main

if __name__ == "__main__":
    sys.exit(main())
