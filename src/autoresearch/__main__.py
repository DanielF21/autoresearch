"""``python -m autoresearch``.

The console script is the entry point on the laptop. Inside a box it is not
usable: pip installs it to an interpreter specific directory that is not on the
box's PATH, so the control box invokes the package as a module instead.
"""

from __future__ import annotations

import sys

from autoresearch.cli import main

if __name__ == "__main__":
    sys.exit(main())
