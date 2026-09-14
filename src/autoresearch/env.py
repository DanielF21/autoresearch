"""The one place the Sail API key is read from.

The tool shell starts fresh from the user's profile, so an interactive
``export`` does not reach it. A project local ``.env`` does. Values already in
the environment always win, so a real export or a CI secret is never
overwritten by a stale file.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY = "SAIL_API_KEY"
DOTENV = Path(".env")


def load_dotenv(path: Path = DOTENV) -> None:
    """Put every ``NAME=value`` line of ``path`` into the environment, if absent."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def api_key(path: Path = DOTENV) -> str:
    """The Sail API key, from the environment or ``.env``. Exits if there is none."""
    load_dotenv(path)
    key = os.environ.get(KEY, "")
    if not key:
        raise SystemExit(
            f"{KEY} is not set and {path.resolve()} has no {KEY} line.\n"
            f"  echo '{KEY}=sk_...' >> {path}"
        )
    return key


def launch_env(path: Path = DOTENV) -> dict[str, str]:
    """What a run launched inside the control box needs in its environment.

    The Sail key is required, and it is also the key tracing uses.
    """
    return {KEY: api_key(path)}
