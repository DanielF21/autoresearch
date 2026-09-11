"""Load SAIL_API_KEY from a local .env if it is not already in the environment.

The tool shell starts fresh from the user's profile, so an interactive `export`
does not reach it. This makes a project-local .env work without one.
"""

from __future__ import annotations

import os
import pathlib


def load_dotenv(path: str | pathlib.Path = ".env") -> None:
    p = pathlib.Path(path)
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        os.environ.setdefault(key, val)


def require(name: str) -> str:
    load_dotenv()
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} is not set.\n"
            f"  Either add it to ~/.zshenv, or write it to {pathlib.Path('.env').resolve()}:\n"
            f"    echo '{name}=sk_...' >> .env"
        )
    return val
