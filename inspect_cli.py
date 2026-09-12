#!/usr/bin/env python3
"""A live shell into a sailbox.

    uv run inspect_cli.py --box <sailbox id or name>
    uv run inspect_cli.py --list
    uv run inspect_cli.py                     # the control box in runs/control.json

Type shell commands and see the output. ``cd`` sticks, because the working
directory is tracked here and passed back in on the next command; the box
itself keeps no shell state between calls. ``Q`` quits.

Meta commands start with a colon:

    :help                 this list
    :boxes                every live box
    :box <id or name>     switch to another box
    :follow <command>     stream output as it is produced, for tail -f and friends
    :get <remote> [local] copy a file down to the laptop
    :timeout <seconds>    how long a command may run, default 120
    :pwd                  where you are

Commands are real and run as root. The control box has the run volume mounted
at /mnt/autoresearch, which holds every measurement of every run, so anything
destructive asks first.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from autoresearch import env
from autoresearch.boxes.protocol import BoxError
from autoresearch.boxes.sail_box import SailBox
from autoresearch.control import deploy as ctl

MARKER = "__inspect_pwd__"
DEFAULT_TIMEOUT = 120
HISTORY = Path.home() / ".cache" / "autoresearch" / "inspect_history"
QUIT = {"q", "quit", "exit"}

# Not a sandbox, just a speed bump in front of the run volume.
DESTRUCTIVE = re.compile(
    r"\brm\s+-[a-z]*[rf]|\bmkfs\b|\bdd\s+if=|\bshred\b|\btruncate\b|\bgit\s+push\b|>\s*/mnt/"
)


def colour(text: str, code: str) -> str:
    return text if not sys.stdout.isatty() else f"\033[{code}m{text}\033[0m"


# ----- finding a box -------------------------------------------------------------------


def live_boxes(sail: Any) -> list[Any]:
    return [b for b in sail.Sailbox.list() if "terminated" not in str(b.status).lower()]


def describe(b: Any) -> str:
    return f"{b.name!s:<28} {b.status!s:<10} {b.sailbox_id}"


def resolve(sail: Any, wanted: str) -> Any:
    """A sailbox id, an id prefix, or a name. Ambiguity is reported, never guessed."""
    with contextlib.suppress(Exception):
        return sail.Sailbox.get(wanted)
    boxes = live_boxes(sail)
    matches = [b for b in boxes if str(b.name) == wanted or str(b.sailbox_id).startswith(wanted)]
    if not matches:
        raise SystemExit(
            f"no live box matches {wanted!r}. Live boxes:\n"
            + "\n".join("  " + describe(b) for b in boxes)
        )
    if len(matches) > 1:
        raise SystemExit(
            f"{wanted!r} matches {len(matches)} boxes; use the id:\n"
            + "\n".join("  " + describe(b) for b in matches)
        )
    return matches[0]


def control_box_id() -> str:
    try:
        return ctl.load_control().box_id
    except Exception as e:
        raise SystemExit(
            f"no --box given and no control box recorded in {ctl.CONTROL_RECORD}: {e}"
        ) from e


# ----- running one command -------------------------------------------------------------


def wrap(cmd: str, cwd: str) -> str:
    """Run ``cmd`` in ``cwd`` and report where it ended up, so ``cd`` sticks."""
    return (
        f"cd {shlex.quote(cwd)} 2>/dev/null || cd /\n"
        f"{cmd}\n"
        f"__rc=$?\n"
        f'printf "\\n{MARKER}%s\\n" "$PWD"\n'
        f"exit $__rc\n"
    )


def split_pwd(stdout: str, cwd: str) -> tuple[str, str]:
    """Peel the trailing marker line off the output and return the new directory."""
    lines = stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(MARKER):
            new = lines[i][len(MARKER) :].strip() or cwd
            del lines[i]
            return "\n".join(lines), new
    return stdout, cwd


class Session:
    def __init__(self, sail: Any, sb: Any, cwd: str, timeout: int) -> None:
        self.sail = sail
        self.box = SailBox(sb)
        self.name = str(sb.name)
        self.cwd = cwd
        self.timeout = timeout

    def run(self, cmd: str) -> None:
        try:
            r = self.box.run(wrap(cmd, self.cwd), timeout=self.timeout)
        except BoxError as e:
            print(colour(str(e), "31"))
            return
        out, self.cwd = split_pwd(r.stdout, self.cwd)
        if out.strip():
            print(out.rstrip("\n"))
        if r.stderr.strip():
            print(colour(r.stderr.rstrip("\n"), "33"))
        if r.timed_out:
            print(colour(f"[timed out after {self.timeout}s]", "31"))
        elif r.exit_code != 0:
            print(colour(f"[exit {r.exit_code}]", "31"))

    def follow(self, cmd: str) -> None:
        """Stream a long running command. Ctrl-C stops watching, not the command."""
        log = f"/tmp/inspect_{uuid.uuid4().hex[:8]}.log"
        done = "__inspect_done__"
        try:
            self.box.start(
                f"cd {shlex.quote(self.cwd)}; ({cmd}) > {log} 2>&1; echo {done}$? >> {log}"
            )
        except BoxError as e:
            print(colour(str(e), "31"))
            return
        offset = 0
        print(colour(f"following, Ctrl-C to stop watching ({log})", "90"))
        try:
            while True:
                try:
                    r = self.box.run(f"tail -c +{offset + 1} {log} 2>/dev/null", timeout=60)
                except BoxError as e:
                    print(colour(str(e), "31"))
                    return
                chunk = r.stdout
                if chunk:
                    offset += len(chunk.encode())
                    if done in chunk:
                        head, _, tail = chunk.partition(done)
                        if head.strip():
                            print(head.rstrip("\n"))
                        print(colour(f"[exit {tail.strip() or '?'}]", "90"))
                        return
                    print(chunk.rstrip("\n"))
                time.sleep(1.0)
        except KeyboardInterrupt:
            print(colour(f"\nstopped watching; it is still running, output in {log}", "90"))

    def get(self, remote: str, local: str) -> None:
        target = Path(local)
        if target.is_dir():
            target = target / Path(remote).name
        try:
            data = self.box.read(remote if remote.startswith("/") else f"{self.cwd}/{remote}")
        except BoxError as e:
            print(colour(str(e), "31"))
            return
        target.write_bytes(data)
        print(colour(f"{len(data):,} bytes to {target}", "90"))


# ----- the loop --------------------------------------------------------------------------


def meta(session: Session, line: str) -> None:
    """Handle a colon command."""
    head, _, rest = line[1:].partition(" ")
    rest = rest.strip()
    if head in ("help", "h", "?"):
        print(__doc__)
    elif head == "boxes":
        for b in live_boxes(session.sail):
            mark = " *" if str(b.sailbox_id) == session.box.box_id else ""
            print("  " + describe(b) + mark)
    elif head == "box":
        if not rest:
            print(colour("usage: :box <id or name>", "31"))
            return
        try:
            sb = resolve(session.sail, rest)
        except SystemExit as e:
            print(colour(str(e), "31"))
            return
        session.box = SailBox(sb)
        session.name = str(sb.name)
        session.cwd = "/"
        print(colour(f"now on {session.name} ({session.box.box_id})", "90"))
    elif head == "follow":
        if rest:
            session.follow(rest)
        else:
            print(colour("usage: :follow <command>", "31"))
    elif head == "get":
        parts = shlex.split(rest)
        if not parts:
            print(colour("usage: :get <remote> [local]", "31"))
        else:
            session.get(parts[0], parts[1] if len(parts) > 1 else ".")
    elif head == "timeout":
        if rest.isdigit():
            session.timeout = int(rest)
        print(colour(f"timeout {session.timeout}s", "90"))
    elif head == "pwd":
        print(session.cwd)
    else:
        print(colour(f"unknown command :{head}; try :help", "31"))


def confirm(cmd: str) -> bool:
    print(colour("that looks destructive, and this box holds live run data:", "31"))
    print("  " + cmd)
    try:
        return input("run it? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def load_history() -> None:
    with contextlib.suppress(ImportError, OSError):
        import readline

        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        readline.set_history_length(2000)
        if HISTORY.exists():
            readline.read_history_file(HISTORY)


def save_history() -> None:
    with contextlib.suppress(ImportError, OSError):
        import readline

        readline.write_history_file(HISTORY)


def repl(session: Session) -> None:
    load_history()
    print(colour(f"{session.name}  {session.box.box_id}", "1"))
    print(colour("shell commands run as root on the box. :help for more, Q to quit.", "90"))
    while True:
        try:
            line = input(colour(f"{session.name}:{session.cwd}$ ", "36")).strip()
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in QUIT or line == "Q":
            break
        if line.startswith(":"):
            meta(session, line)
            continue
        if DESTRUCTIVE.search(line) and not confirm(line):
            continue
        session.run(line)

    save_history()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--box", default="", help="sailbox id, id prefix, or name")
    p.add_argument("--list", action="store_true", help="print live boxes and exit")
    p.add_argument("--cwd", default="/", help="where the shell starts")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per command")
    p.add_argument("-c", "--command", default="", help="run one command and exit, no repl")
    args = p.parse_args(argv)

    env.load_dotenv()
    import sail

    if args.list:
        for b in live_boxes(sail):
            print(describe(b))
        return 0

    sb = resolve(sail, args.box or control_box_id())
    session = Session(sail, sb, args.cwd, args.timeout)
    if args.command:
        session.run(args.command)
        return 0
    repl(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
