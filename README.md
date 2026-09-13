# autoresearch

An agent harness that makes CPU bound Python faster and proves it. Built for the Sail
Research agent engineering project. Workers propose patches to a target repository from
inside sailboxes, a referee measures every patch against the same original commit on its
own sailbox, and every measurement is appended to a history that the next workers read.

The target is described entirely by the `[target]` section of a run's config. The
harness knows nothing about any particular repository; the first target was networkx,
and `artifacts/` is the measurement record from building against it.

## Layout

| Path | What |
|---|---|
| `src/autoresearch/` | The harness. Host code that runs on the laptop or the control box. |
| `src/autoresearch/guest/` | Standard library only programs uploaded into sailboxes. |
| `tests/` | Unit tests against fakes. No network. |
| `configs/` | One TOML file per run. |
| `phase1/` | The measurement scripts that produced `artifacts/`. Not part of the harness. |
| `artifacts/` | `repo.md`, `measurements.md` and `generality.md`, the networkx record. |

## Develop

```
uv sync
uv run pre-commit install
scripts/check.sh        # ruff format, ruff lint, mypy strict, pytest
```

`SAIL_API_KEY` is read from `.env`, which is not committed.

## Nothing left running

Every box is created with auto sleep off, so a box nobody terminates bills until
someone does. The harness terminates each one when the process holding it exits,
however it exits:

- `run` terminates its referees on every exit: the last round, `--until`, an
  exception, Ctrl+C, SIGTERM or SIGHUP. A resumed run builds new ones, and first
  terminates any a killed run left named in `boxes.json`.
- A worker terminates its own box at the end of every attempt.
- `check`, `profile`, `calibrate` and `measure` terminate their box unless `--keep`.
- A launched run is followed by `release-control`, which terminates the control
  box once no run is going on it, unless launched with `--keep-control`. `fetch`
  brings up a temporary box on the volume when the control box is gone.

No code in a process survives `kill -9`, a dead laptop or a lost control box. After
any of those:

```
uv run autoresearch reap          # every live box in the app
uv run autoresearch reap --yes    # terminate them all, including any run still going
```

## Adding a target

A target is a `[target]` section. Every fact the harness needs is in it:

```toml
[target]
name = "..."
repo = "https://github.com/org/repo"
sha = "<pinned commit>"
package = "pkg"                # import name; imported by path from the tree, never installed
alias = "p"                    # the name the package is bound to in setup and call
package_root = "."             # directory put on sys.path, relative to the repo: "src" for a src layout
hot_file = "pkg/hot.py"        # where the worker is pointed; the referee checks it ran
call = "p.compute(x)"          # the expression that is timed
pip = []                       # dependencies on top of pytest and pytest-xdist
apt = []                       # Debian packages on top of git, curl, build-essential, time, util-linux
allow = ["pkg/**"]             # what a patch may touch
deny = ["tests/**"]

[target.tests]
module = "tests/test_hot.py"   # the hot module's own tests; the worker runs these
full = "tests"                 # the whole suite; the referee runs it on every submission

[[target.inputs]]
name = "small"
setup = "x = p.make(100)"      # statements run once per launch, with the alias and ROOT bound
                               # noise_floor is absent until autoresearch calibrate has measured it
```

`setup` runs in a namespace holding the package under `alias` and `ROOT`, a `pathlib.Path`
of the tree under test, so an input can read a file from the repository. `call` is
evaluated in that namespace and timed. Its result is fingerprinted so both trees can be
shown to compute the same thing; a target whose result the generic fingerprint cannot
render can give a `fingerprint` expression over `result`.

What the harness can measure, which follow from how the referee works rather than from
any target:

1. **A pure Python hot path.** cProfile self time is attributed to `hot_file`, and the
   referee refuses to time an input on which it did not execute. A C extension never
   shows as executed.
2. **Import by path.** `package_root` on `sys.path` must make `import <package>` resolve
   inside the checkout. Nothing is installed.
3. **A single threaded call.** Every timing launch is pinned to one core, so a call that
   spreads work across threads is measured serialised and a patch that adds parallelism
   measures as no faster.
4. **A deterministic result.** Two launches of the same tree must fingerprint the same.
5. **A call between about 5ms and 80s.** Below that a calibrated floor is too loose to
   grade on; above it `repeats_per_launch` calls do not fit inside one launch timeout.
6. **Both suites pass on the base commit and finish inside the referee's timeout**, 1200
   seconds with four xdist workers for the full suite.
7. **Dependencies installable by pip or apt on Debian.**

The order for a new target. Each step is its own command, so each cost is its own
decision, and nothing runs the next step for you:

```
uv run autoresearch intake <github url>                  # free: clone, judge scope, draft metadata and a brief
uv run autoresearch intake propose runs/intake/<name>    # one model conversation: call, inputs, config
uv run autoresearch check configs/<run_id>.toml          # one referee box, no model: judges the rules above
uv run autoresearch profile configs/<run_id>.toml        # one referee box, no model: the worker's documents
uv run autoresearch calibrate configs/<run_id>.toml      # one referee box, no model: the noise floors
uv run autoresearch run configs/<run_id>.toml --until 1
uv run autoresearch next configs/<run_id>.toml           # free: which step a config is at, what the next creates
```

`intake` never runs the repository's code: it reads files, and refuses a repository that
is not a Python project, has a compiled build, has no package at `.` or `src/`, or has no
tests. `intake propose` gives the model read only tools over the clone and validates its
answer without executing it; the config is written only once it parses back exactly.
Read `runs/intake/<name>/brief.md` before the go for `propose`, and the proposed inputs
before the go for `check`: whether they span the regimes a patch can trade between is a
judgment no command makes.

`check` surveys the base tree, prints every rule's verdict and refuses the target if one
fails. `profile` and `calibrate` write `docs` and each input's `noise_floor` back into
the config, and write nothing if the file changed while their box ran. `run` refuses a
config with any input that has no floor.
