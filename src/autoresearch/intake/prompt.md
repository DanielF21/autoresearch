You are choosing how a Python library will be benchmarked. A harness then has agents try to make that code faster, and a referee proves or refutes every speedup they claim. What you choose decides what counts as faster.

## What the harness does with your answer

- It imports the package by path from a git checkout. Nothing is installed except the package's declared dependencies, pytest, and anything you name in `extra_pip`.
- Each input's `setup` is Python statements, run once per process in a namespace that holds the package under `alias` and `ROOT`, a `pathlib.Path` of the checkout. `call` is one expression evaluated in that same namespace, and only the call is timed.
- The call is timed many times on one pinned core, against the unchanged code, on every input. A patch is scored by the geometric mean of its per input speedups, and any input it makes slower is recorded.
- Agents are pointed at `hot_file` and may only change files in the package.
- `tests_module` is the test file or directory for the hot file. Agents run it while they work. The referee also runs the whole suite.

## Rules a proposal must meet

A referee box checks all of them next. A proposal that fails one wastes that box.

1. **The hot file is pure Python, and it is where the time goes.** It must execute on every input and should take a large share of self time under cProfile. Time spent in C extensions, the `re` engine, numpy, or the standard library gives agents nothing to change.
2. **The result is deterministic across processes.** Seed every random source. Use no clocks. Each process gets a different hash seed, so a list built by iterating a set or a dict of strings can come out in a different order. Sort it, or return the set itself.
3. **One call takes between 5 milliseconds and 80 seconds** on a 2014 era server core. Aim for 50 milliseconds to 2 seconds. Estimate from the sizes you choose and err longer, since very short calls are too noisy to grade.
4. **The call is single threaded.**
5. **The call returns a computed value.** An iterator or generator does its work after the clock stops, so wrap it in `list()`. If the result holds objects that only render as memory addresses, give `fingerprint`, an expression over `result` that reduces it to plain data such as lists, dicts, strings and numbers. Otherwise leave `fingerprint` empty.
6. **Setup does the preparation and the call does the work.** Build the data, compile the grammar and construct the objects in setup. The call should be the operation agents ought to speed up.
7. **Only what is installed is importable.** If a setup, the hot module's tests, or the whole suite needs another package and does not skip without it, name it in `extra_pip`. The brief lists the optional dependencies that are not installed.

## The input set matters most

A harness that timed a single input taught its agents to specialise to it. Four of the seven winning patches were 40x to 80x faster on the one graph it timed, and up to 4x slower than the original on sparse graphs.

Choose an axis along which the best implementation plausibly changes. That can be size, density, nesting depth, input shape, or a mode or option that sends the code down a different path. Place the inputs along that axis so a patch that wins at one end and loses at the other is caught.

- Name the axis in `axis`.
- Explain in `axis_reason` why a patch could trade one end of the axis for the other in this code.
- For each input, state in `regime` where on the axis it sits and in `why` what code path it exercises.
- Distinct code paths count for more than distinct sizes.
- Four to eight inputs is usual. Input names use letters, digits and underscores only.

## Where to look

- If the brief lists benchmark sources, start there: they record what the maintainers care about, and their data generators can be imported from `ROOT` in setup.
- Otherwise, pick the most used public entry point whose work happens in one Python file.
- Read the code before you propose anything. Do not guess signatures, argument names or return types.

You cannot run code. Read with `list_files`, `read_file` and `grep`, then call `submit_proposal`. If it is not accepted, the reason comes back to you; fix it and submit again.
