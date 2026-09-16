# speed up checker dispatch and `_is_eol_token`

`Checker.check_physical` and `Checker.check_logical` run the registered checks on each physical and logical line. The existing loop called `init_checker_state` for every check even though only `module_imports_on_top_of_file` in `pycodestyle.py` uses `checker_state`, and then called `run_check`, which built an argument list with one `getattr` per entry in the check's `argument_names`. `_is_eol_token` had a similar repeated cost: for every token not in `NEWLINE` it sliced and stripped `token[4]` to look for an explicit line join.

`_compile_checks` now builds the dispatch tuples once per `StyleGuide`. Each tuple holds the check, an `operator.attrgetter`, a flag for a single positional argument, and a flag for `checker_state`. `StyleGuide.__init__` stores those tuples as `options.physical_check_runs` and `options.logical_check_runs`. A `Checker` reuses them when they exist and otherwise compiles from the original `options.physical_checks` and `options.logical_checks`, so manually constructed options still work. The existing `self._physical_checks`, `self._logical_checks`, `run_check`, and `init_checker_state` remain in place. The hot loops now call `check(get_args(self))` for single argument checks and `check(*get_args(self))` otherwise, and they reset `self.checker_state` only when the precomputed flag requires it. `_is_eol_token` now returns False as soon as the token line does not end in a backslash followed by a newline, so the slice and `lstrip` run only on backslash continued lines.

Measured against commit `d6c38543a95c3adec7958fc659f4081972a3f1d1`. Each input was run 6 times on the base and 6 times with this change; the numbers are derived from the medians.

| input | base | proposed improvement | speedup |
|---|---|---|---|
| flat_assignments | 563.0 ms | 499.1 ms | 1.13x |
| top_level_defs | 458.5 ms | 399.6 ms | 1.15x |
| method_bodies | 553.7 ms | 470.6 ms | 1.17x |
| multiline_bracketed | 229.7 ms | 212.1 ms | 1.07x |
| long_docstrings | 268.7 ms | 233.5 ms | 1.15x |
| string_literals | 391.4 ms | 345.2 ms | 1.13x |
