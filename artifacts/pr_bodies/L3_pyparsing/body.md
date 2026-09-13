When `MatchFirst` or `Or` holds a long list of plain `Literal` alternatives, each token position walks the full alternative list. `MatchFirst.parseImpl` calls `_parse` on each alternative; a nonmatching `Literal` still runs its own `preParse`, fails its first character test, raises `ParseException`, and has that exception caught so the next alternative can be tried. `Or.parseImpl` does the same scan through `try_parse`, and before the scan it also evaluates `all(e.callPreparse for e in self.exprs)` on every call, even when every alternative is a plain `Literal`. For the many alternative inputs this repeated exception raising and reattempt work dominates the parse.

### The alternative lookup

`streamline()` now calls `_make_literal_match_map()` for `MatchFirst` and `Or`. The map is installed only when there are 8 or more alternatives and every one is exactly `Literal` with the same nonzero `matchLen`, no `parseAction`, no `resultsName`, no `debug`, no `failAction`, no `callDuringTry`, no `ignoreExprs`, and `skipWhitespace` and `whiteChars` equal to the containing expression. That set is deliberately narrow: alternatives with parse actions, results names, debug hooks, ignore expressions, or different whitespace behavior still take the existing loop unchanged. `_make_literal_match_map` stores each expression under its `match` string.

On a match, `parseImpl` now does one `preParse`, one slice of `_literal_match_len` characters, one dictionary lookup, and a direct `_parse(..., callPreParse=False)` on the matching expression. `Or` caches the `all(e.callPreparse ...)` test as `_all_call_preparse`, so a hit no longer scans the whole alternative list just to decide whether to skip whitespace. A missing slice falls through to the original loop, which still sees every alternative. `ParseExpression.copy()` reruns `_make_literal_match_map()` after copying the `exprs` list, so a copied collection gets a lookup for its own alternatives instead of inheriting a stale one.

### The per token overhead

The other inputs improve because of three small changes on the ordinary parse path. `ParseResults.__init__` called `deprecate_argument` on every construction to look for the legacy `asList` spelling in `kwargs`; the constructor now takes `asList` with a sentinel default and only warns when a caller actually supplies it. `ParserElement._parseNoCache` always called `postParse`, although most token classes inherit the base no operation `return tokenlist`; it now calls `postParse` only when the concrete class overrides it. `_parseNoCache` also always constructed a `ParseResults` wrapper, but `ParseResults.__new__` already returns an existing `ParseResults` unchanged, so when there is no results name and no parse action it now returns the existing object directly. Finally `_parseCache` captures `self.debug` once and reuses that value for the try, fail, and match hooks on a packrat hit.

Measured against commit `<efd56db4e59f36b1673ce0eb0823e3afaa9d1201>`. Each input was run 6 times on the base and 6 times with this change; the numbers are derived from the medians.

| input | base | proposed improvement | speedup |
| `flat_csv_words` | 221.8 ms | 170.6 ms | 1.31x |
| `matchfirst_many_alts` | 131.7 ms | 2.5 ms | 52.95x |
| `or_many_alts` | 634.3 ms | 2.9 ms | 224.78x |
| `infix_recursive` | 96.1 ms | 89.4 ms | 1.08x |
| `left_recursive` | 27.0 ms | 22.0 ms | 1.22x |
