# Improve MatchFirst performance for single character literal alternatives

`MatchFirst.parseImpl` handles alternation by walking `self.exprs` and calling `e._parse` on each expression until one returns. In the four timed grammars, `addsub` and `muldiv` are both built as `Literal("+") | Literal("-")` and `Literal("*") | Literal("/")`, so this walk runs once per operator in the parsed text. The generic per alternative cost is `_parseNoCache` itself: it calls `preParse`, calls `parseImpl`, and on success constructs a `ParseResults`, while a failing single character alternative raises a `ParseException` that `MatchFirst.parseImpl` catches so it can track `maxExcLoc`. At high operator counts the failed alternative and its exception dominate the total.

This change adds a guarded fast path for the exact shape the timed grammars use: two `_SingleCharLiteral` alternatives with no parse action, result name, ignore expression, debug flag, or fail action, and with the same whitespace settings as the `MatchFirst`. It calls `preParse` once, compares `instring[pre_loc]` with the two `firstMatchChar` values, and returns the matched token or raises the failure directly. The outer `_parseNoCache` call for the `MatchFirst` then builds the final `ParseResults`, which is the same result object the generic alternative path produces because the alternatives have no result names or parse actions. The rest of the method, including the error selection logic and the generic loop, is untouched, so any larger alternation or any decorated literal continues through the existing path.

Measured against commit `efd56db4e59f36b1673ce0eb0823e3afaa9d1201`. Each input was run 6 times on the base and 6 times with this change; the numbers are derived from the medians.

| input | base | proposed improvement | speedup |
|---|---|---|---|
| `flat_long` | 88.5 ms | 62.4 ms | 1.42x |
| `shallow_groups` | 123.6 ms | 88.8 ms | 1.39x |
| `balanced_bush` | 186.2 ms | 137.7 ms | 1.36x |
| `deep_forest` | 201.2 ms | 149.1 ms | 1.35x |
