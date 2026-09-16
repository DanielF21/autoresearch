"""AlphaEvolve, built from the paper, measured by the harness's referee.

Novikov et al. 2025, "AlphaEvolve: A coding agent for scientific and algorithmic
discovery" (arXiv 2506.13131). This package exists to put AlphaEvolve's search
beside the harness on one plot: cumulative tokens spent against the best held
out speedup so far, on the same targets, the same model and the same referee.

What is built, each a component the paper's ablations name:
- an evolutionary program database of islands and a MAP Elites grid (``database``)
- rich context in the prompt: the target, the profile documents and the source
  (``prompt``)
- meta prompt evolution: instructions written by the model and selected by the
  programs they produced (``meta``)
- evolution of several blocks across files, marked from the profile (``blocks``)

What is left out, and why:
- the model ensemble: one model, the harness's own, so the curves differ by
  search method alone
- the evaluation cascade: evaluation cost is not on the x axis, and the referee
  must measure both arms identically
- scores graded by a model: the y axis is the referee's number alone
- the asynchronous controller: candidates come in synchronous batches of
  ``[run].width``, so the records are the harness's rounds and resume like them

The paper publishes no database parameters. The mechanics and every value are
OpenEvolve's (codelion/openevolve at 411fb59c886c18704caaffb611e17cf9e7d824d2),
cited in ``database`` and in each config's ``[evolve]`` section. The meta prompt
mechanics are specified by neither and are this package's own; ``meta`` says so.

A candidate is an ordinary attempt in an ordinary run directory, written by the
harness's round loop through ``worker.EvolveWorker``, so ``autoresearch status``
reads it and ``curve`` draws both arms from the same records.
"""
